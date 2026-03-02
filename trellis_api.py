import os
import sys
import gc
import io
import uuid
import shutil
import asyncio
import tempfile
import time
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'model_refs', 'TRELLIS.2'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'model_refs', 'TRELLIS.2', 'o-voxel'))

import torch
import httpx
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel, field_validator
from typing import Optional

VALID_RESOLUTIONS = {'512', '1024', '1536'}

from trellis2.pipelines import Trellis2ImageTo3DPipeline, Trellis2TexturingPipeline
import o_voxel

# ── Global state ──────────────────────────────────────────────────────────────

pipeline: Trellis2ImageTo3DPipeline = None
tex_pipeline: Trellis2TexturingPipeline = None
jobs: dict[str, dict] = {}
_executor = ThreadPoolExecutor(max_workers=1)
OUTPUT_DIR = tempfile.mkdtemp()
QWEN_API_URL = os.getenv('QWEN_API_URL', 'http://localhost:8190')

print(f"📂 TRELLIS Output directory: {OUTPUT_DIR}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _update_job(job_id: str, percent: int, stage: str, **kwargs):
    jobs[job_id].update({'percent': percent, 'stage': stage, **kwargs})


def _make_job() -> str:
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        'status': 'pending',
        'percent': 0,
        'stage': 'Queued',
        'result': None,
        'error': None,
        'created_at': time.time(),
    }
    return job_id


def _parse_params(
    resolution: str,
    seed: int,
    preprocess_image: bool,
    decimation_target: int,
    texture_size: int,
    ss_guidance_strength: float,
    ss_guidance_rescale: float,
    ss_sampling_steps: int,
    ss_rescale_t: float,
    shapslat_guidance_strength: float,
    shapslat_guidance_rescale: float,
    shapslat_sampling_steps: int,
    shapslat_rescale_t: float,
    texslat_guidance_strength: float,
    texslat_guidance_rescale: float,
    texslat_sampling_steps: int,
    texslat_rescale_t: float,
) -> dict:
    return dict(
        resolution=resolution,
        seed=seed,
        preprocess_image=preprocess_image,
        decimation_target=decimation_target,
        texture_size=texture_size,
        ss_guidance_strength=ss_guidance_strength,
        ss_guidance_rescale=ss_guidance_rescale,
        ss_sampling_steps=ss_sampling_steps,
        ss_rescale_t=ss_rescale_t,
        shapslat_guidance_strength=shapslat_guidance_strength,
        shapslat_guidance_rescale=shapslat_guidance_rescale,
        shapslat_sampling_steps=shapslat_sampling_steps,
        shapslat_rescale_t=shapslat_rescale_t,
        texslat_guidance_strength=texslat_guidance_strength,
        texslat_guidance_rescale=texslat_guidance_rescale,
        texslat_sampling_steps=texslat_sampling_steps,
        texslat_rescale_t=texslat_rescale_t,
    )


# ── Core inference ─────────────────────────────────────────────────────────────

def _run_image_inference(job_id: str, pil_image: Image.Image, params: dict):
    try:
        jobs[job_id]['status'] = 'running'
        _update_job(job_id, 5, 'Preprocessing image...')

        resolution = params['resolution']
        pipeline_type_map = {'512': '512', '1024': '1024_cascade', '1536': '1536_cascade'}
        pipeline_type = pipeline_type_map.get(resolution, '1024_cascade')

        _update_job(job_id, 10, 'Stage 1: Generating sparse structure...')

        results = pipeline.run(
            pil_image,
            seed=params['seed'],
            preprocess_image=params['preprocess_image'],
            sparse_structure_sampler_params={
                'steps': params['ss_sampling_steps'],
                'guidance_strength': params['ss_guidance_strength'],
                'guidance_rescale': params['ss_guidance_rescale'],
                'rescale_t': params['ss_rescale_t'],
            },
            shape_slat_sampler_params={
                'steps': params['shapslat_sampling_steps'],
                'guidance_strength': params['shapslat_guidance_strength'],
                'guidance_rescale': params['shapslat_guidance_rescale'],
                'rescale_t': params['shapslat_rescale_t'],
            },
            tex_slat_sampler_params={
                'steps': params['texslat_sampling_steps'],
                'guidance_strength': params['texslat_guidance_strength'],
                'guidance_rescale': params['texslat_guidance_rescale'],
                'rescale_t': params['texslat_rescale_t'],
            },
            pipeline_type=pipeline_type,
        )
        mesh = results[0]

        _update_job(job_id, 40, 'Stage 2: Generating 3D shape...')
        _update_job(job_id, 70, 'Stage 3: Generating materials...')

        # nvdiffrast triangle limit — required
        mesh.simplify(16777216)

        _update_job(job_id, 88, 'Extracting GLB...')

        job_dir = os.path.join(OUTPUT_DIR, job_id)
        os.makedirs(job_dir, exist_ok=True)
        glb_path = os.path.join(job_dir, 'output.glb')

        glb = o_voxel.postprocess.to_glb(
            vertices=mesh.vertices,
            faces=mesh.faces,
            attr_volume=mesh.attrs,
            coords=mesh.coords,
            attr_layout=mesh.layout,
            voxel_size=mesh.voxel_size,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=params['decimation_target'],
            texture_size=params['texture_size'],
            remesh=True,
            remesh_band=1,
            remesh_project=0,
            verbose=False,
        )
        glb.export(glb_path, extension_webp=True)

        _update_job(job_id, 100, 'Done ✓')
        jobs[job_id]['status'] = 'completed'
        jobs[job_id]['result'] = {'glb_url': f'/download/{job_id}/output.glb'}

    except Exception as e:
        print(f"❌ TRELLIS inference error [{job_id}]: {e}")
        jobs[job_id]['status'] = 'failed'
        jobs[job_id]['error'] = str(e)
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _run_text_job(job_id: str, qwen_payload: dict, params: dict):
    try:
        jobs[job_id]['status'] = 'running'
        _update_job(job_id, 0, 'Sending prompt to Qwen...')

        with httpx.Client(timeout=300) as client:
            resp = client.post(f'{QWEN_API_URL}/text2img', json=qwen_payload)
            resp.raise_for_status()
            data = resp.json()

        _update_job(job_id, 8, 'Image ready — starting TRELLIS.2...')

        img_url = QWEN_API_URL + data['url']
        with httpx.Client(timeout=60) as client:
            img_resp = client.get(img_url)
            img_resp.raise_for_status()

        pil_image = Image.open(io.BytesIO(img_resp.content)).convert('RGB')
        _run_image_inference(job_id, pil_image, params)

    except Exception as e:
        print(f"❌ TRELLIS text-mode error [{job_id}]: {e}")
        jobs[job_id]['status'] = 'failed'
        jobs[job_id]['error'] = str(e)


def _run_multiview_job(job_id: str, pil_images: list, params: dict, multiview_mode: str):
    try:
        jobs[job_id]['status'] = 'running'
        _update_job(job_id, 5, 'Preprocessing images...')

        resolution = params['resolution']
        pipeline_type_map = {'512': '512', '1024': '1024_cascade', '1536': '1536_cascade'}
        pipeline_type = pipeline_type_map.get(resolution, '1024_cascade')

        _update_job(job_id, 10, 'Stage 1: Generating sparse structure (multi-view)...')

        results = pipeline.run_multi_image(
            pil_images,
            seed=params['seed'],
            mode=multiview_mode,
            pipeline_type=pipeline_type,
            sparse_structure_sampler_params={
                'steps': params['ss_sampling_steps'],
                'guidance_strength': params['ss_guidance_strength'],
                'guidance_rescale': params['ss_guidance_rescale'],
                'rescale_t': params['ss_rescale_t'],
            },
            shape_slat_sampler_params={
                'steps': params['shapslat_sampling_steps'],
                'guidance_strength': params['shapslat_guidance_strength'],
                'guidance_rescale': params['shapslat_guidance_rescale'],
                'rescale_t': params['shapslat_rescale_t'],
            },
            tex_slat_sampler_params={
                'steps': params['texslat_sampling_steps'],
                'guidance_strength': params['texslat_guidance_strength'],
                'guidance_rescale': params['texslat_guidance_rescale'],
                'rescale_t': params['texslat_rescale_t'],
            },
        )
        mesh = results[0]

        _update_job(job_id, 70, 'Stage 3: Generating materials...')
        mesh.simplify(16777216)
        _update_job(job_id, 88, 'Extracting GLB...')

        job_dir = os.path.join(OUTPUT_DIR, job_id)
        os.makedirs(job_dir, exist_ok=True)
        glb_path = os.path.join(job_dir, 'output.glb')

        glb = o_voxel.postprocess.to_glb(
            vertices=mesh.vertices,
            faces=mesh.faces,
            attr_volume=mesh.attrs,
            coords=mesh.coords,
            attr_layout=mesh.layout,
            voxel_size=mesh.voxel_size,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=params['decimation_target'],
            texture_size=params['texture_size'],
            remesh=True,
            remesh_band=1,
            remesh_project=0,
            verbose=False,
        )
        glb.export(glb_path, extension_webp=True)

        _update_job(job_id, 100, 'Done ✓')
        jobs[job_id]['status'] = 'completed'
        jobs[job_id]['result'] = {'glb_url': f'/download/{job_id}/output.glb'}

    except Exception as e:
        print(f"❌ TRELLIS multiview error [{job_id}]: {e}")
        jobs[job_id]['status'] = 'failed'
        jobs[job_id]['error'] = str(e)
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _run_texture_job(job_id: str, mesh_path: str, ref_image_path: str, params: dict):
    try:
        jobs[job_id]['status'] = 'running'
        _update_job(job_id, 5, 'Loading mesh...')

        ref_image = Image.open(ref_image_path).convert('RGB')
        _update_job(job_id, 20, 'Stage 3: Generating material textures...')

        result = tex_pipeline.run(
            mesh_path,
            ref_image,
            seed=params['seed'],
            tex_slat_sampler_params={
                'steps': params['texslat_sampling_steps'],
                'guidance_strength': params['texslat_guidance_strength'],
                'guidance_rescale': params['texslat_guidance_rescale'],
                'rescale_t': params['texslat_rescale_t'],
            },
            texture_size=params['texture_size'],
        )

        _update_job(job_id, 88, 'Exporting GLB...')
        job_dir = os.path.join(OUTPUT_DIR, job_id)
        os.makedirs(job_dir, exist_ok=True)
        glb_path = os.path.join(job_dir, 'output.glb')
        result.export(glb_path, extension_webp=True)

        _update_job(job_id, 100, 'Done ✓')
        jobs[job_id]['status'] = 'completed'
        jobs[job_id]['result'] = {'glb_url': f'/download/{job_id}/output.glb'}

    except Exception as e:
        print(f"❌ TRELLIS texture error [{job_id}]: {e}")
        jobs[job_id]['status'] = 'failed'
        jobs[job_id]['error'] = str(e)
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ── Cleanup task ───────────────────────────────────────────────────────────────

async def _cleanup_loop():
    while True:
        await asyncio.sleep(30 * 60)
        cutoff = time.time() - 60 * 60
        for job_id, job in list(jobs.items()):
            if job.get('created_at', 0) < cutoff:
                job_dir = os.path.join(OUTPUT_DIR, job_id)
                shutil.rmtree(job_dir, ignore_errors=True)
                jobs.pop(job_id, None)


# ── App lifespan ───────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline, tex_pipeline
    print("⏳ Loading TRELLIS.2 pipelines...")
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained('microsoft/TRELLIS.2-4B')
    pipeline.cuda()
    tex_pipeline = Trellis2TexturingPipeline.from_pretrained('microsoft/TRELLIS.2-4B')
    tex_pipeline.cuda()
    print("✅ TRELLIS.2 pipelines loaded")
    cleanup_task = asyncio.create_task(_cleanup_loop())
    yield
    cleanup_task.cancel()
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)


app = FastAPI(title='TRELLIS.2 API', lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get('/health')
async def health_check():
    return {
        'status': 'ok',
        'pipelines_loaded': pipeline is not None and tex_pipeline is not None,
        'active_jobs': len(jobs),
    }


@app.post('/generate/image')
async def generate_from_image(
    file: UploadFile = File(...),
    resolution: str = Form('1024', description="One of: '512', '1024', '1536'"),
    seed: int = Form(42),
    preprocess_image: bool = Form(True),
    decimation_target: int = Form(200000),
    texture_size: int = Form(1024),
    ss_guidance_strength: float = Form(7.5),
    ss_guidance_rescale: float = Form(0.7),
    ss_sampling_steps: int = Form(12),
    ss_rescale_t: float = Form(0.5),
    shapslat_guidance_strength: float = Form(3.0),
    shapslat_guidance_rescale: float = Form(0.0),
    shapslat_sampling_steps: int = Form(12),
    shapslat_rescale_t: float = Form(0.5),
    texslat_guidance_strength: float = Form(3.0),
    texslat_guidance_rescale: float = Form(0.0),
    texslat_sampling_steps: int = Form(12),
    texslat_rescale_t: float = Form(0.5),
):
    if resolution not in VALID_RESOLUTIONS:
        raise HTTPException(status_code=422, detail=f"Invalid resolution '{resolution}'. Must be one of: {sorted(VALID_RESOLUTIONS)}")

    job_id = _make_job()
    contents = await file.read()
    pil_image = Image.open(io.BytesIO(contents)).convert('RGB')

    params = _parse_params(
        resolution, seed, preprocess_image, decimation_target, texture_size,
        ss_guidance_strength, ss_guidance_rescale, ss_sampling_steps, ss_rescale_t,
        shapslat_guidance_strength, shapslat_guidance_rescale, shapslat_sampling_steps, shapslat_rescale_t,
        texslat_guidance_strength, texslat_guidance_rescale, texslat_sampling_steps, texslat_rescale_t,
    )
    _executor.submit(_run_image_inference, job_id, pil_image, params)
    return {'job_id': job_id, 'status': 'pending'}


VALID_MULTIVIEW_MODES = {'stochastic', 'multidiffusion'}


@app.post('/generate/multiview')
async def generate_from_multiview(
    files: list[UploadFile] = File(...),
    multiview_mode: str = Form('stochastic', description="One of: 'stochastic', 'multidiffusion'"),
    resolution: str = Form('1024', description="One of: '512', '1024', '1536'"),
    seed: int = Form(42),
    preprocess_image: bool = Form(True),
    decimation_target: int = Form(200000),
    texture_size: int = Form(1024),
    ss_guidance_strength: float = Form(7.5),
    ss_guidance_rescale: float = Form(0.7),
    ss_sampling_steps: int = Form(12),
    ss_rescale_t: float = Form(0.5),
    shapslat_guidance_strength: float = Form(3.0),
    shapslat_guidance_rescale: float = Form(0.0),
    shapslat_sampling_steps: int = Form(12),
    shapslat_rescale_t: float = Form(0.5),
    texslat_guidance_strength: float = Form(3.0),
    texslat_guidance_rescale: float = Form(0.0),
    texslat_sampling_steps: int = Form(12),
    texslat_rescale_t: float = Form(0.5),
):
    if resolution not in VALID_RESOLUTIONS:
        raise HTTPException(status_code=422, detail=f"Invalid resolution '{resolution}'. Must be one of: {sorted(VALID_RESOLUTIONS)}")
    if multiview_mode not in VALID_MULTIVIEW_MODES:
        raise HTTPException(status_code=422, detail=f"Invalid multiview_mode '{multiview_mode}'. Must be one of: {sorted(VALID_MULTIVIEW_MODES)}")

    job_id = _make_job()
    pil_images = []
    for f in files:
        contents = await f.read()
        pil_images.append(Image.open(io.BytesIO(contents)).convert('RGB'))

    params = _parse_params(
        resolution, seed, preprocess_image, decimation_target, texture_size,
        ss_guidance_strength, ss_guidance_rescale, ss_sampling_steps, ss_rescale_t,
        shapslat_guidance_strength, shapslat_guidance_rescale, shapslat_sampling_steps, shapslat_rescale_t,
        texslat_guidance_strength, texslat_guidance_rescale, texslat_sampling_steps, texslat_rescale_t,
    )
    _executor.submit(_run_multiview_job, job_id, pil_images, params, multiview_mode)
    return {'job_id': job_id, 'status': 'pending'}


class TrellisTextBody(BaseModel):
    qwen: dict
    trellis: Optional[dict] = None

    @field_validator('qwen')
    @classmethod
    def qwen_must_have_prompt(cls, v):
        if 'prompt' not in v or not str(v['prompt']).strip():
            raise ValueError("'qwen.prompt' is required and must not be empty")
        return v


@app.post('/generate/text')
async def generate_from_text(body: TrellisTextBody):
    job_id = _make_job()
    qwen_payload = body.qwen
    trellis_params_raw = body.trellis or {}

    resolution = trellis_params_raw.get('resolution', '1024')
    if resolution not in VALID_RESOLUTIONS:
        raise HTTPException(status_code=422, detail=f"Invalid trellis.resolution '{resolution}'. Must be one of: {sorted(VALID_RESOLUTIONS)}")

    params = _parse_params(
        resolution=trellis_params_raw.get('resolution', '1024'),
        seed=trellis_params_raw.get('seed', 42),
        preprocess_image=trellis_params_raw.get('preprocess_image', True),
        decimation_target=trellis_params_raw.get('decimation_target', 200000),
        texture_size=trellis_params_raw.get('texture_size', 1024),
        ss_guidance_strength=trellis_params_raw.get('ss_guidance_strength', 7.5),
        ss_guidance_rescale=trellis_params_raw.get('ss_guidance_rescale', 0.7),
        ss_sampling_steps=trellis_params_raw.get('ss_sampling_steps', 12),
        ss_rescale_t=trellis_params_raw.get('ss_rescale_t', 0.5),
        shapslat_guidance_strength=trellis_params_raw.get('shapslat_guidance_strength', 3.0),
        shapslat_guidance_rescale=trellis_params_raw.get('shapslat_guidance_rescale', 0.0),
        shapslat_sampling_steps=trellis_params_raw.get('shapslat_sampling_steps', 12),
        shapslat_rescale_t=trellis_params_raw.get('shapslat_rescale_t', 0.5),
        texslat_guidance_strength=trellis_params_raw.get('texslat_guidance_strength', 3.0),
        texslat_guidance_rescale=trellis_params_raw.get('texslat_guidance_rescale', 0.0),
        texslat_sampling_steps=trellis_params_raw.get('texslat_sampling_steps', 12),
        texslat_rescale_t=trellis_params_raw.get('texslat_rescale_t', 0.5),
    )
    _executor.submit(_run_text_job, job_id, qwen_payload, params)
    return {'job_id': job_id, 'status': 'pending'}


@app.post('/generate/batch')
async def generate_batch(files: list[UploadFile] = File(...)):
    job_ids = []
    for f in files:
        job_id = _make_job()
        contents = await f.read()
        pil_image = Image.open(io.BytesIO(contents)).convert('RGB')
        params = _parse_params('1024', 42, True, 200000, 1024, 7.5, 0.7, 12, 0.5, 3.0, 0.0, 12, 0.5, 3.0, 0.0, 12, 0.5)
        _executor.submit(_run_image_inference, job_id, pil_image, params)
        job_ids.append(job_id)
    return {'job_ids': job_ids, 'status': 'pending'}


@app.post('/texture')
async def texture_mesh(
    mesh: UploadFile = File(...),
    reference: UploadFile = File(...),
    seed: int = Form(42),
    texture_size: int = Form(1024),
    texslat_guidance_strength: float = Form(3.0),
    texslat_guidance_rescale: float = Form(0.0),
    texslat_sampling_steps: int = Form(12),
    texslat_rescale_t: float = Form(0.5),
):
    job_id = _make_job()
    job_dir = os.path.join(OUTPUT_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    mesh_path = os.path.join(job_dir, 'input.glb')
    ref_path = os.path.join(job_dir, 'reference.png')

    with open(mesh_path, 'wb') as f:
        shutil.copyfileobj(mesh.file, f)
    with open(ref_path, 'wb') as f:
        shutil.copyfileobj(reference.file, f)

    params = dict(
        seed=seed, texture_size=texture_size,
        texslat_guidance_strength=texslat_guidance_strength,
        texslat_guidance_rescale=texslat_guidance_rescale,
        texslat_sampling_steps=texslat_sampling_steps,
        texslat_rescale_t=texslat_rescale_t,
    )
    _executor.submit(_run_texture_job, job_id, mesh_path, ref_path, params)
    return {'job_id': job_id, 'status': 'pending'}


@app.get('/jobs/{job_id}')
async def get_job(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail='Job not found')
    return {
        'status': job['status'],
        'percent': job['percent'],
        'stage': job['stage'],
        'result': job['result'],
        'error': job['error'],
    }


@app.get('/download/{job_id}/{filename}')
async def download_file(job_id: str, filename: str):
    file_path = os.path.join(OUTPUT_DIR, job_id, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail='File not found')
    return FileResponse(file_path, media_type='application/octet-stream')


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8100, workers=1)
