import os
import sys
import gc
import uuid
import shutil
import asyncio
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'model_refs', 'Depth-Anything-3', 'src'))

import cv2
import torch
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from depth_anything_3.api import DepthAnything3

# ── Global state ──────────────────────────────────────────────────────────────

da3_model: DepthAnything3 = None
DA3_MODEL = os.getenv('DA3_MODEL', 'depth-anything/da3-large')
jobs: dict[str, dict] = {}
_executor = ThreadPoolExecutor(max_workers=1)
OUTPUT_DIR = tempfile.mkdtemp()
MAX_FRAMES = 60

print(f"📂 DA3 Output directory: {OUTPUT_DIR}")


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


def _extract_frames(video_path: str, output_dir: str, fps: float) -> list[str]:
    """Extract frames from video at given fps, capped at MAX_FRAMES."""
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = max(1, int(round(video_fps / fps)))

    saved = []
    frame_idx = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            fname = os.path.join(output_dir, f'frame_{saved_count:04d}.jpg')
            cv2.imwrite(fname, frame)
            saved.append(fname)
            saved_count += 1
            if saved_count >= MAX_FRAMES:
                break
        frame_idx += 1

    cap.release()
    return sorted(saved)


def _run_da3_job(job_id: str, image_paths: list[str]):
    try:
        jobs[job_id]['status'] = 'running'
        _update_job(job_id, 10, 'Running DA3 depth + Gaussian inference...')

        job_dir = os.path.join(OUTPUT_DIR, job_id)

        prediction = da3_model.inference(
            image=image_paths,
            export_dir=job_dir,
            export_format='npz-glb-gs_ply-gs_video',
            align_to_input_ext_scale=True,
            infer_gs=True,
        )

        _update_job(job_id, 90, 'Exporting Gaussian splat (gs_ply)...')

        # DA3 writes outputs at fixed relative paths inside export_dir:
        #   {job_dir}/exports/npz/result.npz
        #   {job_dir}/gs_ply/0000.ply
        #   {job_dir}/gs_video/0000_extend.mp4

        _update_job(job_id, 100, 'Done ✓')
        jobs[job_id]['status'] = 'completed'
        jobs[job_id]['result'] = {
            'gs_ply_url': f'/download/{job_id}/gs_ply/0000.ply',
            'npz_url': f'/download/{job_id}/exports/npz/result.npz',
            'video_url': f'/download/{job_id}/gs_video/0000_extend.mp4',
            'frame_count': len(image_paths),
        }

    except Exception as e:
        print(f"❌ DA3 inference error [{job_id}]: {e}")
        jobs[job_id]['status'] = 'failed'
        jobs[job_id]['error'] = str(e)
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _run_video_job(job_id: str, video_path: str, fps: float):
    try:
        jobs[job_id]['status'] = 'running'
        _update_job(job_id, 5, 'Extracting frames...')

        job_dir = os.path.join(OUTPUT_DIR, job_id)
        frames_dir = os.path.join(job_dir, 'frames')
        os.makedirs(frames_dir, exist_ok=True)

        image_paths = _extract_frames(video_path, frames_dir, fps)

        if not image_paths:
            raise RuntimeError('No frames extracted from video')

        print(f"📹 Extracted {len(image_paths)} frames from video")
        _run_da3_job(job_id, image_paths)

    except Exception as e:
        if jobs[job_id]['status'] != 'failed':
            print(f"❌ DA3 video error [{job_id}]: {e}")
            jobs[job_id]['status'] = 'failed'
            jobs[job_id]['error'] = str(e)


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
    global da3_model
    print(f"⏳ Loading Depth-Anything-3 model: {DA3_MODEL}...")
    da3_model = DepthAnything3.from_pretrained(DA3_MODEL).cuda().eval()
    print("✅ DA3 model loaded")
    cleanup_task = asyncio.create_task(_cleanup_loop())
    yield
    cleanup_task.cancel()
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)


app = FastAPI(title='Depth-Anything-3 API', lifespan=lifespan)

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
        'model_loaded': da3_model is not None,
        'active_jobs': len(jobs),
        'max_frames': MAX_FRAMES,
    }


@app.post('/reconstruct/video')
async def reconstruct_from_video(
    file: UploadFile = File(...),
    fps: float = Form(1.0),
):
    if fps <= 0:
        raise HTTPException(status_code=422, detail=f"fps must be > 0, got {fps}")

    job_id = _make_job()
    job_dir = os.path.join(OUTPUT_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    ext = os.path.splitext(file.filename or 'video.mp4')[1] or '.mp4'
    video_path = os.path.join(job_dir, f'input{ext}')
    with open(video_path, 'wb') as f:
        f.write(await file.read())

    _executor.submit(_run_video_job, job_id, video_path, fps)
    return {'job_id': job_id, 'status': 'pending'}


@app.post('/reconstruct/images')
async def reconstruct_from_images(
    files: list[UploadFile] = File(...),
):
    job_id = _make_job()
    job_dir = os.path.join(OUTPUT_DIR, job_id)
    frames_dir = os.path.join(job_dir, 'frames')
    os.makedirs(frames_dir, exist_ok=True)

    total_received = len(files)
    capped = total_received > MAX_FRAMES
    if capped:
        print(f"⚠️ Received {total_received} images; capping at MAX_FRAMES={MAX_FRAMES}")

    image_paths = []
    for i, f in enumerate(files[:MAX_FRAMES]):
        ext = os.path.splitext(f.filename or f'frame_{i}.jpg')[1] or '.jpg'
        img_path = os.path.join(frames_dir, f'frame_{i:04d}{ext}')
        with open(img_path, 'wb') as out:
            out.write(await f.read())
        image_paths.append(img_path)

    _executor.submit(_run_da3_job, job_id, sorted(image_paths))
    response = {'job_id': job_id, 'status': 'pending', 'frames_queued': len(image_paths)}
    if capped:
        response['warning'] = f"Received {total_received} images; only first {MAX_FRAMES} will be processed"
    return response


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


@app.get('/download/{job_id}/{file_path:path}')
async def download_file(job_id: str, file_path: str):
    full_path = os.path.join(OUTPUT_DIR, job_id, file_path)
    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail='File not found')
    return FileResponse(full_path, media_type='application/octet-stream')


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8200, workers=1)
