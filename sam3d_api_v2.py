# Copyright (c) Meta Platforms, Inc. and affiliates.
# SAM-3D API - FastAPI wrapper for SAM3D inference

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import os
import sys
import uuid
import shutil
import tempfile
import time
import numpy as np
import torch
import trimesh
from PIL import Image

OUTPUT_CLEANUP_TTL = 60 * 60  # clean up output files after 1 hour

# --- PyTorch3D Imports for Transformation Logic ---
from pytorch3d.transforms import Transform3d, Rotate, Translate, Scale, quaternion_to_matrix

# Add notebook path for inference imports
sys.path.append("notebook")

app = FastAPI(title="SAM-3D API", description="Image-to-3D generation using SAM3D")

# CORS 設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 臨時輸出目錄
OUTPUT_DIR = tempfile.mkdtemp()
print(f"SAM-3D 輸出目錄: {OUTPUT_DIR}")

# 全域模型實例
inference = None

# Track output directories for cleanup: {request_id: created_at}
_output_registry: dict[str, float] = {}


async def _output_cleanup_loop():
    """Background task: remove output dirs older than OUTPUT_CLEANUP_TTL."""
    while True:
        await asyncio.sleep(30 * 60)  # check every 30 minutes
        cutoff = time.time() - OUTPUT_CLEANUP_TTL
        expired = [rid for rid, ts in list(_output_registry.items()) if ts < cutoff]
        for rid in expired:
            _output_registry.pop(rid, None)
            job_dir = os.path.join(OUTPUT_DIR, rid)
            shutil.rmtree(job_dir, ignore_errors=True)
            print(f"🧹 Cleaned up expired output: {rid}")

# --- Transformation Helpers (From provided script) ---

def compose_transform(scale, rotation, translation):
    """
    Fallback implementation of compose_transform.
    Composition order: Scale -> Rotate -> Translate.
    """
    t = Transform3d(device=scale.device)
    t = t.compose(Scale(scale))
    t = t.compose(Rotate(rotation))
    t = t.compose(Translate(translation))
    return t

# Coordinate rotation matrices (Z-Up <-> Y-Up)
_R_ZUP_TO_YUP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
_R_YUP_TO_ZUP = _R_ZUP_TO_YUP.T

# -----------------------------------------------------

@app.on_event("startup")
async def startup():
    global inference
    try:
        from inference import Inference
        tag = "hf"
        config_path = f"checkpoints/{tag}/pipeline.yaml"
        inference = Inference(config_path, compile=False)
        print("✅ SAM-3D 模型載入成功")
    except Exception as e:
        print(f"❌ SAM-3D 模型載入失敗: {str(e)}")
        raise RuntimeError(f"SAM-3D 模型載入失敗: {str(e)}")
    asyncio.create_task(_output_cleanup_loop())


def load_image(path):
    """與 SAM3D 完全相同的 load_image"""
    image = Image.open(path)
    image = np.array(image)
    image = image.astype(np.uint8)
    return image


def load_mask(path):
    """與 SAM3D 完全相同的 load_mask"""
    mask = load_image(path)
    mask = mask > 0
    if mask.ndim == 3:
        mask = mask[..., -1]  # 取最後一個通道 (alpha)
    return mask


@app.post("/generate")
async def generate_3d(
    image: UploadFile = File(..., description="原圖 (RGBA)"),
    mask_image: UploadFile = File(..., description="去背圖 (RGBA，透明背景)"),
    seed: int = 42
):
    """
    從原圖 + 去背圖生成 3D 模型 (保留原始邏輯，未修改)
    """
    global inference
    
    if inference is None:
        raise HTTPException(status_code=503, detail="模型尚未載入")
    
    try:
        request_id = str(uuid.uuid4())
        print(f"SAM-3D 請求 ID: {request_id}")
        
        work_dir = os.path.join(OUTPUT_DIR, request_id)
        os.makedirs(work_dir, exist_ok=True)
        
        image_path = os.path.join(work_dir, "image.png")
        mask_image_path = os.path.join(work_dir, "mask.png")
        
        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        with open(mask_image_path, "wb") as buffer:
            shutil.copyfileobj(mask_image.file, buffer)
        
        img = load_image(image_path)
        mask = load_mask(mask_image_path)
        
        output = inference(img, mask, seed=seed)
        
        glb_path = os.path.join(work_dir, "output.glb")
        output["glb"].export(glb_path)

        _output_registry[request_id] = time.time()

        return {
            "request_id": request_id,
            "glb_file": f"/download/{request_id}/output.glb",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate-batch")
async def generate_batch(
    image: UploadFile = File(..., description="原圖 (RGBA)"),
    mask_images: list[UploadFile] = File(..., description="多張去背圖 (RGBA，透明背景)"),
    seed: int = 42
):
    """
    從原圖 + 多張去背圖批次生成多個 3D 模型

    HTTP status:
      200 — all masks succeeded
      207 — partial success
      503 — model not loaded
      500 — all masks failed
    """
    global inference

    if inference is None:
        raise HTTPException(status_code=503, detail="模型尚未載入")

    request_id = str(uuid.uuid4())
    print(f"SAM-3D 批次請求 ID: {request_id}, 共 {len(mask_images)} 個 masks")

    work_dir = os.path.join(OUTPUT_DIR, request_id)
    os.makedirs(work_dir, exist_ok=True)

    try:
        image_path = os.path.join(work_dir, "image.png")
        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        img = load_image(image_path)

        results = []

        for i, mask_file in enumerate(mask_images):
            mask_path = os.path.join(work_dir, f"{i}.png")
            with open(mask_path, "wb") as buffer:
                shutil.copyfileobj(mask_file.file, buffer)

            try:
                mask = load_mask(mask_path)
                output = inference(img, mask, seed=seed)

                mesh = output["glb"].copy()
                vertices = mesh.vertices.astype(np.float32) @ _R_YUP_TO_ZUP

                device = output["rotation"].device
                vertices_tensor = torch.from_numpy(vertices).float().to(device)

                R_l2c = quaternion_to_matrix(output["rotation"])
                l2c_transform = compose_transform(
                    scale=output["scale"],
                    rotation=R_l2c,
                    translation=output["translation"],
                )
                vertices_transformed = l2c_transform.transform_points(vertices_tensor.unsqueeze(0))
                mesh.vertices = vertices_transformed.squeeze(0).cpu().numpy()

                glb_path = os.path.join(work_dir, f"output_{i}.glb")
                mesh.export(glb_path)

                results.append({
                    "mask_index": i,
                    "status": "success",
                    "glb_file": f"/download/{request_id}/output_{i}.glb",
                })

            except Exception as e:
                print(f"  ❌ mask {i} failed: {e}")
                results.append({
                    "mask_index": i,
                    "status": "failed",
                    "error": str(e),
                })

        _output_registry[request_id] = time.time()

        succeeded = sum(1 for r in results if r["status"] == "success")
        failed = len(results) - succeeded

        response_body = {
            "request_id": request_id,
            "total": len(mask_images),
            "succeeded": succeeded,
            "failed": failed,
            "results": results,
        }

        if succeeded == 0:
            raise HTTPException(status_code=500, detail=f"All {len(mask_images)} masks failed. See 'results' for per-mask errors.")

        status_code = 207 if failed > 0 else 200
        return JSONResponse(content=response_body, status_code=status_code)

    except HTTPException:
        raise
    except Exception as e:
        print(f"Batch generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/download/{request_id}/{file_name}")
async def download_file(request_id: str, file_name: str):
    """下載生成的檔案"""
    file_path = os.path.join(OUTPUT_DIR, request_id, file_name)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="找不到檔案")
    return FileResponse(file_path, media_type='application/octet-stream', filename=file_name)


@app.get("/health")
async def health_check():
    """健康檢查"""
    return {
        "status": "ok",
        "model_loaded": inference is not None
    }


@app.on_event("shutdown")
async def cleanup():
    """清理臨時檔案"""
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    print("🧹 SAM-3D 臨時檔案已清理")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
