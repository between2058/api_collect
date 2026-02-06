# Copyright (c) Meta Platforms, Inc. and affiliates.
# SAM-3D API - FastAPI wrapper for SAM3D inference

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import sys
import uuid
import shutil
import tempfile
import numpy as np
import torch
import trimesh
from PIL import Image

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
async def load_model():
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
        
        return {
            "request_id": request_id,
            "glb_file": f"/download/{request_id}/output.glb"
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
    
    ** 更新: 已加入位置修正功能 (Official Fix) **
    - 套用座標轉換 (Y-Up -> Z-Up)
    - 套用預測的 Scale/Rotation/Translation
    """
    global inference
    
    if inference is None:
        raise HTTPException(status_code=503, detail="模型尚未載入")
    
    try:
        request_id = str(uuid.uuid4())
        print(f"SAM-3D 批次請求 ID: {request_id}, 共 {len(mask_images)} 個 masks")
        
        work_dir = os.path.join(OUTPUT_DIR, request_id)
        os.makedirs(work_dir, exist_ok=True)
        
        image_path = os.path.join(work_dir, "image.png")
        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        
        img = load_image(image_path)
        
        masks = []
        for i, mask_file in enumerate(mask_images):
            mask_path = os.path.join(work_dir, f"{i}.png")
            with open(mask_path, "wb") as buffer:
                shutil.copyfileobj(mask_file.file, buffer)
            masks.append(load_mask(mask_path))
        
        # 1. 執行推論
        outputs = [inference(img, mask, seed=seed) for mask in masks]
        
        glb_files = []
        
        # 2. 應用座標轉換與變形 (Official Fix Logic)
        for i, output in enumerate(outputs):
            # A. Copy the raw mesh
            mesh = output["glb"].copy()
            
            # B. Coordinate Conversion (Y-Up to Z-Up)
            # The raw GLB from inference is Y-Up, but predicted params are Z-Up.
            vertices = mesh.vertices.astype(np.float32) @ _R_YUP_TO_ZUP
            
            # Ensure tensor is on correct device
            device = output["rotation"].device
            vertices_tensor = torch.from_numpy(vertices).float().to(device)
            
            # C. Prepare Transformation Matrix
            R_l2c = quaternion_to_matrix(output["rotation"])
            
            l2c_transform = compose_transform(
                scale=output["scale"],
                rotation=R_l2c,
                translation=output["translation"],
            )
            
            # D. Apply Transformation
            vertices_transformed = l2c_transform.transform_points(vertices_tensor.unsqueeze(0))
            vertices_final = vertices_transformed.squeeze(0).cpu().numpy()
            
            # E. Update Mesh Vertices
            mesh.vertices = vertices_final
            
            # F. Export
            glb_path = os.path.join(work_dir, f"output_{i}.glb")
            mesh.export(glb_path)
            glb_files.append(f"/download/{request_id}/output_{i}.glb")
        
        return {
            "request_id": request_id,
            "count": len(glb_files),
            "glb_files": glb_files
        }
        
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
