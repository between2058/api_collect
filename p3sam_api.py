
# import os
# import shutil
# import tempfile
# import uuid
# import trimesh
# from fastapi import FastAPI, File, UploadFile, HTTPException
# from fastapi.responses import FileResponse
# from fastapi.middleware.cors import CORSMiddleware
# import uvicorn
# import argparse

# import traceback 

# # Import AutoMask from the provided script
# try:
#     from auto_mask import AutoMask
# except ImportError as e:
#     print(f"Warning: Could not import AutoMask. Dependencies might be missing. Error: {e}")
#     AutoMask = None

# app = FastAPI()

# # CORS configuration
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Configuration
# OUTPUT_DIR = tempfile.mkdtemp()
# OS_SEPARATOR = os.sep

# # Global model instance
# auto_mask_model = None

# @app.on_event("startup")
# async def load_model():
#     global auto_mask_model
#     if AutoMask is None:
#         print("❌ AutoMask class not available. Service will not function correctly.")
#         return

#     # Check for checkpoint
#     # ckpt_path = os.getenv("P3SAM_CHECKPOINT", "weights/last.ckpt")
#     ckpt_path=None
#     # if not os.path.exists(ckpt_path):
#     #     print(f"⚠️ Checkpoint not found at {ckpt_path}. Model loading might fail.")
#         # We allow startup even if model fails, but requests will error
    
#     try:
#         # Initialize AutoMask
#         # Note: AutoMask __init__ expects ckpt_path
#         # We assume defaults for other params as per auto_mask.py logic
#         print(f"Loading P3SAM model from {ckpt_path}...")
#         auto_mask_model = AutoMask(
#             ckpt_path=ckpt_path,
#             point_num=100000,
#             prompt_num=400,
#             threshold=0.95,
#             post_process=True
#         )
#         print("✅ P3SAM model loaded successfully.")
#     except Exception as e:
#         # print(f"❌ Failed to load P3SAM model: {e}")
#         print(f"❌ Failed to load P3SAM model. 真正的錯誤如下：")
#         print(f"Error Type: {type(e).__name__}")
#         print(f"Error Message: {e}")
#         traceback.print_exc()  # 這行最重要，它會印出哪一行掛掉

# @app.post("/segment")
# async def segment_3d(file: UploadFile = File(...)):
#     """
#     Accepts a .glb/.ply/.obj file, runs P3-SAM segmentation, 
#     and returns a .glb file (scene with segmented parts).
#     """
#     if auto_mask_model is None:
#         raise HTTPException(status_code=503, detail="Segmentation model is not loaded.")

#     request_id = str(uuid.uuid4())
#     job_dir = os.path.join(OUTPUT_DIR, request_id)
#     os.makedirs(job_dir, exist_ok=True)

#     input_filename = file.filename or "input.glb"
#     input_path = os.path.join(job_dir, input_filename)
#     output_glb_path = os.path.join(job_dir, "segmented_output_parts.glb")

#     try:
#         # Save uploaded file
#         with open(input_path, "wb") as buffer:
#             shutil.copyfileobj(file.file, buffer)
        
#         # Load mesh using trimesh
#         # auto_mask.py expects a trimesh object
#         print("✅ Model loaded.")
#         mesh = trimesh.load(input_path, force='mesh')

#         # Run prediction
#         # predict_aabb returns aabb, face_ids, mesh
#         # It also saves files to save_path if provided
#         print("✅ P3SAM model Start do segmentation.")
#         await run_in_threadpool(
#             auto_mask_model.predict_aabb,
#             mesh,
#             save_path=job_dir,
#             save_mid_res=False, # Don't flood with debug files
#             show_info=True,
#             clean_mesh_flag=True
#         )
#         print("✅ Segmentation Done")

#         # auto_mask.py logic (lines 1341-1350 in the provided file) 
#         # exports "auto_mask_mesh_final_parts.glb" if save_path is present.
#         # Let's verify if that file exists, or the output_path logic in the script.
#         # Looking at auto_mask.py:
#         # parts_scene_path = os.path.join(save_path, "auto_mask_mesh_final_parts.glb")
        
#         expected_output = os.path.join(job_dir, "auto_mask_mesh_final_parts.glb")
        
#         if not os.path.exists(expected_output):
#             # Fallback: maybe it saved as something else or we returned it?
#             # predict_aabb returns (aabb, final_face_ids, mesh)
#             # We could export it manually if the script didn't.
#             # But the logic I saw in auto_mask.py had the export.
#             raise HTTPException(status_code=500, detail="Model finished but output file was not found.")

#         # Rename to generic name for download
#         shutil.move(expected_output, output_glb_path)

#         return {
#             "segmented_glb": f"/download/{request_id}/segmented_output_parts.glb",
#             "request_id": request_id
#         }

#     except Exception as e:
#         print(f"Segmentation Error: {e}")
#         raise HTTPException(status_code=500, detail=str(e))

# @app.get("/download/{request_id}/{file_name}")
# async def download_file(request_id: str, file_name: str):
#     print("✅ Download Model")
#     file_path = os.path.join(OUTPUT_DIR, request_id, file_name)
#     if not os.path.exists(file_path):
#         raise HTTPException(status_code=404, detail="File not found")
#     return FileResponse(file_path, media_type='application/octet-stream', filename=file_name)

# @app.on_event("shutdown")
# async def cleanup():
#     shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

# # Async helper to run blocking code 
# from starlette.concurrency import run_in_threadpool

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=5001)
import os
import random
import shutil
import tempfile
import uuid
import trimesh
import gc  # 引入 garbage collection
import numpy as np
import torch # 引入 torch 以控制顯存
from typing import List
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
import uvicorn
import traceback

# Import AutoMask from the provided script
try:
    from auto_mask import AutoMask
except ImportError as e:
    print(f"Warning: Could not import AutoMask. Dependencies might be missing. Error: {e}")
    AutoMask = None

# --- Response Schemas ---

class HealthResponse(BaseModel):
    status: str
    model_available: bool = Field(description="AutoMask 是否成功 import（False 表示依賴未安裝）")

class SegmentResponse(BaseModel):
    segmented_glb: str = Field(description="分割結果 GLB 的下載路徑，傳入 GET /download/{request_id}/{file_name}")
    request_id: str = Field(description="此次請求的 UUID")
    num_parts: int = Field(description="偵測到的零件數量（face label >= 0 的唯一 ID 數）")

app = FastAPI(title="P3-SAM 3D Segmentation API")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
OUTPUT_DIR = tempfile.mkdtemp()
SUPPORTED_MESH_EXTENSIONS = {'.glb', '.ply', '.obj'}

# 注意：我們移除了全域變數 auto_mask_model，因為我们要每次請求都重新建立並銷毀

def set_seed(seed: int):
    """全局設定所有隨機種子，確保結果可重現"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_model_instance(
    point_num: int = 100000,
    prompt_num: int = 400,
    threshold: float = 0.95,
    post_process: bool = True,
):
    """
    每次呼叫時建立一個新的模型實例
    """
    if AutoMask is None:
        raise RuntimeError("AutoMask class not available.")

    ckpt_path = None # Set your checkpoint path here

    print(f"🔄 Loading P3SAM model (point_num={point_num}, prompt_num={prompt_num}, threshold={threshold}, post_process={post_process})...")
    model = AutoMask(
        ckpt_path=ckpt_path,
        point_num=point_num,
        prompt_num=prompt_num,
        threshold=threshold,
        post_process=post_process,
    )
    return model

def classify_exception(e: Exception) -> tuple[int, str, str]:
    """
    將例外分類為對應的 HTTP 狀態碼、錯誤代碼與說明。
    回傳 (status_code, error_code, human_readable_message)

    HTTP status code 語義：
      503 GPU_OOM           — GPU 記憶體不足，稍後可重試（附 Retry-After header）
      503 MODEL_UNAVAILABLE — 模型載入失敗或依賴不可用，稍後可重試
      507 DISK_FULL         — 磁碟空間不足，需人工介入
      500 INFERENCE_ERROR   — 未知推論錯誤，不可自動重試
    """
    if isinstance(e, torch.cuda.OutOfMemoryError):
        return 503, "GPU_OOM", "GPU out of memory. Free some VRAM and retry."
    if isinstance(e, RuntimeError) and "out of memory" in str(e).lower():
        return 503, "GPU_OOM", "GPU out of memory. Free some VRAM and retry."
    if isinstance(e, RuntimeError) and (
        "Model loading failed" in str(e) or "class not available" in str(e)
    ):
        return 503, "MODEL_UNAVAILABLE", str(e)
    if isinstance(e, OSError) and getattr(e, "errno", None) == 28:
        return 507, "DISK_FULL", "Server disk is full. Contact administrator."
    return 500, "INFERENCE_ERROR", str(e)

# Swagger 文件用的 responses 描述
GPU_ERROR_RESPONSES = {
    503: {
        "description": "GPU OOM 或模型/依賴不可用，稍後可重試（含 `Retry-After: 30` header）",
        "content": {
            "application/json": {
                "examples": {
                    "GPU_OOM": {
                        "summary": "GPU out of memory",
                        "value": {"detail": {"error_code": "GPU_OOM", "message": "GPU out of memory. Free some VRAM and retry."}},
                    },
                    "MODEL_UNAVAILABLE": {
                        "summary": "AutoMask unavailable",
                        "value": {"detail": {"error_code": "MODEL_UNAVAILABLE", "message": "AutoMask class not available."}},
                    },
                }
            }
        },
    },
    507: {
        "description": "Server 磁碟空間不足，需人工介入",
        "content": {
            "application/json": {
                "example": {"detail": {"error_code": "DISK_FULL", "message": "Server disk is full. Contact administrator."}}
            }
        },
    },
    500: {
        "description": "未知推論錯誤，不可自動重試",
        "content": {
            "application/json": {
                "example": {"detail": {"error_code": "INFERENCE_ERROR", "message": "<exception message>"}}
            }
        },
    },
}

def release_model_memory(model):
    """
    強制釋放模型佔用的 CPU 和 GPU 記憶體。
    即使清理過程本身發生例外也不會向上拋出，確保 finally 區塊不會二次崩潰。
    """
    print("🧹 Cleaning up GPU memory...")
    try:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        print("✨ GPU memory released.")
    except Exception as cleanup_err:
        print(f"⚠️ GPU memory cleanup failed (non-critical): {type(cleanup_err).__name__}: {cleanup_err}")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return {
        "status": "ok",
        "model_available": AutoMask is not None,
    }


@app.post("/segment", response_model=SegmentResponse, responses=GPU_ERROR_RESPONSES)
async def segment_3d(
    file: UploadFile = File(...),
    point_num: int = Form(100000, ge=1000, le=500000, description="點雲取樣數量，越大越精確但越慢"),
    prompt_num: int = Form(400, ge=10, le=1000, description="分割 Prompt 數量"),
    threshold: float = Form(0.95, ge=0.0, le=1.0, description="分割信心閾值"),
    post_process: bool = Form(True, description="是否套用後處理"),
    clean_mesh: bool = Form(True, description="推論前是否清理 Mesh"),
    seed: int = Form(42, description="隨機種子，控制點雲取樣與 Prompt 選取的可重現性"),
    prompt_bs: int = Form(32, ge=1, le=400, description="Prompt 推理 batch size，越大越快但佔用更多 VRAM"),
):
    ext = os.path.splitext(file.filename or '')[1].lower()
    if ext not in SUPPORTED_MESH_EXTENSIONS:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file type '{ext}'. Must be one of: {sorted(SUPPORTED_MESH_EXTENSIONS)}",
        )

    model = None # 初始化變數以便 finally 區塊存取

    request_id = str(uuid.uuid4())
    job_dir = os.path.join(OUTPUT_DIR, request_id)
    os.makedirs(job_dir, exist_ok=True)

    input_filename = file.filename or "input.glb"
    input_path = os.path.join(job_dir, input_filename)
    output_glb_path = os.path.join(job_dir, "segmented_output_parts.glb")

    try:
        # 1. 儲存檔案
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 2. 載入 Mesh (process=False 與官方 demo 一致，避免預處理改變結果)
        mesh = trimesh.load(input_path, force='mesh', process=False)

        # 3. 載入模型 (Load)
        model = load_model_instance(
            point_num=point_num,
            prompt_num=prompt_num,
            threshold=threshold,
            post_process=post_process,
        )

        # 4. 執行預測 (Inference)
        set_seed(seed)  # 全局設定隨機種子
        print("✅ P3SAM model Start do segmentation.")
        aabb, face_ids, final_mesh = await run_in_threadpool(
            model.predict_aabb,
            mesh,
            save_path=job_dir,
            save_mid_res=False,
            show_info=True,
            clean_mesh_flag=clean_mesh,
            seed=seed,
            prompt_bs=prompt_bs,
        )
        print("✅ Segmentation Done")

        # 5. 根據 face_ids 上色並匯出 GLB
        unique_ids = np.unique(face_ids)
        color_map = {
            i: (np.random.rand(3) * 255).astype(np.uint8)
            for i in unique_ids if i >= 0
        }
        face_colors = np.array(
            [color_map[fid] if fid >= 0 else [0, 0, 0] for fid in face_ids],
            dtype=np.uint8,
        )
        mesh_out = final_mesh.copy()
        mesh_out.visual.face_colors = face_colors
        mesh_out.export(output_glb_path)

        num_parts = int(np.sum(unique_ids >= 0))

        return {
            "segmented_glb": f"/download/{request_id}/segmented_output_parts.glb",
            "request_id": request_id,
            "num_parts": num_parts,
        }

    except Exception as e:
        status, error_code, message = classify_exception(e)
        print(f"❌ [{error_code}] request_id={request_id} | {type(e).__name__}: {e}")
        traceback.print_exc()
        headers = {"Retry-After": "30"} if status == 503 else {}
        raise HTTPException(
            status_code=status,
            detail={"error_code": error_code, "message": message},
            headers=headers,
        )
        
    finally:
        # 5. 清理資源 (Unload)
        # 無論成功或失敗，這段程式碼都會執行
        if model is not None:
            release_model_memory(model)
            model = None # 確保變數指向空

@app.get("/download/{request_id}/{file_name}")
async def download_file(request_id: str, file_name: str):
    file_path = os.path.join(OUTPUT_DIR, request_id, file_name)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, media_type='application/octet-stream', filename=file_name)

@app.on_event("shutdown")
async def cleanup():
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5001)
