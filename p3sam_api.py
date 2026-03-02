
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
import shutil
import tempfile
import uuid
import trimesh
import gc  # 引入 garbage collection
import torch # 引入 torch 以控制顯存
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
import uvicorn
import traceback 

# Import AutoMask from the provided script
try:
    from auto_mask import AutoMask
except ImportError as e:
    print(f"Warning: Could not import AutoMask. Dependencies might be missing. Error: {e}")
    AutoMask = None

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

def release_model_memory(model):
    """
    強制釋放模型佔用的 CPU 和 GPU 記憶體
    """
    print("🧹 Cleaning up GPU memory...")
    
    # 1. 刪除模型引用
    del model
    
    # 2. 強制執行 Python 垃圾回收
    gc.collect()
    
    # 3. 清空 PyTorch CUDA 快取 (這一步最重要)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect() # 有時候這行也有幫助
        
    print("✨ GPU memory released.")

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "model_available": AutoMask is not None,
    }


@app.post("/segment")
async def segment_3d(
    file: UploadFile = File(...),
    point_num: int = Form(100000, description="點雲取樣數量，越大越精確但越慢"),
    prompt_num: int = Form(400, description="分割 Prompt 數量"),
    threshold: float = Form(0.95, description="分割信心閾值 (0.0–1.0)"),
    post_process: bool = Form(True, description="是否套用後處理"),
    clean_mesh: bool = Form(True, description="推論前是否清理 Mesh"),
):
    ext = os.path.splitext(file.filename or '')[1].lower()
    if ext not in SUPPORTED_MESH_EXTENSIONS:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file type '{ext}'. Must be one of: {sorted(SUPPORTED_MESH_EXTENSIONS)}",
        )

    if not (1000 <= point_num <= 500000):
        raise HTTPException(status_code=422, detail="point_num must be between 1000 and 500000")
    if not (10 <= prompt_num <= 1000):
        raise HTTPException(status_code=422, detail="prompt_num must be between 10 and 1000")
    if not (0.0 <= threshold <= 1.0):
        raise HTTPException(status_code=422, detail="threshold must be between 0.0 and 1.0")

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

        # 2. 載入 Mesh
        mesh = trimesh.load(input_path, force='mesh')

        # 3. 載入模型 (Load)
        model = load_model_instance(
            point_num=point_num,
            prompt_num=prompt_num,
            threshold=threshold,
            post_process=post_process,
        )

        # 4. 執行預測 (Inference)
        print("✅ P3SAM model Start do segmentation.")
        await run_in_threadpool(
            model.predict_aabb,
            mesh,
            save_path=job_dir,
            save_mid_res=False,
            show_info=True,
            clean_mesh_flag=clean_mesh,
        )
        print("✅ Segmentation Done")

        # 檢查輸出
        expected_output = os.path.join(job_dir, "auto_mask_mesh_final_parts.glb")
        if not os.path.exists(expected_output):
            raise HTTPException(status_code=500, detail="Model finished but output file was not found.")

        shutil.move(expected_output, output_glb_path)

        return {
            "segmented_glb": f"/download/{request_id}/segmented_output_parts.glb",
            "request_id": request_id
        }

    except Exception as e:
        print(f"Segmentation Error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
        
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
