import os
import io
import gc
import uuid
import shutil
import asyncio
import base64
import random
import torch
import tempfile
import numpy as np
from typing import List, Optional
from PIL import Image

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
import traceback

# --- Diffusers Imports ---
from diffusers import QwenImagePipeline, QwenImageEditPlusPipeline

# --- 1. 初始化設定 ---
app = FastAPI(title="Qwen All-in-One API (Text2Img, Edit, Angle)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 建立暫存目錄
OUTPUT_DIR = tempfile.mkdtemp()
print(f"📂 Output directory: {OUTPUT_DIR}")

# 硬體設定
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
MAX_SEED = 2147483647

# 全局 GPU 鎖 (確保一次只跑一個模型)
gpu_lock = asyncio.Lock()

# --- 2. 輔助函式與資料結構 ---

def flush_gpu():
    """強制清理 GPU 記憶體"""
    gc.collect()
    torch.cuda.empty_cache()
    print("🧹 GPU Memory Flushed")

def classify_exception(e: Exception) -> tuple[int, str, str]:
    """
    將例外分類為對應的 HTTP 狀態碼、錯誤代碼與說明。
    回傳 (status_code, error_code, human_readable_message)

    HTTP status code 語義：
      503 GPU_OOM           — GPU 記憶體不足，稍後可重試（附 Retry-After header）
      503 MODEL_UNAVAILABLE — 模型載入失敗，稍後可重試
      507 DISK_FULL         — 磁碟空間不足，需人工介入
      500 INFERENCE_ERROR   — 未知推論錯誤，不可自動重試
    """
    # GPU 記憶體不足（PyTorch >= 2.1 的具名例外）
    if isinstance(e, torch.cuda.OutOfMemoryError):
        return 503, "GPU_OOM", "GPU out of memory. Free some VRAM and retry."
    # 舊版 PyTorch OOM（RuntimeError: CUDA out of memory ...）
    if isinstance(e, RuntimeError) and "out of memory" in str(e).lower():
        return 503, "GPU_OOM", "GPU out of memory. Free some VRAM and retry."
    # 模型/依賴不可用
    if isinstance(e, RuntimeError) and (
        "Model loading failed" in str(e) or "class not available" in str(e)
    ):
        return 503, "MODEL_UNAVAILABLE", str(e)
    # 磁碟空間不足（ENOSPC errno=28）
    if isinstance(e, OSError) and getattr(e, "errno", None) == 28:
        return 507, "DISK_FULL", "Server disk is full. Contact administrator."
    # 其他未知推論錯誤
    return 500, "INFERENCE_ERROR", str(e)

# Swagger 文件用的 responses 描述（套用在所有推論 endpoint）
GPU_ERROR_RESPONSES = {
    503: {
        "description": "GPU OOM 或模型載入失敗，稍後可重試（含 `Retry-After: 30` header）",
        "content": {
            "application/json": {
                "examples": {
                    "GPU_OOM": {
                        "summary": "GPU out of memory",
                        "value": {"detail": {"error_code": "GPU_OOM", "message": "GPU out of memory. Free some VRAM and retry."}},
                    },
                    "MODEL_UNAVAILABLE": {
                        "summary": "Model failed to load",
                        "value": {"detail": {"error_code": "MODEL_UNAVAILABLE", "message": "Model loading failed: <reason>"}},
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

def save_image(image: Image.Image, folder: str, filename: str) -> str:
    path = os.path.join(folder, filename)
    image.save(path)
    return path

# --- Angle 相關映射表 ---
AZIMUTH_MAP = {
    0: "front view", 45: "front-right quarter view", 90: "right side view",
    135: "back-right quarter view", 180: "back view", 225: "back-left quarter view",
    270: "left side view", 315: "front-left quarter view"
}
ELEVATION_MAP = {-30: "low-angle shot", 0: "eye-level shot", 30: "elevated shot", 60: "high-angle shot"}
DISTANCE_MAP = {0.6: "close-up", 1.0: "medium shot", 1.8: "wide shot"}

def snap_to_nearest(value: float, options: List[float]) -> float:
    return min(options, key=lambda x: abs(x - value))

def build_angle_prompt(azimuth: float, elevation: float = 0.0, distance: float = 1.0) -> str:
    az_snap = snap_to_nearest(azimuth, list(AZIMUTH_MAP.keys()))
    el_snap = snap_to_nearest(elevation, list(ELEVATION_MAP.keys()))
    dist_snap = snap_to_nearest(distance, list(DISTANCE_MAP.keys()))
    return f"<sks> {AZIMUTH_MAP[az_snap]} {ELEVATION_MAP[el_snap]} {DISTANCE_MAP[dist_snap]}"

SUPPORTED_IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff'}

def _validate_image_upload(file: UploadFile, field_name: str = "file"):
    ext = os.path.splitext(file.filename or '')[1].lower()
    if ext not in SUPPORTED_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported image type '{ext}' for field '{field_name}'. Must be one of: {sorted(SUPPORTED_IMAGE_EXTENSIONS)}",
        )

# Text2Image 的比例設定
ASPECT_RATIOS = {
    "1:1": (1328, 1328), "16:9": (1664, 928), "9:16": (928, 1664),
    "4:3": (1472, 1104), "3:4": (1104, 1472), "3:2": (1584, 1056), "2:3": (1056, 1584),
}

# --- Request Models ---
class Text2ImgRequest(BaseModel):
    prompt: str
    negative_prompt: str = "low quality, bad anatomy, blurry, distorted"
    aspect_ratio: str = "16:9"  # Keys in ASPECT_RATIOS
    num_steps: int = 50
    cfg_scale: float = 4.0
    seed: int = Field(default_factory=lambda: random.randint(0, MAX_SEED))
    num_samples: int = 1  # 生成張數，每張使用獨立隨機 seed

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "device": DEVICE,
        "gpu_busy": gpu_lock.locked(),
    }


# ==========================================
# MODEL 1: Text-to-Image (Text -> Image)
# ==========================================
@app.post("/text2img", responses=GPU_ERROR_RESPONSES)
async def text_to_image(req: Text2ImgRequest):
    """
    [Model 1] Qwen-Image-2512
    輸入: 純文字 Prompt
    輸出: 生成的圖片
    """
    request_id = str(uuid.uuid4())
    req_dir = os.path.join(OUTPUT_DIR, request_id)
    os.makedirs(req_dir, exist_ok=True)

    if req.aspect_ratio not in ASPECT_RATIOS:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid aspect_ratio '{req.aspect_ratio}'. Must be one of: {list(ASPECT_RATIOS.keys())}",
        )
    if not (1 <= req.num_samples <= 8):
        raise HTTPException(
            status_code=422,
            detail="num_samples must be between 1 and 8",
        )
    width, height = ASPECT_RATIOS[req.aspect_ratio]
    print(f"📝 [Text2Img] ID: {request_id} | Prompt: {req.prompt[:50]}... | Size: {width}x{height} | Samples: {req.num_samples}")

    async with gpu_lock:
        def run_inference():
            pipe = None
            try:
                print("⏳ Loading Qwen-Image-2512...")
                pipe = QwenImagePipeline.from_pretrained(
                    "Qwen/Qwen-Image-2512",
                    torch_dtype=DTYPE
                ).to(DEVICE)

                seeds = [random.randint(0, MAX_SEED) for _ in range(req.num_samples)]
                paths = []
                for i, seed_i in enumerate(seeds):
                    generator = torch.Generator(device=DEVICE).manual_seed(seed_i)
                    print(f"🚀 Generating image {i+1}/{req.num_samples} (seed={seed_i})...")
                    image = pipe(
                        prompt=req.prompt,
                        negative_prompt=req.negative_prompt,
                        width=width,
                        height=height,
                        num_inference_steps=req.num_steps,
                        true_cfg_scale=req.cfg_scale,
                        generator=generator
                    ).images[0]
                    path = save_image(image, req_dir, f"output_{i}.png")
                    paths.append(path)

                return paths, seeds
            finally:
                if pipe: del pipe
                flush_gpu()

        try:
            output_paths, used_seeds = await run_in_threadpool(run_inference)
            urls = [f"/download/{request_id}/output_{i}.png" for i in range(len(output_paths))]
            return {
                "status": "success",
                "request_id": request_id,
                "urls": urls,
                "seeds": used_seeds
            }
        except Exception as e:
            status, error_code, message = classify_exception(e)
            print(f"❌ [{error_code}] request_id={request_id} | {type(e).__name__}: {e}")
            traceback.print_exc()
            if error_code == "GPU_OOM":
                flush_gpu()
            headers = {"Retry-After": "30"} if status == 503 else {}
            raise HTTPException(
                status_code=status,
                detail={"error_code": error_code, "message": message},
                headers=headers,
            )


# ==========================================
# MODEL 2: Edit (Image + Prompt -> Image)
# ==========================================
@app.post("/edit", responses=GPU_ERROR_RESPONSES)
async def edit_image(
    file: UploadFile = File(...),
    prompt: str = Form(..., description="編輯指令"),
    steps: int = Form(40),
    cfg_scale: float = Form(4.0),
    seed: int = Form(42),
    num_samples: int = Form(1)
):
    """
    [Model 2] Qwen-Image-Edit-2511 (Base Model)
    輸入: 圖片 + Prompt
    功能: 根據文字指令修改圖片內容
    """
    _validate_image_upload(file, "file")

    if not (1 <= num_samples <= 8):
        raise HTTPException(status_code=422, detail="num_samples must be between 1 and 8")

    request_id = str(uuid.uuid4())
    req_dir = os.path.join(OUTPUT_DIR, request_id)
    os.makedirs(req_dir, exist_ok=True)

    # 儲存上傳圖片
    input_path = os.path.join(req_dir, "input.png")
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    print(f"🎨 [Edit] ID: {request_id} | Prompt: {prompt} | Samples: {num_samples}")

    async with gpu_lock:
        def run_inference():
            pipe = None
            try:
                print("⏳ Loading Qwen-Image-Edit-2511...")
                pipe = QwenImageEditPlusPipeline.from_pretrained(
                    "Qwen/Qwen-Image-Edit-2511",
                    torch_dtype=DTYPE
                ).to(DEVICE)

                img_obj = Image.open(input_path).convert("RGB")
                seeds = [random.randint(0, MAX_SEED) for _ in range(num_samples)]
                paths = []
                for i, seed_i in enumerate(seeds):
                    generator = torch.Generator(device=DEVICE).manual_seed(seed_i)
                    print(f"🚀 Editing image {i+1}/{num_samples} (seed={seed_i})...")
                    output = pipe(
                        image=[img_obj],
                        prompt=prompt,
                        num_inference_steps=steps,
                        true_cfg_scale=cfg_scale,
                        generator=generator,
                        num_images_per_prompt=1
                    ).images[0]
                    path = save_image(output, req_dir, f"result_{i}.png")
                    paths.append(path)

                return paths, seeds
            finally:
                if pipe: del pipe
                flush_gpu()

        try:
            output_paths, used_seeds = await run_in_threadpool(run_inference)
            result_urls = [f"/download/{request_id}/result_{i}.png" for i in range(len(output_paths))]
            return {
                "status": "success",
                "request_id": request_id,
                "input_url": f"/download/{request_id}/input.png",
                "result_urls": result_urls,
                "seeds": used_seeds
            }
        except Exception as e:
            status, error_code, message = classify_exception(e)
            print(f"❌ [{error_code}] request_id={request_id} | {type(e).__name__}: {e}")
            traceback.print_exc()
            if error_code == "GPU_OOM":
                flush_gpu()
            headers = {"Retry-After": "30"} if status == 503 else {}
            raise HTTPException(
                status_code=status,
                detail={"error_code": error_code, "message": message},
                headers=headers,
            )

# ==========================================
# MODEL 2: Edit Multi (Multiple Images + 1 Prompt)
# ==========================================
@app.post("/edit-multi", responses=GPU_ERROR_RESPONSES)
async def edit_multi_images(
    files: List[UploadFile] = File(..., description="上傳多張圖片"),
    prompt: str = Form(..., description="編輯指令"),
    steps: int = Form(40),
    cfg_scale: float = Form(4.0),
    seed: int = Form(42)
):
    """
    [Model 2] 
    輸入: 多張圖片 + Prompt
    功能: 根據文字指令修改圖片內容
    """
    for i, f in enumerate(files):
        _validate_image_upload(f, f"files[{i}]")

    request_id = str(uuid.uuid4())
    req_dir = os.path.join(OUTPUT_DIR, request_id)
    os.makedirs(req_dir, exist_ok=True)

    print(f"🎨 [Edit-Cascade] ID: {request_id} | Images: {len(files)} | Prompt: {prompt}")

    input_images_pil = []
    input_urls = []

    # 1. 讀取圖片
    for i, file in enumerate(files):
        filename = f"input_{i}.png"
        file_path = os.path.join(req_dir, filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        img = Image.open(file_path).convert("RGB")
        input_images_pil.append(img)
        input_urls.append(f"/download/{request_id}/{filename}")

    async with gpu_lock:
        def run_inference():
            pipe = None
            try:
                # --- Step B: 推論 (Inference) ---
                print("⏳ Loading Qwen-Image-Edit-2511...")
                pipe = QwenImageEditPlusPipeline.from_pretrained(
                    "Qwen/Qwen-Image-Edit-2511",
                    torch_dtype=DTYPE
                ).to(DEVICE)
                
                generator = torch.Generator(device=DEVICE).manual_seed(seed)

                print("🚀 Editing stitched image...")
                output_stitched = pipe(
                    image=input_images_pil,
                    prompt=prompt, 
                    num_inference_steps=steps,
                    true_cfg_scale=cfg_scale,
                    generator=generator,
                    num_images_per_prompt=1
                ).images[0]

                # --- Step C (修改): 直接儲存大圖，不要切開 ---
                output_filename = "result_stitched.png"
                save_image(output_stitched, req_dir, output_filename)
                
                # 回傳單一結果的 List
                return [f"/download/{request_id}/{output_filename}"]

            finally:
                if pipe: del pipe
                flush_gpu()

        try:
            result_urls = await run_in_threadpool(run_inference)
            return {
                "status": "success",
                "request_id": request_id,
                "count": len(files),
                "inputs": input_urls,
                "results": result_urls  # 這裡現在只會包含一張大圖的 URL
            }
        except Exception as e:
            status, error_code, message = classify_exception(e)
            print(f"❌ [{error_code}] request_id={request_id} | {type(e).__name__}: {e}")
            traceback.print_exc()
            if error_code == "GPU_OOM":
                flush_gpu()
            headers = {"Retry-After": "30"} if status == 503 else {}
            raise HTTPException(
                status_code=status,
                detail={"error_code": error_code, "message": message},
                headers=headers,
            )

            
# ==========================================
# MODEL 3: Angle (Image + Angle -> Image)
# ==========================================
@app.post("/angle", responses=GPU_ERROR_RESPONSES)
async def change_angle(
    file: UploadFile = File(...),
    mode: str = Form("custom", description="'custom' for single angle, 'multi' for 3 views"),
    azimuth: float = Form(0, description="Horizontal angle (0-360)"),
    elevation: float = Form(0, description="Vertical angle (-30 to 60)"),
    distance: float = Form(1.0, description="Distance (0.6, 1.0, 1.8)")
):
    """
    [Model 3] Qwen-Image-Edit-2511 + Angle LoRAs
    輸入: 圖片 + 角度參數
    功能: 旋轉物體視角 (需載入 LoRA)
    """
    # 先驗證，再建立目錄與寫檔，避免無效請求留下垃圾檔案
    _validate_image_upload(file, "file")

    if mode not in ("custom", "multi"):
        raise HTTPException(
            status_code=422,
            detail=f"Invalid mode '{mode}'. Must be 'custom' or 'multi'.",
        )

    request_id = str(uuid.uuid4())
    req_dir = os.path.join(OUTPUT_DIR, request_id)
    os.makedirs(req_dir, exist_ok=True)

    input_path = os.path.join(req_dir, "input.png")
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    print(f"🔄 [Angle] ID: {request_id} | Mode: {mode}")

    # 準備 Prompts
    prompts_map = {}
    if mode == "multi":
        prompts_map["right"] = build_angle_prompt(90, 0, 1.0)
        prompts_map["back"] = build_angle_prompt(180, 0, 1.0)
        prompts_map["left"] = build_angle_prompt(270, 0, 1.0)
    else:
        # Custom single view
        prompts_map["custom"] = build_angle_prompt(azimuth, elevation, distance)

    async with gpu_lock:
        def run_inference():
            pipe = None
            try:
                print("⏳ Loading Qwen-Image-Edit-2511 with LoRAs...")
                pipe = QwenImageEditPlusPipeline.from_pretrained(
                    "Qwen/Qwen-Image-Edit-2511",
                    torch_dtype=DTYPE
                ).to(DEVICE)

                # 載入 LoRA
                print("   Loading Adapters (Lightning + Angle)...")
                pipe.load_lora_weights(
                    "lightx2v/Qwen-Image-Edit-2511-Lightning",
                    weight_name="Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
                    adapter_name="lightning"
                )
                pipe.load_lora_weights(
                    "fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA",
                    weight_name="qwen-image-edit-2511-multiple-angles-lora.safetensors",
                    adapter_name="angles"
                )
                pipe.set_adapters(["lightning", "angles"], adapter_weights=[1.0, 1.0])

                img_obj = Image.open(input_path).convert("RGB").resize((1024, 1024), Image.LANCZOS)
                results = {}
                seed = random.randint(0, MAX_SEED)

                for key, prompt_str in prompts_map.items():
                    print(f"   Generating view: {key} ({prompt_str})")
                    generator = torch.Generator(device=DEVICE).manual_seed(seed)
                    
                    out_img = pipe(
                        image=[img_obj],
                        prompt=prompt_str,
                        height=1024,
                        width=1024,
                        num_inference_steps=4, # Lightning LoRA 只需要很少步數
                        generator=generator,
                        guidance_scale=1.0,
                        num_images_per_prompt=1,
                    ).images[0]
                    
                    filename = f"output_{key}.png"
                    save_image(out_img, req_dir, filename)
                    results[key] = f"/download/{request_id}/{filename}"
                
                return results

            finally:
                if pipe: del pipe
                flush_gpu()

        try:
            results = await run_in_threadpool(run_inference)
            return {
                "status": "success",
                "request_id": request_id,
                "input_url": f"/download/{request_id}/input.png",
                "results": results
            }
        except Exception as e:
            status, error_code, message = classify_exception(e)
            print(f"❌ [{error_code}] request_id={request_id} | {type(e).__name__}: {e}")
            traceback.print_exc()
            if error_code == "GPU_OOM":
                flush_gpu()
            headers = {"Retry-After": "30"} if status == 503 else {}
            raise HTTPException(
                status_code=status,
                detail={"error_code": error_code, "message": message},
                headers=headers,
            )


# --- Common: Download & Cleanup ---
@app.get("/download/{request_id}/{file_name}")
async def download_file(request_id: str, file_name: str):
    file_path = os.path.join(OUTPUT_DIR, request_id, file_name)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)

@app.on_event("shutdown")
async def cleanup():
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    print("🧹 Temporary files cleaned up")

if __name__ == "__main__":
    import uvicorn
    # 啟動 Server
    uvicorn.run(app, host="0.0.0.0", port=8190)
