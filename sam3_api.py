# Copyright (c) Meta Platforms, Inc. and affiliates.
# SAM3 API - FastAPI wrapper for SAM3 2D Interactive Segmentation (SAM1 task)

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import sys
import uuid
import shutil
import tempfile
import numpy as np
from PIL import Image
from typing import List, Optional
import json

app = FastAPI(title="SAM3 API", description="2D Interactive Segmentation using SAM3")

# CORS 設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 設定環境變數
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# 臨時輸出目錄
OUTPUT_DIR = tempfile.mkdtemp()
print(f"SAM3 輸出目錄: {OUTPUT_DIR}")

# 全域模型與處理器
model = None
processor = None
device = None

@app.on_event("startup")
async def load_model():
    global model, processor, device
    try:
        import torch
        import sam3
        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        # 選擇設備
        if torch.cuda.is_available():
            device = torch.device("cuda")
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        print(f"使用設備: {device}")

        # 取得 sam3 root 路徑
        sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
        bpe_path = f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"

        # 載入模型
        model = build_sam3_image_model(enable_inst_interactivity=True)
        processor = Sam3Processor(model)

        print("✅ SAM3 模型載入成功")
    except Exception as e:
        print(f"❌ SAM3 模型載入失敗: {str(e)}")
        raise RuntimeError(f"SAM3 模型載入失敗: {str(e)}")


# 儲存 inference_state 的字典（用於多步驟互動）
inference_states = {}


@app.post("/set_image")
async def set_image(image: UploadFile = File(..., description="要分割的圖片")):
    """
    設定要進行分割的圖片，返回 session_id 用於後續預測。
    
    這會計算圖片的 embedding，後續可以用 session_id 進行多次分割預測。
    """
    global model, processor, inference_states

    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="模型尚未載入")

    try:
        session_id = str(uuid.uuid4())
        work_dir = os.path.join(OUTPUT_DIR, session_id)
        os.makedirs(work_dir, exist_ok=True)

        # 儲存圖片
        image_path = os.path.join(work_dir, "image.png")
        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        # 載入圖片並設定到處理器
        pil_image = Image.open(image_path)
        original_mode = pil_image.mode
        print(f"原始圖片模式: {original_mode}, 尺寸: {pil_image.size}")
        
        # 確保圖片是 RGB 格式（SAM3 需要 3 通道）
        if pil_image.mode == 'RGBA':
            # 將 RGBA 轉換為 RGB，透明部分變白色背景
            background = Image.new('RGB', pil_image.size, (255, 255, 255))
            background.paste(pil_image, mask=pil_image.split()[3])  # 使用 alpha 作為 mask
            pil_image = background
            print(f"已將 RGBA 轉換為 RGB (白色背景)")
        elif pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
            print(f"已將 {original_mode} 轉換為 RGB")
        
        # 最終驗證：確保是 RGB
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
            print(f"強制轉換為 RGB")
        
        # 驗證通道數
        import numpy as np
        img_array = np.array(pil_image)
        print(f"最終圖片 shape: {img_array.shape}, mode: {pil_image.mode}")
        if len(img_array.shape) == 3 and img_array.shape[2] == 4:
            # 如果還是 4 通道，強制取前 3 個
            img_array = img_array[:, :, :3]
            pil_image = Image.fromarray(img_array)
            print(f"強制移除 alpha 通道")
        
        # 重新保存為 RGB 格式（確保磁碟上的檔案也是 RGB）
        pil_image.save(image_path, 'PNG')
        print(f"已將 RGB 圖片保存至: {image_path}")
        
        inference_state = processor.set_image(pil_image)

        # 儲存 inference_state
        inference_states[session_id] = {
            "state": inference_state,
            "image_path": image_path,
            "image_size": pil_image.size,  # (width, height)
            "last_logits": None
        }

        return {
            "session_id": session_id,
            "image_size": {"width": pil_image.size[0], "height": pil_image.size[1]},
            "message": "圖片已設定，可以開始分割"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict")
async def predict(
    session_id: str = Form(..., description="set_image 返回的 session_id"),
    point_coords: str = Form(None, description="點座標 JSON，格式: [[x1,y1], [x2,y2], ...]"),
    point_labels: str = Form(None, description="點標籤 JSON，格式: [1, 0, ...]（1=前景, 0=背景）"),
    box: str = Form(None, description="框座標 JSON，格式: [x1, y1, x2, y2]"),
    use_previous_mask: bool = Form(False, description="是否使用上一次預測的 mask 作為輸入"),
    multimask_output: bool = Form(True, description="是否輸出多個 mask")
):
    """
    根據點或框提示進行分割預測。
    
    Returns:
        - masks: 二進制 mask 圖片 (PNG)
        - scores: 每個 mask 的分數
    """
    global model, inference_states

    if model is None:
        raise HTTPException(status_code=503, detail="模型尚未載入")

    if session_id not in inference_states:
        raise HTTPException(status_code=404, detail="Session 不存在，請先呼叫 /set_image")

    try:
        session = inference_states[session_id]
        inference_state = session["state"]

        # 解析提示
        input_point = None
        input_label = None
        input_box = None
        mask_input = None

        if point_coords:
            input_point = np.array(json.loads(point_coords))
        if point_labels:
            input_label = np.array(json.loads(point_labels))
        if box:
            input_box = np.array(json.loads(box))

        # 使用上一次的 mask
        if use_previous_mask and session.get("last_logits") is not None:
            last_logits = session["last_logits"]
            mask_input = last_logits[np.argmax(session.get("last_scores", [0])), :, :]
            mask_input = mask_input[None, :, :]

        # 執行預測
        masks, scores, logits = model.predict_inst(
            inference_state,
            point_coords=input_point,
            point_labels=input_label,
            box=input_box[None, :] if input_box is not None else None,
            mask_input=mask_input,
            multimask_output=multimask_output,
        )

        # 按分數排序
        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind]
        scores = scores[sorted_ind]
        logits = logits[sorted_ind]

        # 儲存 logits 供下次使用
        session["last_logits"] = logits
        session["last_scores"] = scores

        # 儲存 mask 圖片
        work_dir = os.path.dirname(session["image_path"])
        mask_paths = []

        for i, mask in enumerate(masks):
            # 轉換為 PNG (0/255)
            mask_img = Image.fromarray((mask * 255).astype(np.uint8))
            mask_path = os.path.join(work_dir, f"mask_{i}.png")
            mask_img.save(mask_path)
            mask_paths.append(f"/download/{session_id}/mask_{i}.png")

        return {
            "session_id": session_id,
            "mask_count": len(masks),
            "masks": mask_paths,
            "scores": scores.tolist(),
            "best_mask": mask_paths[0] if mask_paths else None
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_and_apply")
async def predict_and_apply(
    session_id: str = Form(...),
    point_coords: str = Form(None),
    point_labels: str = Form(None),
    use_previous_mask: bool = Form(False),
    return_rgba: bool = Form(True, description="是否返回 RGBA 去背圖")
):
    """
    分割並直接返回最佳的 RGBA 去背圖（alpha = mask）
    """
    global model, inference_states

    if model is None:
        raise HTTPException(status_code=503, detail="模型尚未載入")

    if session_id not in inference_states:
        raise HTTPException(status_code=404, detail="Session 不存在")

    try:
        session = inference_states[session_id]
        inference_state = session["state"]

        # 解析提示
        input_point = np.array(json.loads(point_coords)) if point_coords else None
        input_label = np.array(json.loads(point_labels)) if point_labels else None

        mask_input = None
        if use_previous_mask and session.get("last_logits") is not None:
            last_logits = session["last_logits"]
            mask_input = last_logits[np.argmax(session.get("last_scores", [0])), :, :]
            mask_input = mask_input[None, :, :]

        # 預測（單一輸出）
        masks, scores, logits = model.predict_inst(
            inference_state,
            point_coords=input_point,
            point_labels=input_label,
            mask_input=mask_input,
            multimask_output=False,
        )

        # 儲存
        session["last_logits"] = logits
        session["last_scores"] = scores

        best_mask = masks[0]
        work_dir = os.path.dirname(session["image_path"])

        if return_rgba:
            # 載入原圖並套用 mask 成 RGBA
            original = Image.open(session["image_path"]).convert("RGBA")
            original_np = np.array(original)

            # 套用 mask 作為 alpha
            original_np[:, :, 3] = (best_mask * 255).astype(np.uint8)

            rgba_img = Image.fromarray(original_np)
            rgba_path = os.path.join(work_dir, "rgba_output.png")
            rgba_img.save(rgba_path)
            
            # 也儲存 mask 圖片
            mask_img = Image.fromarray((best_mask * 255).astype(np.uint8))
            mask_path = os.path.join(work_dir, "mask_best.png")
            mask_img.save(mask_path)

            return {
                "session_id": session_id,
                "score": float(scores[0]),
                "rgba_image": f"/download/{session_id}/rgba_output.png",
                "mask": f"/download/{session_id}/mask_best.png"
            }
        else:
            mask_img = Image.fromarray((best_mask * 255).astype(np.uint8))
            mask_path = os.path.join(work_dir, "mask_best.png")
            mask_img.save(mask_path)

            return {
                "session_id": session_id,
                "score": float(scores[0]),
                "mask": f"/download/{session_id}/mask_best.png"
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """刪除 session 並清理資源"""
    if session_id in inference_states:
        work_dir = os.path.dirname(inference_states[session_id]["image_path"])
        shutil.rmtree(work_dir, ignore_errors=True)
        del inference_states[session_id]
        return {"message": "Session 已刪除"}
    else:
        raise HTTPException(status_code=404, detail="Session 不存在")


@app.get("/download/{session_id}/{file_name}")
async def download_file(session_id: str, file_name: str):
    """下載生成的檔案"""
    file_path = os.path.join(OUTPUT_DIR, session_id, file_name)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="找不到檔案")
    return FileResponse(file_path, media_type='image/png', filename=file_name)


@app.get("/health")
async def health_check():
    """健康檢查"""
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "active_sessions": len(inference_states)
    }


@app.on_event("shutdown")
async def cleanup():
    """清理所有臨時檔案"""
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    print("🧹 SAM3 臨時檔案已清理")


@app.post("/apply_last_mask")
async def apply_last_mask(
    session_id: str = Form(..., description="set_image 返回的 session_id"),
    mask_index: int = Form(0, description="要套用的 mask 索引 (從 /predict 返回的列表順序)，預設 0 為最佳"),
    return_rgba: bool = Form(True, description="是否返回 RGBA 去背圖")
):
    """
    不經過模型運算，直接使用上一次 /predict 生成的 Mask 來進行去背或輸出。
    適用於使用者在前端已經預覽過 Mask，並決定「套用」的情境。
    """
    global inference_states

    # 1. 檢查 Session
    if session_id not in inference_states:
        raise HTTPException(status_code=404, detail="Session 不存在")

    try:
        session = inference_states[session_id]
        work_dir = os.path.dirname(session["image_path"])
        
        # 2. 尋找上一次 /predict 生成的 Mask 檔案
        # 在 /predict 中，檔案命名格式為 mask_{i}.png
        mask_filename = f"mask_{mask_index}.png"
        mask_path = os.path.join(work_dir, mask_filename)

        if not os.path.exists(mask_path):
            raise HTTPException(status_code=400, detail=f"找不到索引為 {mask_index} 的 Mask，請確認是否已呼叫過 /predict")

        # 3. 讀取 Mask
        # 讀取為 'L' (灰階) 模式，預期數值為 0 或 255
        best_mask_img = Image.open(mask_path).convert("L")
        
        # 4. 處理輸出
        if return_rgba:
            # 載入原圖
            original = Image.open(session["image_path"]).convert("RGBA")
            
            # 確保尺寸一致 (防呆)
            if best_mask_img.size != original.size:
                best_mask_img = best_mask_img.resize(original.size, Image.NEAREST)

            original_np = np.array(original)
            mask_np = np.array(best_mask_img)

            # 將 mask 套用到 alpha channel (假設 mask 已經是 0/255)
            # 這裡直接使用 mask 圖片的數值作為 alpha
            original_np[:, :, 3] = mask_np

            # 儲存結果
            rgba_img = Image.fromarray(original_np)
            
            # 使用 unique name 或固定 name，視需求而定
            output_filename = f"rgba_applied_{mask_index}.png"
            rgba_path = os.path.join(work_dir, output_filename)
            rgba_img.save(rgba_path)

            return {
                "session_id": session_id,
                "applied_mask_index": mask_index,
                "rgba_image": f"/download/{session_id}/{output_filename}",
                "mask": f"/download/{session_id}/{mask_filename}",
                "message": "已套用上一次的預測結果"
            }
        else:
            # 如果只要確認 mask 路徑
            return {
                "session_id": session_id,
                "applied_mask_index": mask_index,
                "mask": f"/download/{session_id}/{mask_filename}"
            }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
        
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)