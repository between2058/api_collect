# SAM-3D Image-to-3D API

**File:** `api_collect/sam3d_api_v2.py`
**Port:** `8001`
**Model:** SAM3D (`Inference` class from `notebook/inference.py`, checkpoint: `checkpoints/hf/pipeline.yaml`)
**Pattern:** Synchronous.

---

## Startup Behaviour

The model is loaded at startup via `@app.on_event("startup")`. If loading fails, `RuntimeError` is raised — the server will be in a broken state and return `503` on model-dependent endpoints.

> **Note:** Inference in `/generate` and `/generate-batch` is synchronous and can take several minutes. Set HTTP client timeouts accordingly.

---

## Endpoints

### `GET /health`

Check server and model status.

**Response** — `200 OK`

```json
{
  "status": "ok",
  "model_loaded": true
}
```

| Field | Type | Description |
|---|---|---|
| `status` | `string` | Always `"ok"` if server is running |
| `model_loaded` | `boolean` | Whether the SAM3D inference object is ready |

---

### `POST /generate`

Generate a 3D GLB from a full image and a segmentation mask image.

**Request** — `multipart/form-data`

| Field | Type | Required | Description |
|---|---|---|---|
| `image` | `UploadFile` | ✅ | Full scene image (RGBA). Used as the colour reference. |
| `mask_image` | `UploadFile` | ✅ | Segmented object image (RGBA with transparent background). The alpha channel is used as the binary mask. |
| `seed` | `integer` (Form) | ❌ (default `42`) | RNG seed |

**Mask loading logic:**
- Mask is converted to uint8 (`> 0 → True`)
- If 3D array (H, W, C), the **last channel** (alpha) is used as the mask

**Success Response** — `200 OK`

```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "glb_file": "/download/550e8400-e29b-41d4-a716-446655440000/output.glb"
}
```

| Field | Type | Description |
|---|---|---|
| `request_id` | `string` (UUID4) | Unique ID for this request |
| `glb_file` | `string` | Download URL for the generated GLB |

**Error Responses**

| Status | Condition | `detail` |
|---|---|---|
| `503` | Model not loaded (startup failure) | `"模型尚未載入"` |
| `422` | Missing `image` or `mask_image` | FastAPI validation |
| `500` | Image or mask cannot be read | `"<PIL/IO exception>"` |
| `500` | Inference failure | `"<exception message>"` |
| `500` | GLB export failure | `"<exception message>"` |

---

### `POST /generate-batch`

Generate multiple 3D GLB models from one image and multiple segmentation masks. Each mask produces one GLB. Applies coordinate system correction (Y-Up → Z-Up) and predicted scale/rotation/translation from the model.

**Request** — `multipart/form-data`

| Field | Type | Required | Description |
|---|---|---|---|
| `image` | `UploadFile` | ✅ | Full scene image (RGBA) |
| `mask_images` | `List[UploadFile]` | ✅ | One or more segmentation mask images (RGBA, transparent background). Each produces one output GLB. |
| `seed` | `integer` (Form) | ❌ (default `42`) | RNG seed applied to every item |

**Transformation applied per mask:**
1. Coordinate conversion: Y-Up → Z-Up (mesh vertices rotated by `R_YUP_TO_ZUP`)
2. Predicted `scale`, `rotation` (quaternion → matrix), `translation` applied via `compose_transform`

**Success Response** — `200 OK`

```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "count": 3,
  "glb_files": [
    "/download/550e8400.../output_0.glb",
    "/download/550e8400.../output_1.glb",
    "/download/550e8400.../output_2.glb"
  ]
}
```

| Field | Type | Description |
|---|---|---|
| `request_id` | `string` (UUID4) | Shared request ID for all outputs |
| `count` | `integer` | Number of GLBs generated |
| `glb_files` | `List[string]` | Download URLs, indexed to match `mask_images` order |

**Error Responses**

| Status | Condition | `detail` |
|---|---|---|
| `503` | Model not loaded | `"模型尚未載入"` |
| `422` | Missing `image` or `mask_images` | FastAPI validation |
| `500` | Any mask image unreadable, inference failure, or coordinate transformation failure | `"<exception message>"` |

> **Note:** If any single mask fails, the entire batch returns `500`. There is no partial-success response.

---

### `GET /download/{request_id}/{file_name}`

Download a generated GLB.

| Param | Type | Description |
|---|---|---|
| `request_id` | `string` | UUID4 from `/generate` or `/generate-batch` |
| `file_name` | `string` | e.g. `output.glb`, `output_0.glb` |

**Success Response** — `200 OK`
Content-Type: `application/octet-stream`

**Error Responses**

| Status | Condition |
|---|---|
| `404` | File not found |

---

## Typical Workflow

```
# Step 1: Set image in SAM3 API (port 8002)
POST http://localhost:8002/set_image   → { "session_id": "..." }

# Step 2: Predict masks with point/box prompts
POST http://localhost:8002/predict     → { "masks": [...], "scores": [...] }

# Step 3: Get RGBA cutout
POST http://localhost:8002/predict_and_apply → { "rgba_image": "..." }

# Step 4: Generate 3D from original image + mask
POST http://localhost:8001/generate
  image=<original RGBA>
  mask_image=<RGBA from predict_and_apply>
  → { "glb_file": "..." }

# Or batch: generate from multiple masks at once
POST http://localhost:8001/generate-batch
  image=<original RGBA>
  mask_images[]=<mask_0.png>
  mask_images[]=<mask_1.png>
  → { "glb_files": ["...", "..."] }
```

---

## Known Limitations

- Both endpoints are fully synchronous with no progress feedback. For long meshes, configure client timeout accordingly.
- Output files are cleaned up automatically after **1 hour**. The cleanup loop runs every 30 minutes.
- `/generate-batch`: returns `207 Multi-Status` on partial failure, `500` if all masks fail. Check `results[i].status` per mask.
