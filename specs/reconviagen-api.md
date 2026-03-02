# ReconViaGen API (TRELLIS-VGGT)

**File:** `api_collect/reconviagen_api_v4.py`
**Port:** `52069`
**Model:** `esther11/trellis-vggt-v0-2` (`TrellisVGGTTo3DPipeline`)
**Pattern:** Synchronous (long-polling) with lazy model loading.

---

## Error Response Format

All non-2xx responses (except `422` FastAPI validation errors and `400`) return a structured JSON body:

```json
{
  "detail": {
    "error_code": "GPU_OOM",
    "message": "GPU out of memory. Free some VRAM and retry."
  }
}
```

| `error_code` | HTTP | Retryable | Meaning |
|---|---|---|---|
| `GPU_OOM` | `503` | ✅ (after delay) | GPU out of memory — `Retry-After: 30` header is included |
| `MODEL_UNAVAILABLE` | `503` | ✅ (after delay) | Model failed to load |
| `DISK_FULL` | `507` | ❌ | Server disk full — human intervention required |
| `INFERENCE_ERROR` | `500` | ❌ | Unhandled inference exception |

> **Log format:** Every error prints to stdout: `❌ [ERROR_CODE] request_id=<uuid> | ExceptionType: message` followed by a full traceback.

---

## Startup Behaviour

The model is **not** loaded at startup. It is loaded lazily on the first request inside the `gpu_lock`. Subsequent requests reuse the in-memory pipeline.

```python
ensure_model_loaded()  # called inside gpu_lock at each request
```

If model loading fails during the first request, a `503 MODEL_UNAVAILABLE` is returned and `pipeline` stays `None`, so every subsequent request also attempts to load (and fails) until the server is restarted.

> **Warning:** These endpoints are synchronous. For large images or complex geometry, inference can take several minutes. Configure HTTP client timeouts accordingly (recommend ≥ 10 minutes).

---

## Shared Generation Parameters

All three generation endpoints accept the following form fields. Constraints are enforced by Pydantic — values out of range return `422`.

| Parameter | Type | Default | Range | Description |
|---|---|---|---|---|
| `seed` | `integer` | `0` | — | RNG seed |
| `simplify` | `float` | `0.95` | `0.9` – `0.98` | Mesh simplification ratio. Lower = fewer faces. |
| `texture_size` | `integer` | `1024` | `512` – `2048` | Texture atlas resolution in pixels. Recommended steps: 512, 1024, 1536, 2048. |
| `ss_guidance_strength` | `float` | `7.5` | `0.0` – `10.0` | **[Stage 1]** Sparse structure — CFG strength |
| `ss_sampling_steps` | `integer` | `30` | `1` – `50` | **[Stage 1]** Sparse structure — diffusion steps |
| `slat_guidance_strength` | `float` | `3.0` | `0.0` – `10.0` | **[Stage 2]** Structured latent — CFG strength |
| `slat_sampling_steps` | `integer` | `12` | `1` – `50` | **[Stage 2]** Structured latent — diffusion steps |

---

## Endpoints

### `POST /generate-single`

Generate a 3D model from a single image.

**Request** — `multipart/form-data`

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | `UploadFile` | ✅ | Input image (PNG/JPG/WEBP). RGBA images are accepted; background removal is done internally. |
| *(shared params)* | Form fields | ❌ | See shared params above |

**Success Response** — `200 OK`

```json
{
  "gaussian_video": "/download/{request_id}/gs.mp4",
  "radiance_video": "/download/{request_id}/rf.mp4",
  "mesh_video":     "/download/{request_id}/mesh.mp4",
  "glb_file":       "/download/{request_id}/output.glb",
  "ply_file":       "/download/{request_id}/output.ply"
}
```

| Field | Type | Description |
|---|---|---|
| `gaussian_video` | `string` | 120-frame Gaussian splat orbit video (15 fps) |
| `radiance_video` | `string` | Same as `gaussian_video` (copy, for frontend compatibility) |
| `mesh_video` | `string` | 120-frame mesh normal-shaded orbit video (15 fps) |
| `glb_file` | `string` | Final textured GLB |
| `ply_file` | `string` | Gaussian splat PLY |

**Error Responses**

| Status | Condition | `error_code` |
|---|---|---|
| `422` | Missing `file` | FastAPI validation (plain string) |
| `503` | GPU OOM during model load or inference | `GPU_OOM` |
| `503` | Model failed to load | `MODEL_UNAVAILABLE` |
| `507` | Server disk full | `DISK_FULL` |
| `500` | Unhandled inference / export exception | `INFERENCE_ERROR` |

---

### `POST /generate-batch`

Process multiple images sequentially, each generating an independent 3D model.

**Request** — `multipart/form-data`

| Field | Type | Required | Description |
|---|---|---|---|
| `files` | `List[UploadFile]` | ✅ | One or more images. Each is processed as single-image inference in order. |
| *(shared params)* | Form fields | ❌ | Applied to every image in the batch |

**Success Response** — `200 OK`

```json
{
  "total_count": 3,
  "results": [
    {
      "original_filename": "car.png",
      "status": "success",
      "gaussian_video": "/download/.../gs.mp4",
      "radiance_video": "/download/.../rf.mp4",
      "mesh_video":     "/download/.../mesh.mp4",
      "glb_file":       "/download/.../output.glb",
      "ply_file":       "/download/.../output.ply"
    },
    {
      "original_filename": "chair.png",
      "status": "failed",
      "error_code": "GPU_OOM",
      "error": "GPU out of memory. Free some VRAM and retry."
    }
  ]
}
```

**HTTP Status Rules:**

| Condition | HTTP Status |
|---|---|
| All items succeeded | `200 OK` |
| Some items succeeded, some failed | `207 Multi-Status` |
| All items failed | `500 Internal Server Error` |
| No files uploaded | `400 Bad Request` |

> **Note:** The HTTP status reflects the overall batch outcome. Always check `results[i].status` for per-item status.

**Error Responses**

| Status | Condition | `error_code` / `detail` |
|---|---|---|
| `400` | Zero files uploaded | plain string: `"請至少上傳一張圖片"` |
| `500` | All items in batch failed | plain string: `"All <N> items failed. See 'results' for per-item errors."` |
| `503` | Unexpected GPU OOM outside item loop | `GPU_OOM` |
| `507` | Server disk full | `DISK_FULL` |
| `500` | Unexpected error outside item loop | `INFERENCE_ERROR` |

> **Per-item errors** in `results[]` now include an `error_code` field alongside `error` for programmatic handling.

---

### `POST /generate-multi`

Generate a single 3D model from multiple images of the same subject simultaneously (multi-view fusion).

**Request** — `multipart/form-data`

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `files` | `List[UploadFile]` | ✅ | — | 2–N images of the same object from different angles |
| `multiimage_algo` | `"stochastic" \| "multidiffusion"` (Form) | ❌ | `"multidiffusion"` | Fusion algorithm. Validated by `Literal` type — invalid values return `422`. |
| *(shared params)* | Form fields | ❌ | — | See shared params |

**Success Response** — `200 OK`

Same schema as `/generate-single`.

**Error Responses**

| Status | Condition | `error_code` |
|---|---|---|
| `400` | Zero files uploaded | plain string: `"請至少上傳一張圖片"` |
| `422` | Missing `files` field | FastAPI validation (plain string) |
| `422` | Invalid `multiimage_algo` | FastAPI validation (plain string) |
| `503` | GPU OOM during model load or inference | `GPU_OOM` |
| `503` | Model failed to load | `MODEL_UNAVAILABLE` |
| `500` | Unhandled inference exception | `INFERENCE_ERROR` |

---

### `GET /download/{request_id}/{file_name}`

Download a generated output file.

| Param | Type | Description |
|---|---|---|
| `request_id` | `string` | UUID4 (with optional `_batch_{i}` suffix for batch items) |
| `file_name` | `string` | e.g. `output.glb`, `output.ply`, `gs.mp4`, `mesh.mp4`, `rf.mp4` |

**Success Response** — `200 OK`

Content-Type set by file extension:
- `.mp4` → `video/mp4`
- `.glb` → `model/gltf-binary`
- other → `application/octet-stream`

**Error Responses**

| Status | Condition |
|---|---|
| `404` | File not found |

---

## Known Limitations

- Endpoints are synchronous; very long jobs may hit HTTP gateway/proxy timeouts. Configure client timeout ≥ 10 minutes.
- GPU lock serialises all requests.
