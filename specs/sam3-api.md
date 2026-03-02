# SAM3 2D Interactive Segmentation API

**File:** `api_collect/sam3_api.py`
**Port:** `8002`
**Model:** SAM3 (`build_sam3_image_model`, `Sam3Processor`)
**Pattern:** Synchronous, session-based. Each image upload creates a persistent `session_id`; subsequent predict calls use the cached image embedding.

---

## Startup Behaviour

The model is loaded at server startup via `@app.on_event("startup")`. If loading fails, `RuntimeError` is raised — the server will be in a broken state and return `503` on model-dependent endpoints.

Device selection priority: `CUDA → MPS → CPU`.

---

## Session Lifecycle

```
POST /set_image   →  { "session_id": "..." }
                         ↓ (can call multiple times)
POST /predict     →  { "masks": [...], "scores": [...] }
POST /predict     →  { "masks": [...], use_previous_mask=true to refine }
                         ↓ optional
POST /apply_last_mask →  { "rgba_image": "..." }
                         ↓
DELETE /session/{session_id}   ← clean up when done
```

Sessions are held in memory. There is no automatic expiration — call `DELETE /session/{session_id}` when finished to free disk space.

---

## Endpoints

### `GET /health`

Check server and model status.

**Response** — `200 OK`

```json
{
  "status": "ok",
  "model_loaded": true,
  "active_sessions": 3
}
```

---

### `POST /set_image`

Upload an image and compute its embedding. Returns a `session_id` for subsequent predict calls.

**Request** — `multipart/form-data`

| Field | Type | Required | Description |
|---|---|---|---|
| `image` | `UploadFile` | ✅ | Input image. Supported: PNG/JPG (RGB or RGBA). RGBA is auto-converted to RGB with white background. |

**Preprocessing applied automatically:**
1. RGBA → RGB (white background composite)
2. Any non-RGB mode → RGB conversion
3. Image saved as PNG to session directory

**Success Response** — `200 OK`

```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "image_size": {
    "width": 1024,
    "height": 768
  },
  "message": "圖片已設定，可以開始分割"
}
```

| Field | Type | Description |
|---|---|---|
| `session_id` | `string` (UUID4) | Use in all subsequent calls |
| `image_size` | `object` | Actual dimensions of the stored image after preprocessing |

**Error Responses**

| Status | Condition | `detail` |
|---|---|---|
| `503` | Model not loaded (startup failure) | `"模型尚未載入"` |
| `500` | Cannot open/read uploaded image | `"<PIL/IO exception>"` |
| `500` | Embedding computation failed | `"<model exception>"` |

---

### `POST /predict`

Run segmentation inference using point and/or box prompts. At least one of `point_coords`/`box` must be provided.

**Request** — `multipart/form-data`

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `session_id` | `string` (Form) | ✅ | — | Session ID from `/set_image` |
| `point_coords` | `string` (Form) | ❌* | `null` | JSON array of `[x, y]` coordinates: `"[[100, 200], [150, 250]]"` |
| `point_labels` | `string` (Form) | ❌ | `null` | JSON array of labels (same length as `point_coords`): `"[1, 0]"`. `1` = foreground, `0` = background. Required if `point_coords` is provided. |
| `box` | `string` (Form) | ❌* | `null` | JSON array `[x1, y1, x2, y2]`: `"[10, 20, 300, 400]"` |
| `use_previous_mask` | `boolean` (Form) | ❌ | `false` | If `true`, the best mask logits from the last `/predict` call are fed as `mask_input` for refinement |
| `multimask_output` | `boolean` (Form) | ❌ | `true` | If `true`, returns 3 candidate masks sorted by score. If `false`, returns 1 mask. |

*At least one of `point_coords` or `box` must be provided. Providing neither will call the model with all-null inputs, which may produce undefined results.

**Coordinates:** All coordinates are in **pixel space** relative to the image stored in the session.

**Success Response** — `200 OK`

```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "mask_count": 3,
  "masks": [
    "/download/550e8400.../mask_0.png",
    "/download/550e8400.../mask_1.png",
    "/download/550e8400.../mask_2.png"
  ],
  "scores": [0.98, 0.87, 0.61],
  "best_mask": "/download/550e8400.../mask_0.png"
}
```

| Field | Type | Description |
|---|---|---|
| `mask_count` | `integer` | Number of masks returned (1 or 3) |
| `masks` | `List[string]` | Download URLs for binary mask PNGs (0 = background, 255 = foreground), sorted best-first |
| `scores` | `List[float]` | Model confidence scores, one per mask, sorted descending |
| `best_mask` | `string \| null` | URL of the highest-confidence mask |

**Mask PNG format:** Single-channel (grayscale), pixel values are `0` or `255`.

**Error Responses**

| Status | Condition | `detail` |
|---|---|---|
| `503` | Model not loaded | `"模型尚未載入"` |
| `404` | Unknown `session_id` | `"Session 不存在，請先呼叫 /set_image"` |
| `422` | Malformed JSON in `point_coords`, `point_labels`, or `box` | `"<json parse error>"` |
| `500` | Inference failure | `"<exception message>"` |

---

### `POST /predict_and_apply`

Run segmentation and immediately return the best mask applied as an alpha channel (RGBA cutout). Always uses `multimask_output=False` (single best mask).

**Request** — `multipart/form-data`

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `session_id` | `string` (Form) | ✅ | — | Session ID |
| `point_coords` | `string` (Form) | ❌ | `null` | JSON `[[x, y], ...]` |
| `point_labels` | `string` (Form) | ❌ | `null` | JSON `[1, 0, ...]` |
| `use_previous_mask` | `boolean` (Form) | ❌ | `false` | Use previous logits as mask input |
| `return_rgba` | `boolean` (Form) | ❌ | `true` | If `true`, return RGBA cutout; if `false`, return only the mask PNG |

**Success Response (`return_rgba=true`)** — `200 OK`

```json
{
  "session_id": "550e8400-...",
  "score": 0.97,
  "rgba_image": "/download/550e8400.../rgba_output.png",
  "mask": "/download/550e8400.../mask_best.png"
}
```

**Success Response (`return_rgba=false`)** — `200 OK`

```json
{
  "session_id": "550e8400-...",
  "score": 0.97,
  "mask": "/download/550e8400.../mask_best.png"
}
```

| Field | Type | Description |
|---|---|---|
| `score` | `float` | Confidence of the single returned mask |
| `rgba_image` | `string` | RGBA PNG — foreground preserved, background transparent |
| `mask` | `string` | Greyscale mask PNG |

**Error Responses**

| Status | Condition |
|---|---|
| `503` | Model not loaded |
| `404` | Unknown `session_id` |
| `500` | Inference / file write failure |

---

### `POST /apply_last_mask`

Apply a previously predicted mask without running inference again. Uses mask PNGs written by the last `/predict` call.

**Request** — `multipart/form-data`

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `session_id` | `string` (Form) | ✅ | — | Session ID |
| `mask_index` | `integer` (Form) | ❌ | `0` | Index into the mask list from the last `/predict` response (`0` = best) |
| `return_rgba` | `boolean` (Form) | ❌ | `true` | Return RGBA cutout (true) or mask path only (false) |

**Success Response (`return_rgba=true`)** — `200 OK`

```json
{
  "session_id": "550e8400-...",
  "applied_mask_index": 0,
  "rgba_image": "/download/550e8400.../rgba_applied_0.png",
  "mask": "/download/550e8400.../mask_0.png",
  "message": "已套用上一次的預測結果"
}
```

**Success Response (`return_rgba=false`)** — `200 OK`

```json
{
  "session_id": "550e8400-...",
  "applied_mask_index": 0,
  "mask": "/download/550e8400.../mask_0.png"
}
```

**Error Responses**

| Status | Condition | `detail` |
|---|---|---|
| `404` | Unknown `session_id` | `"Session 不存在"` |
| `400` | `mask_index` refers to a mask that doesn't exist (no prior `/predict`) | `"找不到索引為 {i} 的 Mask，請確認是否已呼叫過 /predict"` |
| `500` | File read/write failure | `"<exception message>"` |

---

### `DELETE /session/{session_id}`

Delete a session and clean up all associated files.

**Path Parameters**

| Param | Type | Description |
|---|---|---|
| `session_id` | `string` | Session ID to delete |

**Success Response** — `200 OK`

```json
{
  "message": "Session 已刪除"
}
```

**Error Responses**

| Status | Condition |
|---|---|
| `404` | Unknown `session_id` |

---

### `GET /download/{session_id}/{file_name}`

Download a file generated within a session.

| Param | Type | Description |
|---|---|---|
| `session_id` | `string` | Session ID |
| `file_name` | `string` | e.g. `mask_0.png`, `rgba_output.png`, `mask_best.png` |

**Success Response** — `200 OK`
Content-Type: `image/png`

**Error Responses**

| Status | Condition |
|---|---|
| `404` | File not found |

---

## Known Limitations

- Sessions expire automatically after **1 hour** (configured via `SESSION_TTL_SECONDS`). The cleanup loop runs every 10 minutes. Calling `DELETE /session/{session_id}` immediately cleans up resources.
