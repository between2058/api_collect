# TRELLIS.2 3D Generation API

**File:** `api_collect/trellis_api.py`
**Port:** `8100`
**Model:** `microsoft/TRELLIS.2-4B` (`Trellis2ImageTo3DPipeline` + `Trellis2TexturingPipeline`)
**Pattern:** **Async Job** — endpoints return `job_id` immediately; inference runs in a `ThreadPoolExecutor(max_workers=1)`.

---

## Startup Behaviour

Both pipelines are loaded at startup via the `lifespan` context manager. If loading fails, the server will not finish starting. The server exposes endpoints only after both pipelines are ready.

```
TRELLIS.2-4B (image pipeline) → cuda
TRELLIS.2-4B (texture pipeline) → cuda
```

---

## Async Job Lifecycle

```
POST /generate/...  →  202 { "job_id": "...", "status": "pending" }
                               ↓ poll
GET /jobs/{job_id}  →  200 { "status": "running",   "percent": 40, "stage": "..." }
GET /jobs/{job_id}  →  200 { "status": "completed", "percent": 100, "result": { "glb_url": "..." } }
GET /jobs/{job_id}  →  200 { "status": "failed",    "error": "..." }
                               ↓ on completed
GET /download/{job_id}/output.glb
```

> **Jobs expire after 60 minutes.** The cleanup loop runs every 30 minutes. Expired jobs and their files are deleted.

---

## Shared Generation Parameters

The following parameters are common across `/generate/image`, `/generate/multiview`, and `/generate/text`. All have defaults and are optional.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `resolution` | `string` | `"1024"` | Output resolution. One of: `"512"`, `"1024"`, `"1536"` |
| `seed` | `integer` | `42` | RNG seed |
| `preprocess_image` | `boolean` | `true` | Auto-remove background before generation |
| `decimation_target` | `integer` | `200000` | Target face count after mesh decimation |
| `texture_size` | `integer` | `1024` | Texture atlas resolution in pixels |
| `ss_guidance_strength` | `float` | `7.5` | Sparse structure — guidance strength |
| `ss_guidance_rescale` | `float` | `0.7` | Sparse structure — guidance rescale |
| `ss_sampling_steps` | `integer` | `12` | Sparse structure — diffusion steps |
| `ss_rescale_t` | `float` | `0.5` | Sparse structure — rescale t |
| `shapslat_guidance_strength` | `float` | `3.0` | Shape SLaT — guidance strength |
| `shapslat_guidance_rescale` | `float` | `0.0` | Shape SLaT — guidance rescale |
| `shapslat_sampling_steps` | `integer` | `12` | Shape SLaT — diffusion steps |
| `shapslat_rescale_t` | `float` | `0.5` | Shape SLaT — rescale t |
| `texslat_guidance_strength` | `float` | `3.0` | Texture SLaT — guidance strength |
| `texslat_guidance_rescale` | `float` | `0.0` | Texture SLaT — guidance rescale |
| `texslat_sampling_steps` | `integer` | `12` | Texture SLaT — diffusion steps |
| `texslat_rescale_t` | `float` | `0.5` | Texture SLaT — rescale t |

---

## Endpoints

### `POST /generate/image`

Generate a 3D GLB from a single image.

**Request** — `multipart/form-data`

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | `UploadFile` | ✅ | Input image (PNG/JPG/WEBP). RGB image of the subject. |
| *(shared params)* | Form fields | ❌ | See [Shared Generation Parameters](#shared-generation-parameters) |

**Success Response** — `200 OK`

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending"
}
```

**Error Responses**

| Status | Condition |
|---|---|
| `422` | Missing `file` field |
| `500` | Cannot parse uploaded image |

---

### `POST /generate/multiview`

Generate a 3D GLB from multiple images of the same subject.

**Request** — `multipart/form-data`

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `files` | `List[UploadFile]` | ✅ | — | 2–N images of the same object from different angles |
| `multiview_mode` | `string` (Form) | ❌ | `"stochastic"` | Multi-image fusion mode. Supported: `"stochastic"`, `"multidiffusion"` |
| *(shared params)* | Form fields | ❌ | — | See shared params |

**Success Response** — `200 OK`

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending"
}
```

**Error Responses**

| Status | Condition |
|---|---|
| `422` | Missing `files` |
| `500` | Cannot parse one or more uploaded images |

---

### `POST /generate/text`

Generate a 3D GLB from a text prompt (internally calls the Qwen Image API at `$QWEN_API_URL` first to produce an image, then runs TRELLIS.2).

**Request** — `application/json`

```json
{
  "qwen": {
    "prompt": "A red sports car",
    "negative_prompt": "low quality, blurry",
    "aspect_ratio": "1:1",
    "num_steps": 50,
    "cfg_scale": 4.0,
    "seed": 42
  },
  "trellis": {
    "resolution": "1024",
    "seed": 42,
    "preprocess_image": true,
    "decimation_target": 200000,
    "texture_size": 1024
  }
}
```

| Top-level Field | Type | Required | Description |
|---|---|---|---|
| `qwen` | `object` | ✅ | Passed as-is to `POST {QWEN_API_URL}/text2img`. See [Qwen Image API](./qwen-image-api.md#post-text2img) for field definitions. |
| `trellis` | `object` | ❌ | Override [shared generation parameters](#shared-generation-parameters). Unset fields use defaults. |

**Environment Variable**

| Variable | Default | Description |
|---|---|---|
| `QWEN_API_URL` | `http://localhost:8190` | Base URL for the Qwen Image API |

**Success Response** — `200 OK`

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending"
}
```

**Error Responses**

| Status | Condition | `detail` |
|---|---|---|
| `422` | Malformed JSON body | FastAPI validation |
| `500` | Qwen API unreachable / returned error | Error captured in job `error` field, not HTTP status |

> **Note:** Qwen API connectivity errors are captured inside the background job. The endpoint itself returns 200 if the job was queued. Check `GET /jobs/{job_id}` — if status is `"failed"`, the `error` field contains the Qwen error.

---

### `POST /generate/batch`

Submit multiple images as independent generation jobs.

**Request** — `multipart/form-data`

| Field | Type | Required | Description |
|---|---|---|---|
| `files` | `List[UploadFile]` | ✅ | One or more images. Each becomes a separate job with default parameters. |

**Success Response** — `200 OK`

```json
{
  "job_ids": [
    "550e8400-e29b-41d4-a716-446655440000",
    "6ba7b810-9dad-11d1-80b4-00c04fd430c8"
  ],
  "status": "pending"
}
```

Poll each `job_id` independently via `GET /jobs/{job_id}`.

**Error Responses**

| Status | Condition |
|---|---|
| `422` | Missing `files` |
| `500` | Cannot parse one or more images at request time |

---

### `POST /texture`

Re-texture an existing untextured GLB mesh using a reference image.

**Request** — `multipart/form-data`

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `mesh` | `UploadFile` | ✅ | — | Input GLB (untextured mesh) |
| `reference` | `UploadFile` | ✅ | — | Reference image for texture generation |
| `seed` | `integer` (Form) | ❌ | `42` | RNG seed |
| `texture_size` | `integer` (Form) | ❌ | `1024` | Texture atlas resolution |
| `texslat_guidance_strength` | `float` (Form) | ❌ | `3.0` | Texture SLaT guidance strength |
| `texslat_guidance_rescale` | `float` (Form) | ❌ | `0.0` | Texture SLaT guidance rescale |
| `texslat_sampling_steps` | `integer` (Form) | ❌ | `12` | Texture SLaT diffusion steps |
| `texslat_rescale_t` | `float` (Form) | ❌ | `0.5` | Texture SLaT rescale t |

**Success Response** — `200 OK`

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending"
}
```

**Error Responses**

| Status | Condition |
|---|---|
| `422` | Missing `mesh` or `reference` |
| `500` | File write failure |

---

### `GET /jobs/{job_id}`

Poll the status of an async job.

**Path Parameters**

| Param | Type | Description |
|---|---|---|
| `job_id` | `string` | UUID4 from any generation endpoint |

**Response** — `200 OK`

```json
{
  "status": "pending | running | completed | failed",
  "percent": 88,
  "stage": "Extracting GLB...",
  "result": null,
  "error": null
}
```

| Field | Type | Description |
|---|---|---|
| `status` | `string` | `"pending"` / `"running"` / `"completed"` / `"failed"` |
| `percent` | `integer` | Progress `0–100` |
| `stage` | `string` | Human-readable current stage label |
| `result` | `object \| null` | On `"completed"`: `{ "glb_url": "/download/{job_id}/output.glb" }` |
| `error` | `string \| null` | On `"failed"`: error message from the background thread |

**Error Responses**

| Status | Condition |
|---|---|
| `404` | Unknown `job_id` (or job expired after 60 minutes) |

---

### `GET /download/{job_id}/{filename}`

Download a completed output file.

| Param | Type | Description |
|---|---|---|
| `job_id` | `string` | UUID4 |
| `filename` | `string` | e.g. `output.glb` |

**Success Response** — `200 OK`
Content-Type: `application/octet-stream`

**Error Responses**

| Status | Condition |
|---|---|
| `404` | File not found |

---

## Stage Labels Reference

| Stage string | Meaning |
|---|---|
| `"Queued"` | Job created, not started yet |
| `"Preprocessing image..."` | Image pre-processing / background removal |
| `"Stage 1: Generating sparse structure..."` | Sparse voxel grid generation |
| `"Stage 2: Generating 3D shape..."` | Shape SLaT decoding |
| `"Stage 3: Generating materials..."` | Texture SLaT decoding |
| `"Extracting GLB..."` | Mesh decimation + WebP texture export |
| `"Done ✓"` | Completed |
| `"Sending prompt to Qwen..."` | (text mode) Waiting for Qwen image |
| `"Image ready — starting TRELLIS.2..."` | (text mode) Image received, starting inference |

---

## Known Limitations

- Only one inference job runs at a time (single `ThreadPoolExecutor` worker). Additional submissions queue behind it.
