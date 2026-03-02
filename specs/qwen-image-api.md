# Qwen Image API

**File:** `api_collect/qwen_image_api.py`
**Port:** `8190`
**Models:**
- `Qwen/Qwen-Image-2512` — text-to-image
- `Qwen/Qwen-Image-Edit-2511` — image editing
- `Qwen/Qwen-Image-Edit-2511` + LoRAs — view angle synthesis
**Pattern:** Synchronous, serialised via `asyncio.Lock` (one GPU job at a time).

---

## Startup Behaviour

No model is pre-loaded. Each endpoint loads the required model into GPU, runs inference, then deletes the pipeline and calls `torch.cuda.empty_cache()` in `finally`.

---

## Supported Image Formats

All endpoints that accept image uploads validate the file extension. Accepted formats: `.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`, `.tiff`

Unsupported formats return `422` immediately before any GPU work begins.

---

## Endpoints

### `GET /health`

Returns server status and GPU lock state.

**Success Response** — `200 OK`

```json
{
  "status": "ok",
  "device": "cuda",
  "gpu_busy": false
}
```

| Field | Description |
|---|---|
| `device` | `"cuda"` or `"cpu"` |
| `gpu_busy` | `true` if a model is currently running inference |

---

### `POST /text2img`

Generate one or more images from a text prompt.

**Request** — `application/json`

```json
{
  "prompt": "A sleek sports car on a mountain road at sunset",
  "negative_prompt": "low quality, bad anatomy, blurry, distorted",
  "aspect_ratio": "16:9",
  "num_steps": 50,
  "cfg_scale": 4.0,
  "seed": 42,
  "num_samples": 1
}
```

| Field | Type | Default | Constraints | Description |
|---|---|---|---|---|
| `prompt` | `string` | — | Required | Generation prompt |
| `negative_prompt` | `string` | `"low quality, bad anatomy, blurry, distorted"` | — | Negative prompt |
| `aspect_ratio` | `string` | `"16:9"` | One of the values in the table below | Output image aspect ratio |
| `num_steps` | `integer` | `50` | > 0 | Number of diffusion steps |
| `cfg_scale` | `float` | `4.0` | > 0 | Classifier-free guidance scale |
| `seed` | `integer` | random | — | RNG seed; ignored at runtime — each sample always gets its own independent random seed |
| `num_samples` | `integer` | `1` | `1`–`8` | Number of images to generate in one call |

**Aspect Ratio → Resolution Map**

| Ratio | Width × Height |
|---|---|
| `1:1` | 1328 × 1328 |
| `16:9` | 1664 × 928 |
| `9:16` | 928 × 1664 |
| `4:3` | 1472 × 1104 |
| `3:4` | 1104 × 1472 |
| `3:2` | 1584 × 1056 |
| `2:3` | 1056 × 1584 |

**Success Response** — `200 OK`

```json
{
  "status": "success",
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "urls": [
    "/download/550e8400-e29b-41d4-a716-446655440000/output_0.png",
    "/download/550e8400-e29b-41d4-a716-446655440000/output_1.png"
  ],
  "seeds": [1827364910, 983741200]
}
```

> `urls` and `seeds` are parallel arrays — `seeds[i]` is the seed used to generate `urls[i]`.

**Error Responses**

| Status | Condition | `detail` example |
|---|---|---|
| `422` | Missing required `prompt` | FastAPI validation error |
| `422` | Invalid `aspect_ratio` | `"Invalid aspect_ratio '21:9'. Must be one of: ..."` |
| `422` | `num_samples` out of range | `"num_samples must be between 1 and 8"` |
| `500` | Model load failure (OOM, network, etc.) | `"<exception message>"` |
| `500` | Inference failure | `"<exception message>"` |

---

### `POST /edit`

Edit an image using a text instruction.

**Request** — `multipart/form-data`

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `file` | `UploadFile` | ✅ | — | Input image |
| `prompt` | `string` (Form) | ✅ | — | Editing instruction (e.g. `"Change the car color to red"`) |
| `steps` | `integer` (Form) | ❌ | `40` | Diffusion steps |
| `cfg_scale` | `float` (Form) | ❌ | `4.0` | Guidance scale |
| `seed` | `integer` (Form) | ❌ | `42` | Accepted but ignored — each sample uses its own independent random seed |
| `num_samples` | `integer` (Form) | ❌ | `1` | Number of result images to generate (`1`–`8`) |

**Success Response** — `200 OK`

```json
{
  "status": "success",
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "input_url": "/download/550e8400-e29b-41d4-a716-446655440000/input.png",
  "result_urls": [
    "/download/550e8400-e29b-41d4-a716-446655440000/result_0.png",
    "/download/550e8400-e29b-41d4-a716-446655440000/result_1.png"
  ],
  "seeds": [1827364910, 983741200]
}
```

> `result_urls` and `seeds` are parallel arrays — `seeds[i]` is the seed used to generate `result_urls[i]`.

**Error Responses**

| Status | Condition |
|---|---|
| `422` | Missing required form fields |
| `422` | Unsupported image format |
| `422` | `num_samples` out of range |
| `500` | Model load failure / inference failure |

---

### `POST /edit-multi`

Edit multiple images together using a single prompt. The model receives all images simultaneously and outputs one stitched result image.

**Request** — `multipart/form-data`

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `files` | `List[UploadFile]` | ✅ | — | 1–N input images |
| `prompt` | `string` (Form) | ✅ | — | Editing instruction |
| `steps` | `integer` (Form) | ❌ | `40` | Diffusion steps |
| `cfg_scale` | `float` (Form) | ❌ | `4.0` | Guidance scale |
| `seed` | `integer` (Form) | ❌ | `42` | RNG seed |

**Success Response** — `200 OK`

```json
{
  "status": "success",
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "count": 3,
  "inputs": [
    "/download/550e8400.../input_0.png",
    "/download/550e8400.../input_1.png",
    "/download/550e8400.../input_2.png"
  ],
  "results": [
    "/download/550e8400.../result_stitched.png"
  ]
}
```

> **Note:** `results` always contains exactly **one** URL — the combined output image, not one URL per input.

**Error Responses**

| Status | Condition |
|---|---|
| `422` | Missing required form fields |
| `422` | Any file has an unsupported image format |
| `500` | Model load failure / inference failure |

---

### `POST /angle`

Synthesise a new view of an object from a different camera angle.

**Request** — `multipart/form-data`

| Field | Type | Required | Default | Constraints | Description |
|---|---|---|---|---|---|
| `file` | `UploadFile` | ✅ | — | — | Input image (resized to 1024×1024 internally) |
| `mode` | `string` (Form) | ❌ | `"custom"` | `"custom"` or `"multi"` | `"custom"` = one view; `"multi"` = right/back/left views |
| `azimuth` | `float` (Form) | ❌ | `0` | `[0, 360)` | Horizontal rotation in degrees |
| `elevation` | `float` (Form) | ❌ | `0` | `-30` to `60` | Vertical angle in degrees |
| `distance` | `float` (Form) | ❌ | `1.0` | `0.6`, `1.0`, or `1.8` | Camera distance |

> `azimuth`, `elevation`, and `distance` are only used when `mode="custom"`.

**Angle snapping:** All three parameters are snapped to the nearest supported value before building the prompt.

| Axis | Supported values |
|---|---|
| Azimuth | `0`, `45`, `90`, `135`, `180`, `225`, `270`, `315` |
| Elevation | `-30`, `0`, `30`, `60` |
| Distance | `0.6`, `1.0`, `1.8` |

**LoRAs loaded at runtime:**
- `lightx2v/Qwen-Image-Edit-2511-Lightning` — 4-step lightning adapter
- `fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA` — angle synthesis adapter

Both adapters are set with `adapter_weights=[1.0, 1.0]` and inference runs with `num_inference_steps=4`, `guidance_scale=1.0`.

**Success Response — `mode="custom"`** — `200 OK`

```json
{
  "status": "success",
  "request_id": "550e8400-...",
  "input_url": "/download/550e8400.../input.png",
  "results": {
    "custom": "/download/550e8400.../output_custom.png"
  }
}
```

**Success Response — `mode="multi"`** — `200 OK`

```json
{
  "status": "success",
  "request_id": "550e8400-...",
  "input_url": "/download/550e8400.../input.png",
  "results": {
    "right": "/download/550e8400.../output_right.png",
    "back":  "/download/550e8400.../output_back.png",
    "left":  "/download/550e8400.../output_left.png"
  }
}
```

**Error Responses**

| Status | Condition |
|---|---|
| `422` | Missing required `file` |
| `422` | Unsupported image format |
| `422` | Invalid `mode` (not `"custom"` or `"multi"`) |
| `500` | LoRA load failure / model failure / inference failure |

---

### `GET /download/{request_id}/{file_name}`

Download a generated output file.

| Param | Type | Description |
|---|---|---|
| `request_id` | `string` | UUID4 from any endpoint's response |
| `file_name` | `string` | e.g. `output_0.png`, `result.png`, `result_stitched.png`, `output_custom.png` |

**Success Response** — `200 OK`, Content-Type: inferred by FastAPI from the file extension.

**Error Responses**

| Status | Condition |
|---|---|
| `404` | File not found |

---

## Known Limitations

- GPU lock serialises all requests; concurrent calls queue behind the lock.
- Output files are stored in a temporary directory and deleted on server shutdown.
- The `seed` field in `/text2img` requests is accepted but ignored — each sample always uses its own independently drawn random seed.
