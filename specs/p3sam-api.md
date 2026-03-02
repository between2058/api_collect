# P3-SAM 3D Segmentation API

**File:** `api_collect/p3sam_api.py`
**Port:** `5001`
**Model:** P3-SAM (`AutoMask` from `auto_mask.py`)
**Reference:** [Hunyuan3D-Part / P3-SAM](https://github.com/Tencent-Hunyuan/Hunyuan3D-Part)
**Pattern:** Synchronous — each request loads the model fresh, runs inference, then releases GPU memory.

---

## Startup Behaviour

The model is **not** pre-loaded at startup. It is instantiated per-request inside the endpoint handler and released in `finally`. If the `auto_mask` package is unavailable, `AutoMask` is set to `None` and every `/segment` call returns `500`.

---

## Endpoints

### `GET /health`

Returns server status and `AutoMask` availability.

**Success Response** — `200 OK`

```json
{
  "status": "ok",
  "model_available": true
}
```

| Field | Description |
|---|---|
| `model_available` | `false` if `auto_mask` failed to import |

---

### `POST /segment`

Run P3-SAM automatic 3D segmentation on a mesh file. Returns a colour-coded GLB where each detected part has a unique random colour.

**Request** — `multipart/form-data`

| Field | Type | Required | Default | Constraints | Description |
|---|---|---|---|---|---|
| `file` | `UploadFile` | ✅ | — | `.glb`, `.ply`, `.obj` | Input mesh file |
| `point_num` | `integer` (Form) | ❌ | `100000` | `1000`–`500000` | Point cloud sampling count — higher = more accurate but slower |
| `prompt_num` | `integer` (Form) | ❌ | `400` | `10`–`1000` | Number of FPS-sampled segmentation prompts |
| `threshold` | `float` (Form) | ❌ | `0.95` | `0.0`–`1.0` | Confidence threshold passed to `AutoMask` constructor |
| `post_process` | `boolean` (Form) | ❌ | `true` | — | Apply connectivity post-processing to the output mask |
| `clean_mesh` | `boolean` (Form) | ❌ | `true` | — | Merge duplicate vertices and clean degenerate faces before inference |
| `seed` | `integer` (Form) | ❌ | `42` | — | Global random seed — controls point cloud sampling and prompt selection for reproducibility |
| `prompt_bs` | `integer` (Form) | ❌ | `32` | `1`–`400` | Prompt inference batch size — larger = faster but higher VRAM usage |

**Parameter Guide**

| Parameter | Lower value | Higher value |
|---|---|---|
| `point_num` | Faster, less geometric detail | Slower, finer segmentation boundaries |
| `prompt_num` | Fewer candidates sampled | More candidates, better coverage |
| `prompt_bs` | Less VRAM pressure | Faster inference |
| `threshold` | More permissive mask acceptance | More conservative mask acceptance |

> **Note on `threshold`:** This value is stored on the `AutoMask` instance and forwarded to `predict_aabb`. Its effect depends on the installed version of `auto_mask.py`.

**Success Response** — `200 OK`

```json
{
  "segmented_glb": "/download/550e8400-e29b-41d4-a716-446655440000/segmented_output_parts.glb",
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "num_parts": 7
}
```

| Field | Type | Description |
|---|---|---|
| `segmented_glb` | `string` | Relative URL — pass to `GET /download/{request_id}/{file_name}` |
| `request_id` | `string` (UUID4) | Unique ID for this request's output directory |
| `num_parts` | `integer` | Number of distinct parts detected (faces with label `-1` are excluded) |

**Output format:** The returned GLB is a single `trimesh.Trimesh` with `face_colors` applied — one random RGB colour per detected part, black (`[0,0,0]`) for unlabelled faces.

**Error Responses**

| Status | Condition | `detail` example |
|---|---|---|
| `422` | Unsupported mesh format | `"Unsupported file type '.fbx'. Must be one of: ..."` |
| `422` | `point_num` out of range | `"point_num must be between 1000 and 500000"` |
| `422` | `prompt_num` out of range | `"prompt_num must be between 10 and 1000"` |
| `422` | `threshold` out of range | `"threshold must be between 0.0 and 1.0"` |
| `422` | `prompt_bs` out of range | `"prompt_bs must be between 1 and 400"` |
| `500` | `AutoMask` import failed (missing dependency) | `"AutoMask class not available."` |
| `500` | Model weight load failed | `"<traceback message>"` |
| `500` | Trimesh cannot parse uploaded file | `"<trimesh error>"` |
| `500` | Any other unhandled exception | `"<exception message>"` |

> **Note:** A full traceback is printed to stdout on every `500`. Examine server logs for root cause.

---

### `GET /download/{request_id}/{file_name}`

Download a generated output file.

**Path Parameters**

| Param | Type | Description |
|---|---|---|
| `request_id` | `string` | UUID4 from `/segment` response |
| `file_name` | `string` | e.g. `segmented_output_parts.glb` |

**Success Response** — `200 OK`
Content-Type: `application/octet-stream`
Body: raw binary file.

**Error Responses**

| Status | Condition |
|---|---|
| `404` | File not found (wrong `request_id` or `file_name`, or server restarted) |

---

## Implementation Notes

- **Mesh loading:** Uses `trimesh.load(..., process=False)` to match the official demo behaviour and avoid preprocessing that could alter vertex/face structure.
- **Seed handling:** `set_seed(seed)` sets Python `random`, NumPy, and PyTorch global seeds before inference. The same `seed` is also passed directly to `predict_aabb()` for trimesh surface sampling.
- **Output building:** `predict_aabb()` returns `(aabb, face_ids, mesh)`. The API builds the coloured GLB itself by assigning random per-part face colours and exporting via trimesh — no intermediate files are required.
- **GPU lifecycle:** Model is loaded → inference → deleted in `finally` on every request. `gc.collect()` + `torch.cuda.empty_cache()` + `torch.cuda.ipc_collect()` are called after deletion.

## Known Limitations

- Output files are stored in a `tempfile.mkdtemp()` directory. Files are not cleaned up between requests; they are deleted only on server shutdown.
- Model loads and unloads on every request (high VRAM safety, high latency per call).
- No concurrency control — parallel requests will compete for GPU memory.
