# P3-SAM 3D Segmentation API

**File:** `api_collect/p3sam_api.py`
**Port:** `5001`
**Model:** P3-SAM (`AutoMask` from `auto_mask.py`)
**Pattern:** Synchronous — each request loads the model fresh, runs inference, then releases GPU memory.

---

## Startup Behaviour

The model is **not** pre-loaded at startup. It is instantiated per-request inside the endpoint handler and released in `finally`. If the `auto_mask` package is unavailable, `AutoMask` is set to `None` and every `/segment` call returns `500`.

---

## Endpoints

### `POST /segment`

Run P3-SAM automatic 3D segmentation on a mesh file.

**Request** — `multipart/form-data`

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | `UploadFile` | ✅ | Input mesh. Supported formats: `.glb`, `.ply`, `.obj` |

**Success Response** — `200 OK`

```json
{
  "segmented_glb": "/download/{request_id}/segmented_output_parts.glb",
  "request_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

| Field | Type | Description |
|---|---|---|
| `segmented_glb` | `string` | Relative URL — pass to `GET /download/{request_id}/{file_name}` |
| `request_id` | `string` (UUID4) | Unique ID for this request's output directory |

**Error Responses**

| Status | Condition | `detail` example |
|---|---|---|
| `500` | `AutoMask` import failed (missing dependency) | `"AutoMask class not available."` |
| `500` | Model weight load failed | `"<traceback message>"` |
| `500` | Trimesh cannot parse uploaded file | `"<trimesh error>"` |
| `500` | Segmentation ran but output GLB not written | `"Model finished but output file was not found."` |
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

## Known Limitations

- Output files are stored in a `tempfile.mkdtemp()` directory. Files are not cleaned up between requests; they are deleted only on server shutdown.
- Model loads and unloads on every request (high VRAM safety, high latency per call).
