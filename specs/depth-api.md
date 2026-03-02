# Depth-Anything-3 Reconstruction API

**File:** `api_collect/depth_api.py`
**Port:** `8200`
**Model:** `depth-anything/da3-large` (configurable via `DA3_MODEL` env var)
**Pattern:** **Async Job** — inference runs in `ThreadPoolExecutor(max_workers=1)`.

---

## Startup Behaviour

The DA3 model is loaded at startup via `lifespan`. The server only becomes ready after the model is loaded on CUDA. If loading fails, the server does not start.

```
DA3_MODEL = os.getenv('DA3_MODEL', 'depth-anything/da3-large')
DepthAnything3.from_pretrained(DA3_MODEL).cuda().eval()
```

---

## Async Job Lifecycle

```
POST /reconstruct/...  →  200 { "job_id": "...", "status": "pending" }
                                ↓ poll
GET /jobs/{job_id}     →  200 { "status": "running", "percent": 50, ... }
GET /jobs/{job_id}     →  200 { "status": "completed", "result": { ... } }
GET /jobs/{job_id}     →  200 { "status": "failed", "error": "..." }
                                ↓ on completed
GET /download/{job_id}/gs_ply/0000.ply
```

> **Jobs expire after 60 minutes.** Cleanup loop runs every 30 minutes.

---

## Endpoints

### `POST /reconstruct/video`

Extract frames from a video file and run DA3 depth + Gaussian splatting reconstruction.

**Request** — `multipart/form-data`

| Field | Type | Required | Default | Constraints | Description |
|---|---|---|---|---|---|
| `file` | `UploadFile` | ✅ | — | Video file | Input video. Supported: any format OpenCV `VideoCapture` can read (`.mp4`, `.avi`, `.mov`, etc.) |
| `fps` | `float` (Form) | ❌ | `1.0` | Must be > 0 | Frames to extract per second of video. Capped at `MAX_FRAMES = 60` total frames. |

**Frame extraction logic:**
- `frame_interval = max(1, round(video_fps / fps))`
- At most `MAX_FRAMES = 60` frames are extracted, regardless of video length.
- If no frames can be extracted, the job is marked `"failed"`.

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
| `422` | Missing `file` |
| `500` | File write failure |

---

### `POST /reconstruct/images`

Run DA3 depth + Gaussian splatting reconstruction from a set of images (e.g. pre-extracted frames).

**Request** — `multipart/form-data`

| Field | Type | Required | Constraints | Description |
|---|---|---|---|---|
| `files` | `List[UploadFile]` | ✅ | 1–60 images | Input images. Files beyond index 59 are silently discarded. |

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
| `500` | File write failure |

---

### `GET /jobs/{job_id}`

Poll the status of an async reconstruction job.

**Response** — `200 OK`

```json
{
  "status": "pending | running | completed | failed",
  "percent": 90,
  "stage": "Exporting Gaussian splat (gs_ply)...",
  "result": null,
  "error": null
}
```

**`result` shape on `"completed"`:**

```json
{
  "gs_ply_url": "/download/{job_id}/gs_ply/0000.ply",
  "npz_url":    "/download/{job_id}/exports/npz/result.npz",
  "video_url":  "/download/{job_id}/gs_video/0000_extend.mp4",
  "frame_count": 12
}
```

| Field | Type | Description |
|---|---|---|
| `gs_ply_url` | `string` | Gaussian splat PLY file |
| `npz_url` | `string` | Depth prediction in NumPy `.npz` format |
| `video_url` | `string` | Rendered splat preview video (`.mp4`) |
| `frame_count` | `integer` | Number of frames that were processed |

**`error` on `"failed":`**

```json
{
  "status": "failed",
  "percent": 5,
  "stage": "Extracting frames...",
  "result": null,
  "error": "No frames extracted from video"
}
```

**Error Responses**

| Status | Condition |
|---|---|
| `404` | Unknown `job_id` (or job expired) |

---

### `GET /download/{job_id}/{file_path}`

Download any output file from a completed job. The `file_path` parameter is a path (supports slashes).

**Path Parameters**

| Param | Type | Description |
|---|---|---|
| `job_id` | `string` | UUID4 |
| `file_path` | `string` (path) | Relative path inside job directory, e.g. `gs_ply/0000.ply` |

**Success Response** — `200 OK`
Content-Type: `application/octet-stream`

**Error Responses**

| Status | Condition |
|---|---|
| `404` | File not found |

---

## Output Files Reference

All files are written under `{OUTPUT_DIR}/{job_id}/` by the DA3 inference library:

| URL path | Format | Description |
|---|---|---|
| `gs_ply/0000.ply` | PLY | 3D Gaussian splat point cloud |
| `exports/npz/result.npz` | NumPy | Raw depth prediction arrays |
| `gs_video/0000_extend.mp4` | MP4 | Rendered Gaussian splat preview |

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `DA3_MODEL` | `depth-anything/da3-large` | HuggingFace model ID for DA3 |

---

## Known Limitations

- One inference job at a time (single `ThreadPoolExecutor` worker).
- `/reconstruct/images`: when more than `MAX_FRAMES` files are submitted, the response includes a `warning` field and `frames_queued` reflects the actual count processed.
