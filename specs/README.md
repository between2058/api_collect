# Phidias API Microservices — Specs Index

All services run independently. Each exposes a FastAPI HTTP server with CORS enabled (`*`).

| Service | File | Port | Model | Pattern |
|---|---|---|---|---|
| P3-SAM 3D Segmentation | `p3sam_api.py` | 5001 | P3-SAM (AutoMask) | Synchronous |
| Qwen Image | `qwen_image_api.py` | 8190 | Qwen-Image-2512 / Qwen-Image-Edit-2511 | Synchronous |
| TRELLIS.2 3D Generation | `trellis_api.py` | 8100 | microsoft/TRELLIS.2-4B | **Async Job** |
| Depth-Anything-3 | `depth_api.py` | 8200 | depth-anything/da3-large | **Async Job** |
| ReconViaGen (VGGT) | `reconviagen_api_v4.py` | 52069 | esther11/trellis-vggt-v0-2 | Synchronous |
| SAM3 2D Segmentation | `sam3_api.py` | 8002 | SAM3 (session-based) | Synchronous + Session |
| SAM-3D Image-to-3D | `sam3d_api_v2.py` | 8001 | SAM3D (checkpoints/hf) | Synchronous |

## Error Response Convention

All services use FastAPI's standard error envelope:

```json
{
  "detail": "Human-readable error message"
}
```

## Async Job Pattern (TRELLIS.2 & Depth-Anything-3)

These services return a `job_id` immediately and run inference in a background thread.
Poll `GET /jobs/{job_id}` until `status` is `"completed"` or `"failed"`.

```
POST /generate/... → { "job_id": "...", "status": "pending" }
    ↓
GET /jobs/{job_id}  → { "status": "running", "percent": 40, ... }
    ↓
GET /jobs/{job_id}  → { "status": "completed", "result": { ... } }
    ↓
GET /download/{job_id}/output.glb
```

## Spec Files

- [p3sam-api.md](./p3sam-api.md)
- [qwen-image-api.md](./qwen-image-api.md)
- [trellis-api.md](./trellis-api.md)
- [depth-api.md](./depth-api.md)
- [reconviagen-api.md](./reconviagen-api.md)
- [sam3-api.md](./sam3-api.md)
- [sam3d-api.md](./sam3d-api.md)
