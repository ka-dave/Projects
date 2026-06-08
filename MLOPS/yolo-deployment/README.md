# YOLOv3 Object Detection API

A FastAPI service for object detection using YOLOv3 via cvlib. Upload an image and get back the image with bounding boxes drawn around detected objects.

## Requirements

- Python 3.12
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

## Setup

```bash
uv sync
```

## Running

```bash
uv run python main.py
```

The server starts at `http://localhost:7000`. Visit `http://localhost:7000/docs` for the interactive API docs.

## Usage

`POST /predict` — Upload an image and choose a model:

| Field   | Type   | Values                        |
|---------|--------|-------------------------------|
| `model` | string | `yolov3-tiny`, `yolov3`       |
| `file`  | file   | `.jpg`, `.jpeg`, or `.png`    |

Returns the image with bounding boxes and labels overlaid.

## Models

The YOLOv3 weights are downloaded automatically by cvlib on first use and are not stored in this repo.

## Dependencies

| Package | Purpose |
|---------|---------|
| fastapi + uvicorn | API server |
| cvlib | YOLOv3 object detection |
| opencv-python | Image processing |
| tensorflow | Backend for cvlib |
| ultralytics | YOLOv8 support (future use) |
