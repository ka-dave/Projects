# MLOPS

A collection of production-focused machine learning deployment and MLOps projects.

## Projects

| Project | Description |
|---------|-------------|
| [yolo-deployment](./yolo-deployment/) | FastAPI service for YOLOv3 object detection |

## Structure

Each project lives in its own directory with its own dependencies, virtual environment, and `.gitignore`.

## Conventions

- Python version pinned via `.python-version`
- Dependencies managed with [uv](https://github.com/astral-sh/uv)
- Model weights are not committed — downloaded or fetched at runtime
- Virtual environments (`.venv/`) are local only and not pushed
