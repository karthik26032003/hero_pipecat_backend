import uvicorn
from routers.server import app  # noqa: F401

if __name__ == "__main__":
    uvicorn.run("routers.server:app", host="0.0.0.0", port=8011, reload=False)
