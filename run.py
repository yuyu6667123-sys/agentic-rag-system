"""Local development entry point for the Agentic RAG FastAPI service."""

import uvicorn


if __name__ == "__main__":
    uvicorn.run(
        "src.api.server:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )
