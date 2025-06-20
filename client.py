from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from configs.constants import MCP_SERVER_URL
from MCP.client import MCPClient
import logging
from contextlib import asynccontextmanager
import uvicorn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.client = MCPClient()
    try:
        await app.state.client.connect_to_sse_server(server_url=MCP_SERVER_URL)
    except Exception as e:
        raise RuntimeError(f"Failed to connect to SSE server: {e}")
    yield
    await app.state.client.cleanup()
app = FastAPI(title="MCP Client for ResearchSoup", lifespan=lifespan)

# Request body model
class QueryRequest(BaseModel):
    query: str

@app.post("/query")
async def query_mcp(request: QueryRequest):
    query = request.query.strip()
    if query.lower() in {"quit", "exit"}:
        raise HTTPException(status_code=400, detail="Quit/exit command not allowed via API.")

    try:
        response = await app.state.client.process_query(query)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat_mcp(request: QueryRequest):
    query = request.query.strip()
    if query.lower() in {"quit", "exit"}:
        raise HTTPException(status_code=400, detail="Quit/exit command not allowed via API.")

    try:
        response = await app.state.client.chat(query)
        print(response)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)