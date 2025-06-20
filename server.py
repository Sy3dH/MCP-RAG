from contextlib import asynccontextmanager
import os
import uuid
import logging
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi_mcp import FastApiMCP
from typing import List, Optional
from parsers.pdf_parser import parse_pdf_chunks
from utils.input_handler import prepare_documents
from embeddings.vectorizer import FastEmbedder
from ingestion.vector_store_wo_pool import VectorStore
from search.store_search import vector_search_with_filter
from fastapi.responses import JSONResponse
from models.models import Document
from qdrant_client.http.exceptions import UnexpectedResponse
import io

UPLOAD_DIR = r"D:\9D Tech Work\Central_repo\Demo\uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting FastAPI app lifespan...")
    embedder = FastEmbedder()
    app.state.embedder = embedder
    yield
    logger.info("Shutting down FastAPI app lifespan...")

app = FastAPI(title="MCP Server for ResearchSoup", lifespan=lifespan)

@app.post("/ingest_documents")
async def ingest_documents(
    researcher: str = Form(...),
    findings: str = Form(...),
    collection_name: str = Form("AI_store"),
    embedding_model: Optional[str] = Form(None),
    pdfs: List[UploadFile] = File(...)
):
    try:
        embedder = app.state.embedder
        vector_size = embedder.vector_size

        if embedding_model:
            logger.info(f"Using custom embedding model: {embedding_model}")
            embedder = FastEmbedder(embedding_model)
            vector_size = embedder.vector_size

        file_buffers = []
        doc_sources = []
        future_file_paths = []
        all_docs = []

        for idx, pdf in enumerate(pdfs):
            pdf_bytes = await pdf.read()
            file_buffers.append((pdf.filename, pdf_bytes))

            filename = f"{uuid.uuid4()}_{pdf.filename}"
            file_path = os.path.join(UPLOAD_DIR, filename)
            future_file_paths.append(file_path)

            chunks = parse_pdf_chunks(io.BytesIO(pdf_bytes))
            logger.info(f"Extracted {len(chunks)} chunks from {pdf.filename}")

            for chunk in chunks:
                doc = Document(text=chunk, metadata={"file_path": file_path})
                doc_sources.append(idx)
                all_docs.append(doc)

        docs = prepare_documents(researcher, findings, all_docs)
        vectors = embedder.embed([doc.text for doc in all_docs])


        store = VectorStore(collection_name=collection_name, vector_size=vector_size)
        store.init_collection()
        inserted_docs, skipped_docs = store.ingest(docs, vectors)


        used_pdf_indices = set(doc_sources[docs.index(doc)] for doc in inserted_docs)
        saved_paths = []

        for i in used_pdf_indices:
            original_name, content = file_buffers[i]
            file_path = future_file_paths[i]
            with open(file_path, "wb") as f:
                f.write(content)
            logger.info(f"Saved uploaded file: {file_path}")
            saved_paths.append(file_path)

        return {
            "message": f"Ingested {len(inserted_docs)} chunks into collection '{collection_name}' successfully.",
            "docs_ingested": len(inserted_docs),
            "docs_skipped": len(skipped_docs),
            "inserted_documents": inserted_docs,
            "skipped_documents": skipped_docs,
            "saved_files": saved_paths
        }

    except Exception as e:
        logger.exception("Failed to ingest documents")
        return {
            "error": str(e),
            "message": "Document ingestion failed. See server logs for more details."
        }

@app.get("/download/")
def download_file(file_path: str):
    """
    Download a PDF file by providing its full path on the server.
    """
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    if not file_path.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    return FileResponse(
        path=file_path,
        media_type='application/pdf',
        filename=os.path.basename(file_path)
    )

@app.post("/retrieve_documents", operation_id="retrieve_documents")
async def retrieve_documents(
    query_text: str,
    collection_name: str = "AI_store",
    limit: int = 5
):
    try:
        embedder = app.state.embedder
        query_vector = embedder.embed([query_text])[0]
        results = vector_search_with_filter(
            query_vector=query_vector,
            collection_name=collection_name,
            limit=limit
        )

        return {"results": [res for res in results]}

    except UnexpectedResponse as e:
        return JSONResponse(status_code=500, content={"error": f"Qdrant query failed: {str(e)}"})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Unexpected error: {str(e)}"})

if __name__ == "__main__":
    mcp = FastApiMCP(app,include_operations=["retrieve_documents"])
    mcp.mount()
    uvicorn.run(app, host="0.0.0.0", port=8001)