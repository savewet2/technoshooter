from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.config import (
    ALLOWED_EXTENSIONS,
    BOOKS_DIR,
    SEARCH_SCORE_THRESHOLD,
    TEMPLATES_DIR,
    TOP_K,
)
from app.loader import load_book
from app.rag import RAGService
from app.schemas import AskResponse, SearchResponse
from app.search_engine import SearchEngine

app = FastAPI(title="Smart Book Search", version="1.2.0")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

search_engine = SearchEngine()
rag_service = RAGService()


def validate_txt_file(filename: str) -> None:
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Можно загружать только .txt файлы")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    ready = not search_engine.is_empty()
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "books": search_engine.list_books(),
            "ready": ready,
        },
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "books_loaded": len(search_engine.list_books()),
        "chunks_loaded": len(search_engine.chunks),
        "openai_enabled": bool(rag_service.client),
        "ready": not search_engine.is_empty(),
    }


@app.get("/status")
def status():
    ready = not search_engine.is_empty()
    return {
        "ready": ready,
        "books_loaded": len(search_engine.list_books()),
        "chunks_loaded": len(search_engine.chunks),
        "openai_enabled": bool(rag_service.client),
        "message": (
            "Система готова к поиску и вопросам"
            if ready
            else "Сначала загрузите и обработайте хотя бы одну книгу"
        ),
    }


@app.get("/books")
def books():
    return {
        "books": search_engine.list_books(),
        "count": len(search_engine.list_books()),
    }


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Файл без имени")

    validate_txt_file(file.filename)

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Файл пустой")

    save_path = BOOKS_DIR / file.filename
    with open(save_path, "wb") as f:
        f.write(contents)

    chunks = load_book(save_path)
    if not chunks:
        raise HTTPException(status_code=400, detail="Не удалось извлечь текст из файла")

    search_engine.add_chunks(chunks)

    return {
        "message": "Книга успешно загружена и проиндексирована",
        "book": file.filename,
        "chunks_added": len(chunks),
        "total_books": len(search_engine.list_books()),
        "total_chunks": len(search_engine.chunks),
        "ready": not search_engine.is_empty(),
    }


@app.get("/search", response_model=SearchResponse)
def search(
    query: str = Query(..., min_length=2),
    top_k: int = Query(TOP_K, ge=1, le=10),
):
    if search_engine.is_empty():
        return SearchResponse(results=[], message="Сначала загрузите хотя бы одну книгу")

    found = search_engine.search(query, top_k=top_k)
    filtered = [item for item in found if item["score"] <= SEARCH_SCORE_THRESHOLD]
    final_results = filtered if filtered else found

    if not final_results:
        return SearchResponse(results=[], message="Подходящие фрагменты не найдены")

    return SearchResponse(results=final_results[:top_k], message=None)


@app.get("/ask", response_model=AskResponse)
def ask(
    query: str = Query(..., min_length=2),
    top_k: int = Query(TOP_K, ge=1, le=10),
):
    if search_engine.is_empty():
        return AskResponse(
            answer="Сначала загрузите хотя бы одну книгу.",
            citations=[],
            message="База книг пуста",
        )

    found = search_engine.search(query, top_k=top_k)
    filtered = [item for item in found if item["score"] <= SEARCH_SCORE_THRESHOLD]
    context_chunks = filtered if filtered else found

    if not context_chunks:
        return AskResponse(
            answer="В загруженных книгах нет ответа на этот вопрос.",
            citations=[],
            message="Релевантные фрагменты не найдены",
        )

    answer, citations = rag_service.generate_answer(query, context_chunks)

    return AskResponse(
        answer=answer,
        citations=citations,
        message=None,
    )