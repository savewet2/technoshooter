from typing import List, Optional
from pydantic import BaseModel


class ChunkResult(BaseModel):
    text: str
    source: str
    chunk_id: int
    char_start: int
    char_end: int
    relative_position_percent: int
    score: float


class SearchResponse(BaseModel):
    results: List[ChunkResult]
    message: Optional[str] = None


class Citation(BaseModel):
    text: str
    source: str
    chunk_id: int
    char_start: int
    char_end: int


class AskResponse(BaseModel):
    answer: str
    citations: List[Citation]
    message: Optional[str] = None