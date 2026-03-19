from typing import Dict, List, Tuple

from openai import OpenAI
from openai import (
    RateLimitError,
    APIError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
)

from app.config import (
    MAX_CONTEXT_CHUNKS,
    GENAPI_API_KEY,
    GENAPI_BASE_URL,
    GENAPI_MODEL,
)


class RAGService:
    def __init__(self) -> None:
        self.client = None

        if GENAPI_API_KEY:
            self.client = OpenAI(
                api_key=GENAPI_API_KEY,
                base_url=GENAPI_BASE_URL,
            )

    def build_context(self, chunks: List[Dict]) -> str:
        parts = []

        for i, chunk in enumerate(chunks[:MAX_CONTEXT_CHUNKS], start=1):
            parts.append(
                f"""[Фрагмент {i}]
Книга: {chunk["source"]}
Фрагмент: #{chunk["chunk_id"]}
Позиция: ~{chunk["relative_position_percent"]}% книги
Символы: {chunk["char_start"]}-{chunk["char_end"]}
Текст:
{chunk["text"]}"""
            )

        return "\n\n".join(parts)

    def build_citations(self, chunks: List[Dict]) -> List[Dict]:
        return [
            {
                "text": chunk["text"][:350],
                "source": chunk["source"],
                "chunk_id": chunk["chunk_id"],
                "char_start": chunk["char_start"],
                "char_end": chunk["char_end"],
            }
            for chunk in chunks[:MAX_CONTEXT_CHUNKS]
        ]

    def generate_answer(self, query: str, chunks: List[Dict]) -> Tuple[str, List[Dict]]:
        if not chunks:
            return "В загруженных книгах не найдено информации по вашему вопросу.", []

        citations = self.build_citations(chunks)

        if self.client is None:
            return (
                "GENAPI_API_KEY не задан. Поэтому ИИ-ответ сейчас недоступен.\n\n"
                "Ниже показаны самые релевантные цитаты по вашему вопросу.",
                citations,
            )

        context = self.build_context(chunks)

        system_prompt = (
            "Ты отвечаешь только на основе контекста из книг. "
            "Не выдумывай факты. "
            "Используй только предоставленные фрагменты. "
            "Если точного ответа нет, ответь: "
            "\"В предоставленных текстах нет точного ответа на этот вопрос.\" "
            "Пиши кратко, ясно и на русском языке."
        )

        user_prompt = f"""
Контекст:
{context}

Вопрос:
{query}
""".strip()

        try:
            response = self.client.chat.completions.create(
                model=GENAPI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=700,
            )

            answer = response.choices[0].message.content.strip()

            if not answer:
                answer = (
                    "Не удалось получить текстовый ответ от модели.\n\n"
                    "Ниже показаны самые релевантные цитаты."
                )

            return answer, citations

        except RateLimitError:
            return (
                "Сейчас ИИ-ответ недоступен: превышен лимит запросов.\n\n"
                "Но поиск по книге работает. Ниже показаны самые релевантные цитаты.",
                citations,
            )

        except AuthenticationError:
            return (
                "Сейчас ИИ-ответ недоступен: проблема с API-ключом GenAPI.\n\n"
                "Но поиск по книге работает. Ниже показаны самые релевантные цитаты.",
                citations,
            )

        except BadRequestError as e:
            return (
                f"Сейчас ИИ-ответ недоступен: ошибка запроса к модели ({e}).\n\n"
                "Но поиск по книге работает. Ниже показаны самые релевантные цитаты.",
                citations,
            )

        except APITimeoutError:
            return (
                "Сейчас ИИ-ответ недоступен: сервер модели не ответил вовремя.\n\n"
                "Но поиск по книге работает. Ниже показаны самые релевантные цитаты.",
                citations,
            )

        except APIError as e:
            return (
                f"Сейчас ИИ-ответ временно недоступен: ошибка API ({e}).\n\n"
                "Но поиск по книге работает. Ниже показаны самые релевантные цитаты.",
                citations,
            )

        except Exception as e:
            return (
                f"Сейчас ИИ-ответ временно недоступен: {e}\n\n"
                "Но поиск по книге работает. Ниже показаны самые релевантные цитаты.",
                citations,
            )