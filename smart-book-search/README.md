# Smart Book Search

Сервис для поиска фрагментов в книгах `.txt` и ответов на вопросы по их содержанию.

## Возможности

- загрузка книг в формате `.txt`
- поиск релевантных фрагментов по запросу
- ответы на вопросы по загруженным книгам
- вывод цитат, на которых основан ответ
- честный отказ, если ответ не найден
- простой веб-интерфейс для демонстрации

## Стек

- Python
- FastAPI
- Sentence Transformers
- FAISS
- OpenAI API
- HTML + JS

## Установка

```bash
python -m venv .venv
source .venv/bin/activate   # Linux / macOS
# .venv\Scripts\activate    # Windows

pip install -r requirements.txt