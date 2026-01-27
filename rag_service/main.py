import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import logging
import os
import re
import json
from typing import List, Dict, Any
import sys
from contextlib import asynccontextmanager

from app.giga_chat import GigaChatAPI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Конфигурация ---
GIGACHAT_AUTH_TOKEN = os.getenv("GIGACHAT_AUTH_TOKEN")
if not GIGACHAT_AUTH_TOKEN:
    logger.critical("❌ КРИТИЧЕСКАЯ ОШИБКА: Переменная окружения GIGACHAT_AUTH_TOKEN не установлена!")
    sys.exit(1)
MAX_DEPTH = 2
TOP_LEVEL_CHUNK_TARGET_SIZE = 20000
LOWER_LEVEL_CHUNK_TARGET_SIZE = 5000
CACHE_HIT_THRESHOLD = 0.99
CACHE_SHORTCUT_THRESHOLD = 0.92

# Файлы с данными
KNOWLEDGE_BASE_FILE = os.path.join(os.path.dirname(__file__), "knowledge_base.json")
CHUNKS_DATABASE_FILE = os.path.join(os.path.dirname(__file__), "chunks_database.json")

import asyncpg
from pgvector.asyncpg import register_vector

# --- Глобальные переменные ---
db_pool = None
gigachat_client = GigaChatAPI()

async def apply_rag_db_schema():
    """Читает и применяет схему из db_schema.sql для RAG сервиса."""
    try:
        sql_file_path = os.path.join(os.path.dirname(__file__), 'db_schema.sql')
        with open(sql_file_path, 'r', encoding='utf-8') as f:
            init_sql = f.read()

        async with db_pool.acquire() as conn:
            await conn.execute(init_sql)
        logger.info("✅ Схема базы данных RAG успешно применена/проверена.")
    except Exception as e:
        logger.error(f"❌ Не удалось применить схему базы данных RAG из db_schema.sql: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 RAG-сервер (PostgreSQL) запускается...")
    global db_pool
    
    # 1. Подключение к базе данных
    try:
        DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/dbname")
        db_pool = await asyncpg.create_pool(DATABASE_URL, init=register_vector)
        # Проверка соединения
        async with db_pool.acquire() as connection:
            await connection.fetchval("SELECT 1")
        logger.info("✅ Соединение с PostgreSQL (pgvector) установлено.")
    except Exception as e:
        logger.error(f"❌ КРИТИЧЕСКАЯ ОШИБКА: Не удалось подключиться к PostgreSQL: {e}")
        db_pool = None # Убедимся, что пул не используется, если он невалиден
    
    # 2. Применение схемы RAG
    await apply_rag_db_schema()

    # 3. Проверка токена GigaChat
    if not await gigachat_client.get_token(GIGACHAT_AUTH_TOKEN):
        logger.error("КРИТИЧЕСКАЯ ОШИБКА: Не удалось получить токен GigaChat при запуске.")
    
    logger.info("✨ Startup complete.")
    yield
    
    # 4. Закрытие пула соединений при остановке
    if db_pool:
        await db_pool.close()
        logger.info("🛑 Соединение с PostgreSQL закрыто.")


app = FastAPI(title="Pre-indexed RAG MCP Server", version="2.0.0", lifespan=lifespan)


TOOLS_LIST = [
    {
        "name": "answer_question",
        "description": "Принимает вопрос пользователя, находит релевантную информацию в документации и генерирует осмысленный ответ.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Вопрос пользователя"}
            },
            "required": ["query"]
        }
    }
]

# --- Вспомогательные функции ---
async def synthesize_answer(question: str, context: str) -> str:
    """Генерирует финальный ответ на основе контекста."""
    logger.info(f"Синтез ответа на основе {len(context)} символов контекста.")
    system_message = "Ты — полезный ассистент-консультант. Ответь на вопрос пользователя, основываясь ИСКЛЮЧИТЕЛЬНО на предоставленном ниже контексте из документации. Не придумывай ничего от себя. Если ответ нельзя найти в контексте, так и скажи."
    user_message = f"КОНТЕКСТ:\n{context}\n\nВОПРОС: {question}"
    final_response = await gigachat_client.get_chat_completion(system_message, user_message)
    return final_response.get('response', "Не удалось сгенерировать ответ.")

async def find_relevant_chunks(query_vector: List[float], limit: int = 25) -> List[Dict[str, Any]]:
    """Этап 1: Быстрый поиск. Находит N самых похожих чанков в PostgreSQL."""
    if not db_pool:
        raise Exception("Пул соединений с базой данных не инициализирован.")

    logger.info(f"🔍 Выполняю векторный поиск {limit} ближайших чанков в PostgreSQL...")
    async with db_pool.acquire() as connection:
        # Используем оператор <-> из pgvector для поиска по косинусному расстоянию
        records = await connection.fetch(
            """SELECT id, chunk_text, doc_name FROM chunks ORDER BY embedding <-> $1 LIMIT $2""",
            query_vector, limit
        )
    logger.info(f"✅ Найдено {len(records)} чанков-кандидатов.")
    return [dict(record) for record in records]

# Схема для Function Calling, описывающая нужный нам формат вывода
RERANK_FUNCTION_SCHEMA = {
    "name": "save_rerank_result",
    "description": "Сохраняет результат переранжирования чанков, выбранных для ответа на вопрос.",
    "parameters": {
        "type": "object",
        "properties": {
            "best_chunk_ids": {
                "type": "array",
                "description": "Список ID наиболее релевантных чанков.",
                "items": { "type": "string" }
            },
            "reasoning": {
                "type": "string",
                "description": "Краткое объяснение, почему были выбраны именно эти чанки."
            }
        },
        "required": ["best_chunk_ids", "reasoning"]
    }
}

async def rerank_chunks(question: str, chunks: List[Dict[str, Any]], limit: int = 5) -> List[Dict[str, Any]]:
    """Этап 2: Умная фильтрация с использованием Function Calling."""
    logger.info(f"🧠 Выполняю re-ranking для {len(chunks)} чанков с помощью Function Calling...")
    
    # 1. Простой промпт, описывающий только задачу
    system_message = f"Проанализируй фрагменты текста и выбери не более {limit} самых релевантных для ответа на вопрос пользователя."

    context_for_reranking = ""
    for chunk in chunks:
        context_for_reranking += f"--- ЧАНК ID: {chunk['id']} ---\n{chunk['chunk_text']}\n\n"

    user_message = f"ВОПРОС ПОЛЬЗОВАТЕЛЯ: {question}\n\nСПИСОК ФРАГМЕНТОВ:\n{context_for_reranking}"

    # 2. Вызов API с передачей схемы функции
    response = await gigachat_client.get_chat_completion(
        system_message,
        user_message,
        functions=[RERANK_FUNCTION_SCHEMA],
        function_call={"name": "save_rerank_result"} # Принудительно вызываем нашу функцию
    )

    # 3. Надежное получение результата
    if response.get("function_call"):
        try:
            # Аргументы уже являются словарем (dict), а не строкой
            data = response["function_call"]["arguments"]
            
            reasoning = data.get('reasoning', 'No reasoning provided.')
            best_ids = data.get('best_chunk_ids', [])

            logger.info(f"Рассуждение модели (re-ranker): {reasoning}")
            logger.info(f"✅ LLM выбрал лучшие чанки: {best_ids}")

            if not best_ids:
                logger.warning("LLM re-ranker returned an empty list of chunk IDs.")

            selected_chunks = [chunk for chunk in chunks if chunk['id'] in best_ids]
            return selected_chunks[:limit]

        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Function Calling error: Failed to parse arguments. Error: {e}. Returning top-{limit} initial candidates.")
            return chunks[:limit]
    else:
        logger.error(f"Function Calling error: Model did not call the requested function. Response: {response}")
        return chunks[:limit]

@app.post("/")
async def json_rpc_handler(request: Request):
    body = await request.json()
    if "jsonrpc" not in body or body["jsonrpc"] != "2.0" or "method" not in body or "id" not in body:
        return JSONResponse(status_code=400, content={"jsonrpc": "2.0", "id": body.get("id"), "error": {"code": -32600, "message": "Invalid Request"}})
    request_id = body["id"]
    method = body["method"]
    params = body.get("params", {})
    try:
        if method == "tools/list": result = {"tools": TOOLS_LIST}
        elif method == "tools/call": result = await handle_tools_call(params)
        else: raise HTTPException(status_code=404, detail=f"Method '{body.get('method')}' not found")
        return JSONResponse(content={"jsonrpc": "2.0", "id": request_id, "result": result})
    except Exception as e:
        logger.error(f"❌ Ошибка при выполнении метода '{method}': {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"jsonrpc": "2.0", "id": request_id, "error": {"code": -32603, "message": "Internal Error", "data": str(e)}})

async def execute_db_shortcut_rag(question: str, source_chunk_ids: List[str]) -> str:
    """Выполняет RAG с 'расширением контекста' до полных глав с защитой от переполнения."""
    logger.info(f"SHORTCUT: Запуск RAG с расширением контекста для {len(source_chunk_ids)} исходных чанков.")
    CONTEXT_MAX_SIZE = 12000  # Безопасный лимит символов для контекста

    if not db_pool:
        raise Exception("Пул соединений с базой данных не инициализирован.")

    async with db_pool.acquire() as connection:
        # Шаг 1: Находим родительские главы для исходных чанков
        chapter_records = await connection.fetch(
            """SELECT DISTINCT header_1 FROM chunks WHERE id = ANY($1::TEXT[]) AND header_1 IS NOT NULL""",
            source_chunk_ids
        )
        parent_chapters = [record['header_1'] for record in chapter_records]

        context_texts = []
        if parent_chapters:
            logger.info(f"Найдены родительские главы: {parent_chapters}. Расширяем контекст...")
            # Шаг 2: Загружаем ВСЕ чанки из этих глав для полного контекста
            full_chapter_records = await connection.fetch(
                """SELECT chunk_text FROM chunks WHERE header_1 = ANY($1::TEXT[]) ORDER BY doc_name, chunk_sequence_num""",
                parent_chapters
            )
            
            total_context_len = 0
            for record in full_chapter_records:
                record_text = record['chunk_text']
                if total_context_len + len(record_text) > CONTEXT_MAX_SIZE:
                    logger.warning(f"Контекст главы превысил лимит ({CONTEXT_MAX_SIZE} симв.), обрезаем...")
                    break
                context_texts.append(record_text)
                total_context_len += len(record_text)

        else:
            # Fallback: если главы не найдены, используем старую логику (только исходные чанки)
            logger.warning("Родительские главы для кэшированных чанков не найдены. Используем только исходные чанки.")
            fallback_records = await connection.fetch(
                """SELECT chunk_text FROM chunks WHERE id = ANY($1::TEXT[])""",
                source_chunk_ids
            )
            context_texts = [record['chunk_text'] for record in fallback_records]

    if not context_texts:
        logger.error("Не удалось восстановить контекст из кэшированных ID. Запускаем полный RAG.")
        # Для execute_full_rag нужен query_vector, получим его снова
        query_vector = await gigachat_client.get_embedding(question)
        rag_result = await execute_full_rag(question, query_vector)
        return rag_result['answer']

    context = "\n\n---\n\n".join(context_texts)
    return await synthesize_answer(question, context)

async def execute_full_rag(question: str, query_vector: List[float]) -> Dict[str, Any]:
    """Выполняет полный гибридный RAG-пайплайн: Поиск -> Ранжирование -> Синтез."""
    # Шаг 1: Быстрый поиск (Retrieval)
    candidate_chunks = await find_relevant_chunks(query_vector, limit=25)
    if not candidate_chunks:
        return {"answer": "К сожалению, я не смог найти релевантную информацию в базе знаний.", "source_chunk_ids": []}

    # Шаг 2: Умная фильтрация (Re-ranking)
    final_chunks = await rerank_chunks(question, candidate_chunks, limit=5)

    # Шаг 3: Синтез ответа
    context = "\n\n---\n\n".join([c['chunk_text'] for c in final_chunks])
    final_answer = await synthesize_answer(question, context)
    final_chunk_ids = [c['id'] for c in final_chunks]
    
    return {"answer": final_answer, "source_chunk_ids": final_chunk_ids}


async def handle_tools_call(params: dict):
    tool_name = params.get("name")
    arguments = params.get("arguments", {})
    if tool_name == "answer_question":
        query = arguments.get("query")
        if not query: raise ValueError("Для 'answer_question' требуется аргумент 'query'")
        if not gigachat_client.access_token or not db_pool:
             raise Exception("Сервер не готов к работе: токен GigaChat или подключение к БД отсутствуют.")

        # Шаг 1: Получаем вектор для вопроса
        query_vector = await gigachat_client.get_embedding(query)
        if not query_vector: raise Exception("Не удалось получить эмбеддинг для запроса.")

        # Шаг 2: Ищем в кэше вопросов
        async with db_pool.acquire() as connection:
            # Ищем самый похожий вопрос и сразу считаем схожесть
            cached_record = await connection.fetchrow(
                """SELECT final_answer, source_chunk_ids, (1 - (question_vector <=> $1)) AS similarity 
                   FROM question_cache ORDER BY similarity DESC LIMIT 1""",
                query_vector
            )

        # Шаг 3: Принимаем решение на основе кэша
        if cached_record and cached_record['similarity'] >= CACHE_SHORTCUT_THRESHOLD:
            if cached_record['similarity'] >= CACHE_HIT_THRESHOLD:
                logger.info(f"CACHE HIT: Сходство ({cached_record['similarity']:.2f}) очень высокое. Отдаем готовый ответ.")
                final_answer = cached_record['final_answer']
            else:
                logger.info(f"SHORTCUT: Сходство ({cached_record['similarity']:.2f}) среднее. Используем готовые чанки из БД.")
                final_answer = await execute_db_shortcut_rag(query, cached_record['source_chunk_ids'])
        else:
            logger.info("CACHE MISS: Похожих вопросов в кэше не найдено, запускаем полный RAG-цикл.")
            rag_result = await execute_full_rag(query, query_vector)
            final_answer = rag_result['answer']

            # Сохраняем новый результат в кэш, если он не является "отказом"
            not_found_message = "К сожалению, я не смог найти релевантную информацию"
            if not_found_message not in final_answer:
                async with db_pool.acquire() as connection:
                    await connection.execute(
                        """INSERT INTO question_cache (question_text, question_vector, final_answer, source_chunk_ids)
                           VALUES ($1, $2, $3, $4)""",
                        query, query_vector, final_answer, rag_result['source_chunk_ids']
                    )
                logger.info("✅ Новый успешный ответ добавлен в кэш вопросов-ответов.")

        return {"content": [{"type": "text", "text": json.dumps(final_answer)}], "isError": False}
    else:
        raise ValueError(f"Неизвестное имя инструмента: {tool_name}")

if __name__ == "__main__":
    print("🚀 Запуск Pre-indexed RAG MCP Server на http://localhost:8002")
    uvicorn.run(app, host="0.0.0.0", port=8002)
