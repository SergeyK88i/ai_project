-- Включение расширения pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- Таблица для хранения фрагментов документов (чанков)
CREATE TABLE IF NOT EXISTS chunks (
    id VARCHAR(255) PRIMARY KEY,
    doc_name TEXT NOT NULL,
    chunk_sequence_num INTEGER NOT NULL,
    header_1 TEXT,
    header_2 TEXT,
    chunk_text TEXT NOT NULL,
    embedding VECTOR(1024) NOT NULL, -- Размерность эмбеддингов GigaChat
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc_name ON chunks(doc_name);
CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks USING HNSW (embedding vector_l2_ops); -- Для эффективного векторного поиска

-- Таблица для кэширования вопросов и ответов RAG
CREATE TABLE IF NOT EXISTS question_cache (
    id SERIAL PRIMARY KEY,
    question_text TEXT NOT NULL,
    question_vector VECTOR(1024) NOT NULL,
    final_answer TEXT NOT NULL,
    source_chunk_ids TEXT[], -- Массив ID чанков, использованных для ответа
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_question_cache_vector ON question_cache USING HNSW (question_vector vector_l2_ops);
