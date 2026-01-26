# Dockerfile для Python-сервисов
FROM python:3.11-slim

# Устанавливаем рабочую директорию в контейнере
WORKDIR /app

# Копируем файл с зависимостями
COPY requirements.txt .

# Устанавливаем зависимости
# --no-cache-dir чтобы не хранить кэш и уменьшить размер образа
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь остальной код сервиса в рабочую директорию
COPY . .

# Команда для запуска будет указана в docker-compose.yml
