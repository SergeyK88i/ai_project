import aiohttp
import uuid
import json
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class GigaChatAPI:
    def __init__(self):
        self.access_token = None
        self.auth_token = None # Сохраняем основной токен для переполучения
        self.conversation_history = []
        
    async def get_token(self, auth_token: str, scope: str = 'GIGACHAT_API_PERS') -> bool:
        """Получение токена доступа и сохранение данных для обновления."""
        self.auth_token = auth_token # Сохраняем для будущих обновлений
        rq_uid = str(uuid.uuid4())
        url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"

        if auth_token and auth_token.lower().startswith('basic '):
            auth_token = auth_token[6:]

        headers = {
            'Content-Type': 'application/x-www-form-urlencoded',
            'Accept': 'application/json',
            'RqUID': rq_uid,
            'Authorization': f'Basic {auth_token}'
        }
        payload = {'scope': scope}

        try:
            logger.info(f"🔑 Попытка получить токен. URL: {url}")
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, data=payload, ssl=False) as response:
                    if response.status == 200:
                        data = await response.json()
                        self.access_token = data['access_token']
                        logger.info("✅ GigaChat токен получен успешно")
                        return True
                    else:
                        logger.error(f"❌ Ошибка получения токена: {response.status}")
                        return False
        except Exception as e:
            logger.error(f"❌ Непредвиденная ошибка при получении токена: {str(e)}")
            return False

    async def get_chat_completion(self, system_message: str, user_message: str, functions: list = None, function_call: dict = None) -> Dict[str, Any]:
        """Получение ответа от GigaChat с авто-обновлением токена и поддержкой Function Calling."""
        for attempt in range(2):
            if not self.access_token:
                logger.warning("Токен доступа отсутствует, попытка обновления...")
                if not self.auth_token or not await self.get_token(self.auth_token):
                    raise Exception("Не удалось обновить токен, основной токен авторизации отсутствует.")
            
            messages = [{"role": "system", "content": system_message}]
            if self.conversation_history:
                messages.extend(self.conversation_history)
            messages.append({"role": "user", "content": user_message})
            
            url = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"
            payload = {"model": "GigaChat", "messages": messages, "temperature": 1, "top_p": 0.1, "n": 1, "stream": False, "max_tokens": 1024, "repetition_penalty": 1}
            
            # Добавляем параметры для Function Calling, если они переданы
            if functions:
                payload['functions'] = functions
            if function_call:
                payload['function_call'] = function_call

            headers = {'Content-Type': 'application/json', 'Accept': 'application/json', 'Authorization': f'Bearer {self.access_token}'}

            logger.info(f"💬 Отправка запроса в GigaChat. System: '{system_message[:80]}...', User: '{user_message[:80]}...'")

            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, headers=headers, json=payload, ssl=False) as response:
                        if response.status == 200:
                            data = await response.json()
                            message = data.get('choices', [{}])[0].get('message', {})

                            # ПРОВЕРКА НА ВЫЗОВ ФУНКЦИИ
                            if message.get("function_call"):
                                logger.info(f"✅ GigaChat запросил вызов функции: {message['function_call']['name']}")
                                return {
                                    "success": True,
                                    "function_call": message["function_call"]
                                }

                            # Стандартная обработка текстового ответа
                            assistant_response = message.get('content', '')
                            if not assistant_response:
                                 logger.error(f"❌ GigaChat вернул успешный ответ, но он пустой или в неожиданном формате. Ответ: {data}")
                                 return {"success": False, "response": "", "error": "Empty or invalid response structure."}

                            self.conversation_history.append({"role": "user", "content": user_message})
                            self.conversation_history.append({"role": "assistant", "content": assistant_response})
                            
                            logger.info(f"✅ Получен текстовый ответ от GigaChat")
                            return {
                                "success": True,
                                "response": assistant_response
                            }

                        elif response.status == 401 and attempt == 0:
                            logger.warning("⚠️ Токен для chat/completions истек. Обновляю и пробую снова...")
                            self.access_token = None # Сбрасываем токен, чтобы инициировать обновление
                            continue # Переходим ко второй попытке
                        else:
                            error_text = await response.text()
                            logger.error(f"❌ Ошибка API GigaChat: {response.status} - {error_text}")
                            return {"success": False, "error": f"API Error: {response.status} - {error_text}", "response": ""}
            except aiohttp.ClientError as e:
                logger.error(f"❌ Ошибка сети при запросе к GigaChat: {str(e)}")
                return { "success": False, "error": str(e), "response": "" }
            except Exception as e:
                logger.error(f"❌ Непредвиденная ошибка при запросе к GigaChat: {str(e)}", exc_info=True)
                return { "success": False, "error": str(e), "response": "" }
            
        return { "success": False, "error": "Не удалось выполнить запрос после обновления токена.", "response": "" }

    def clear_history(self):
        self.conversation_history = []
        logger.info("🗑️ История диалога очищена")

    async def get_embedding(self, text: str, model: str = 'Embeddings') -> list[float] | None:
        """Получение эмбеддинга с авто-обновлением токена."""
        for attempt in range(2):
            if not self.access_token:
                logger.error("Токен доступа для эмбеддингов отсутствует, попытка обновления...")
                if not self.auth_token or not await self.get_token(self.auth_token):
                    raise Exception("Не удалось обновить токен, основной токен авторизации отсутствует.")

            url = "https://gigachat.devices.sberbank.ru/api/v1/embeddings"
            payload = {"model": model, "input": [text]}
            headers = {'Content-Type': 'application/json', 'Accept': 'application/json', 'Authorization': f'Bearer {self.access_token}'}

            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, headers=headers, json=payload, ssl=False) as response:
                        if response.status == 200:
                            data = await response.json()
                            embedding = data.get('data', [{}])[0].get('embedding')
                            if embedding:
                                if attempt > 0: logger.info("✅ Повторный запрос на эмбеддинг успешен с новым токеном.")
                                return embedding
                            return None
                        elif response.status == 401 and attempt == 0:
                            logger.warning("⚠️ Токен для эмбеддингов истек. Обновляю и пробую снова...")
                            self.access_token = None
                            continue
                        else:
                            logger.error(f"❌ Ошибка API GigaChat (Embeddings): {response.status} - {await response.text()}")
                            return None
            except Exception as e:
                logger.error(f"❌ Непредвиденная ошибка при получении эмбеддинга: {str(e)}")
                return None
        
        logger.error("❌ Не удалось получить эмбеддинг после 2 попыток.")
        return None
