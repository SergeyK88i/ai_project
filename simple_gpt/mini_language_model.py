# mini_language_model.py
# Наш проект по созданию простой GPT-модели для генерации текста.

import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass

# --- 1. Данные ---
# Наш "корпус" текста, на котором будет учиться модель.
TRAINING_TEXT = """
У лукоморья дуб зелёный;
Златая цепь на дубе том:
И днём и ночью кот учёный
Всё ходит по цепи кругом;
Идёт направо — песнь заводит,
Налево — сказку говорит.
Там чудеса: там леший бродит,
Русалка на ветвях сидит;
"""


# --- 2. Токенизатор ---
# Паттерн: Инкапсуляция. Мы "прячем" всю логику работы со словарем
# и преобразованиями внутри одного класса.
class CharTokenizer:
    def __init__(self, text):
        # Создаем словарь: сортированный список уникальных символов в тексте
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        # Создаем таблицы для преобразования: символ в индекс (stoi) и индекс в символ (itos)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, string):
        """Преобразует строку в список чисел (индексов)."""
        return [self.stoi[c] for c in string]

    def decode(self, int_list):
        """Преобразует список чисел (индексов) обратно в строку."""
        return ''.join([self.itos[i] for i in int_list])

# --- 3. Датасет ---
# Паттерн: Адаптер/Итератор. Класс Dataset адаптирует наши сырые данные
# к формату, который PyTorch DataLoader ожидает, и позволяет эффективно
# итерироваться по обучающим примерам.
class ShakespeareDataset(torch.utils.data.Dataset):
    def __init__(self, text, tokenizer, block_size):
        self.block_size = block_size
        # Кодируем весь текст один раз
        self.data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        print(f"Полный размер закодированного корпуса: {len(self.data)} токенов.")

    def __len__(self):
        # Количество возможных обучающих последовательностей (x, y)
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        # Получаем входную последовательность (x) и целевую (y)
        # x - это блок текста длины block_size
        # y - это тот же блок текста, сдвинутый на один символ вправо
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y

# --- 4. Модель GPT ---

# Паттерн: Configuration Object. Все "магические числа" и параметры
# архитектуры хранятся в одном месте, что упрощает эксперименты.
@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 50304 # GPT-2 vocab_size, но мы его переопределим
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

# Паттерн: Composition. Собираем сложные слои из более простых.
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # Проекции для Query, Key, Value для всех голов
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # Выходная проекция
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # Параметры
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # Маска, чтобы внимание "не смотрело в будущее".
        # Используем `register_buffer`, чтобы PyTorch не считал это параметром модели.
        self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # Размер батча, длина последовательности, размер эмбеддинга
        # Рассчитываем Q, K, V для всех голов сразу
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        # Переформатируем Q, K, V, чтобы разделить их по "головам"
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        
        # Матричное умножение Q и K для получения "оценок" внимания
        # att = (q @ k.transpose(-2, -1)) * (1.0 / (k.size(-1)**0.5))
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        
        # Более эффективная реализация через Flash Attention из PyTorch 2.0
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C) # Собираем головы вместе
        # Применяем выходную проекцию
        y = self.c_proj(y)
        return y

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        # Residual Connections (x + ...)
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class SimpleGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd), # Позиционные эмбеддинги
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Нельзя подавать последовательность длиннее block_size ({self.config.block_size})"
        
        # Токеновые и позиционные эмбеддинги
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # (T, n_embd)
        tok_emb = self.transformer.wte(idx) # (B, T, n_embd)
        x = tok_emb + pos_emb

        # Прогоняем через блоки трансформера
        for block in self.transformer.h:
            x = block(x)
        
        # Финальная нормализация и "голова"
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)

        # Если есть цели (targets), считаем ошибку (loss)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        """
        Генерирует продолжение для последовательности idx.
        idx - это (B, T) тензор индексов в текущем контексте.
        """
        self.eval() # Переводим модель в режим генерации
        for _ in range(max_new_tokens):
            # Обрезаем idx до последних block_size токенов, если он стал слишком длинным
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # Получаем предсказания
            logits, _ = self(idx_cond)
            # Берем логиты только для последнего шага
            logits = logits[:, -1, :] # (B, C)
            # Применяем софтмакс, чтобы получить вероятности
            probs = F.softmax(logits, dim=-1) # (B, C)
            # Сэмплируем следующий токен из распределения
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # Добавляем сэмплированный индекс к последовательности
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

# В следующих шагах мы добавим сюда цикл Обучения.

# --- 5. Класс Trainer ---
# Паттерн: Фасад/Стратегия. Класс Trainer скрывает сложность цикла обучения за простым интерфейсом.
class Trainer:
    def __init__(self, model, optimizer, train_dataset, device):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.train_dataset = train_dataset
        self.device = device

    def _get_batch(self, batch_size):
        """Получает случайный батч данных из датасета."""
        # Генерируем случайные индексы
        ix = torch.randint(len(self.train_dataset) - self.train_dataset.block_size, (batch_size,))
        # Собираем батч
        x = torch.stack([self.train_dataset.data[i : i + self.train_dataset.block_size] for i in ix])
        y = torch.stack([self.train_dataset.data[i + 1 : i + self.train_dataset.block_size + 1] for i in ix])
        return x.to(self.device), y.to(self.device)

    def train(self, num_iterations, batch_size, eval_interval):
        print(f"\n--- Запуск Обучения ---")
        self.model.train() # Переводим модель в режим обучения

        for iter_num in range(num_iterations):
            # Получаем батч данных
            xb, yb = self._get_batch(batch_size)

            # Прямой проход и расчет ошибки
            logits, loss = self.model(xb, yb)
            
            # Обратный проход и оптимизация
            self.optimizer.zero_grad() # Обнуляем градиенты
            loss.backward()            # Расчет градиентов
            self.optimizer.step()      # Обновление параметров

            # Вывод прогресса
            if iter_num % eval_interval == 0 or iter_num == num_iterations - 1:
                print(f"Итерация {iter_num}/{num_iterations}: Ошибка (Loss) = {loss.item():.4f}")
        
        print("--- Обучение Завершено! ---")


# --- Демонстрация работы ---
# Этот блок выполнится, только если мы запускаем этот файл напрямую.
# Это стандартный паттерн в Python, чтобы файл можно было и запускать, и импортировать.
if __name__ == '__main__':
    print("--- Демонстрация Токенизатора ---")
    
    # 1. Создаем экземпляр токенизатора
    tokenizer = CharTokenizer(TRAINING_TEXT)
    print(f"Размер словаря: {tokenizer.vocab_size}")
    print(f"Символы в словаре: {''.join(tokenizer.chars)}")

    # 2. Пробуем кодировать и декодировать
    test_string = "У лукоморья"
    encoded = tokenizer.encode(test_string)
    decoded = tokenizer.decode(encoded)

    print(f"\nОригинальная строка: '{test_string}'")
    print(f"Закодированная строка: {encoded}")
    print(f"Декодированная строка: '{decoded}'")
    
    assert test_string == decoded # Убедимся, что все работает корректно
    print("\nТокенизатор работает корректно!")

    print("\n--- Демонстрация Датасета ---")
    
    # 1. Создаем экземпляр датасета
    block_size = 8 # Длина одной последовательности
    dataset = ShakespeareDataset(TRAINING_TEXT, tokenizer, block_size)
    print(f"Количество обучающих последовательностей в датасете: {len(dataset)}")

    # 2. Получаем первый пример из датасета
    x, y = dataset[0]
    print(f"\nВходная последовательность (x): {x} -> '{tokenizer.decode(x.tolist())}'")
    print(f"Целевая последовательность (y): {y} -> '{tokenizer.decode(y.tolist())}'")
    
    # Убедимся, что y - это x, сдвинутый на 1
    assert torch.equal(dataset.data[0 : block_size], x)
    assert torch.equal(dataset.data[1 : block_size + 1], y)
    print("\nДатасет работает корректно (x и y сдвинуты на 1 токен)!")

    print("\n--- Демонстрация Модели ---")

    # 1. Создаем конфиг для нашей модели
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=block_size,
        n_layer=4,
        n_head=4,
        n_embd=64  # Размер эмбеддинга должен быть кратен n_head
    )

    # 2. Создаем экземпляр модели
    model = SimpleGPT(config)
    print(f"Создана модель SimpleGPT с {sum(p.numel() for p in model.parameters())/1e3:.2f}K параметрами.")

    # 3. Прогоняем через модель наш первый пример из датасета
    # unsqueeze(0) добавляет "батчевое" измерение (batch dimension)
    logits, loss = model(x.unsqueeze(0), y.unsqueeze(0))
    print(f"\nShape выходных логитов: {logits.shape}")
    print(f"Рассчитанная ошибка (loss): {loss.item()}")
    print("\nМодель успешно обработала входные данные!")

    # --- Запуск Обучения ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nИспользуемое устройство для обучения: {device}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    trainer = Trainer(model, optimizer, dataset, device)
    trainer.train(num_iterations=2000, batch_size=32, eval_interval=200)
    
    print("\nБот готов к генерации текста!")

    # --- Генерация ---
    print("\n--- Генерация Текста ---")
    # Создаем стартовый контекст: тензор (1, 1) с индексом символа новой строки '\n'
    # Модель начнет генерировать с новой строки.
    start_context = torch.tensor([[tokenizer.stoi['\n']]], dtype=torch.long, device=device)
    
    # Генерируем текст
    generated_indices = model.generate(idx=start_context, max_new_tokens=300)
    
    # Декодируем и печатаем результат
    generated_text = tokenizer.decode(generated_indices[0].tolist())
    print("Сгенерированный текст:")
    print(generated_text)



