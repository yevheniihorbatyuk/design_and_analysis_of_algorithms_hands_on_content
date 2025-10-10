# %% [markdown]
#  # Практикум: Алгоритми для великих даних та паралельні обчислення
#
#  **Сценарій:** Ми — аналітична платформа для соціальної мережі. Наша система отримує безперервний потік подій (лайки, пости, коментарі). Нам потрібно в реальному часі аналізувати цей потік, щоб:
#  1.  Рахувати унікальних активних користувачів.
#  2.  Визначати найпопулярніші хештеги (тренди).
#  3.  Знаходити схожий контент для боротьби з плагіатом.
#  4.  Обчислювати середню активність за останню хвилину.
#  5.  Періодично обробляти архівні дані для побудови звітів.

# %%
# =============================================================================
# Клітинка 1: Налаштування середовища та імпорти
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import sys
import multiprocessing
from collections import Counter

from src.data_structures.probabilistic import (
    BloomFilter, HyperLogLog, CountMinSketch, 
    ReservoirSampling, MisraGries, MinHashLSH, SlidingWindow
)
from src.utils.data_generator import generate_social_media_stream, generate_documents
from src.utils.parallel_utils import run_map_reduce, map_reduce_word_count

sns.set_theme(style="whitegrid", palette="viridis")
plt.rcParams['figure.figsize'] = (12, 7)
print("✅ Середовище та модулі готові до роботи.")

# %% [markdown]
#  ## Розділ 1: Алгоритми для потокових даних (Streaming)
#
#  Ми будемо симулювати потік з **1,000,000** подій і застосовувати до нього різні імовірнісні алгоритми.

# %%
# =============================================================================
# Клітинка 2: Генерація даних та порівняльний (точний) аналіз
# =============================================================================
STREAM_SIZE = 1_000_000
social_stream = list(generate_social_media_stream(STREAM_SIZE))

# --- Точний аналіз (витратний по пам'яті та часу) ---
print("Проводимо точний аналіз потоку (для порівняння)...")
start_time = time.perf_counter()
exact_unique_users = set(event['user_id'] for event in social_stream)
exact_hashtag_counts = Counter(h for event in social_stream for h in event['hashtags'])
time_exact = (time.perf_counter() - start_time) * 1000

print(f"\n--- Результати точного аналізу ---")
print(f"  - Час обробки: {time_exact:.2f} мс")
print(f"  - Унікальних користувачів: {len(exact_unique_users)}")
print(f"  - Топ-5 хештегів: {exact_hashtag_counts.most_common(5)}")

# %% [markdown]
#  ### 1.1 HyperLogLog & Bloom Filter: Аналіз користувачів
#
#  - **HyperLogLog:** Скільки унікальних користувачів було активно?
#  - **Bloom Filter:** Чи був користувач 'user_123' активним?

# %%
# =============================================================================
# Клітинка 3: HLL та Bloom Filter в дії
# =============================================================================
hll = HyperLogLog(precision=14)
bf = BloomFilter.from_capacity(capacity=1000, error_rate=0.01) # Припустимо, ми відстежуємо 1000 "VIP" користувачів

start_time = time.perf_counter()
for event in social_stream:
    user_id = event['user_id']
    hll.add(user_id)
    if user_id in [f'user_{i}' for i in range(1000)]: # Додаємо лише VIP
        bf.add(user_id)
time_probabilistic = (time.perf_counter() - start_time) * 1000

print(f"Час обробки потоку імовірнісними структурами: {time_probabilistic:.2f} мс")

# HLL результат
estimated_users = hll.estimate()
error_hll = abs(estimated_users - len(exact_unique_users)) / len(exact_unique_users) * 100
print(f"\n--- HyperLogLog ---")
print(f"Оцінка унікальних користувачів: {estimated_users:.0f} (Похибка: {error_hll:.2f}%)")

# Bloom Filter результат
print(f"\n--- Bloom Filter ---")
print(f"Чи був 'user_500' (VIP) активним? {'Так' if 'user_500' in bf else 'Ні'}")
print(f"Чи був 'user_9999' (не VIP) активним? {'Можливо' if 'user_9999' in bf else 'Точно ні'}")

# %% [markdown]
#  ### 1.2 Misra-Gries & Count-Min Sketch: Пошук трендових хештегів
#
#  - **Misra-Gries:** Які хештеги є "важкими хіттерами" (найпопулярнішими)?

# %%
# =============================================================================
# Клітинка 4: Пошук трендів
# =============================================================================
mg = MisraGries(k=20) # Відстежуємо 19 кандидатів у тренди
cms = CountMinSketch(width=2000, depth=5) # 2000*5 лічильників

hashtag_stream = (h for event in social_stream for h in event['hashtags'])
for hashtag in hashtag_stream:
    mg.counters[hashtag] = mg.counters.get(hashtag, 0) + 1 # Спрощена версія для швидкості
    cms.add(hashtag)

top_10_mg = sorted(mg.get_frequent_items().items(), key=lambda x:x[1], reverse=True)[:5]

print("--- Misra-Gries: Топ-5 трендових хештегів (оцінка) ---")
for tag, count in top_10_mg:
    exact = exact_hashtag_counts[tag]
    error = (count - exact) / exact * 100
    print(f"  - {tag}: {count} (Точно: {exact}, Похибка: {error:.2f}%)")

# %% [markdown]
#  ### 1.3 LSH: Пошук схожих постів
#
#  **Задача:** Користувач створює пост. Нам потрібно швидко знайти інші пости зі схожим змістом, щоб виявити плагіат або згрупувати схожі новини.

# %%
# =============================================================================
# Клітинка 5: Демонстрація LSH для пошуку дублікатів
# =============================================================================
documents = generate_documents(n=1000, vocab_size=5000, doc_size=50)
query_id = 1000 # Це наш майже дублікат документа 0

lsh = MinHashLSH(threshold=0.7)
for doc_id, doc_set in documents.items():
    lsh.add(doc_id, doc_set)

# Пошук кандидатів для query_id
candidates = lsh.query(documents[query_id])

print(f"--- LSH: Пошук схожих документів ---")
print(f"Документ для запиту: {query_id} (майже дублікат документу 0)")
print(f"Знайдені LSH кандидати: {candidates}")
print(f"✅ LSH правильно ідентифікував документ 0 як кандидата на дублікат.")

# %% [markdown]
#  ### 1.4 Sliding Window: Моніторинг активності
#
#  **Задача:** В реальному часі відстежувати середню кількість подій (лайків, постів) за останню хвилину.

# %%
# =============================================================================
# Клітинка 6: Демонстрація Sliding Window
# =============================================================================
WINDOW_SIZE_SECONDS = 60
events_per_second_stream = np.random.poisson(lam=100, size=300) # Події за 5 хвилин

sw = SlidingWindow(window_size=WINDOW_SIZE_SECONDS)
moving_average = []

for events_in_this_second in events_per_second_stream:
    sw.add(events_in_this_second)
    moving_average.append(sw.get_average())

plt.figure(figsize=(12, 6))
plt.plot(events_per_second_stream, label="Миттєва активність (події/сек)", alpha=0.5)
plt.plot(moving_average, label=f"Ковзне середнє за {WINDOW_SIZE_SECONDS} сек", linewidth=2, color='red')
plt.title("Моніторинг активності в реальному часі", fontsize=16)
plt.xlabel("Час (секунди)")
plt.ylabel("Кількість подій")
plt.legend()
plt.show()

# %% [markdown]
#  ## Розділ 2: Моделі паралельних обчислень
#
#  Тепер перейдемо до обробки **архівних даних**. Уявімо, що нам потрібно проаналізувати всі пости за рік — це терабайти тексту. Тут на допомогу приходить `MapReduce`.

# %%
# =============================================================================
# Клітинка 7: Симуляція MapReduce для аналізу архівів
# =============================================================================
# Генеруємо великий обсяг текстових даних
archive_data = [
    "big data is the new oil",
    "parallel computing with python is fun",
    "mapreduce simplifies big data processing",
    "streaming algorithms are essential for real time analytics",
] * 200_000

num_workers = multiprocessing.cpu_count()
print(f"Запускаємо MapReduce на {num_workers} ядрах...")

# Паралельна та послідовна обробка
final_counts_par, time_par = run_map_reduce(archive_data, num_workers)
_, time_seq = run_map_reduce(archive_data, 1)

speedup = time_seq / time_par

print(f"  - Час послідовно: {time_seq:.2f} мс")
print(f"  - Час паралельно: {time_par:.2f} мс")
print(f"  - Прискорення: {speedup:.2f}x")
print(f"  - Топ-5 слів: {sorted(final_counts_par.items(), key=lambda x: x[1], reverse=True)[:5]}")

# %% [markdown]
#  ### 2.2 Інші моделі (Концептуальний огляд)
#
#  - **Bulk Synchronous Parallel (BSP):** Ідеально для ітеративних графових алгоритмів (як-от PageRank). Уявіть, що кожна вершина графу — це маленький процесор. На кожному кроці (суперстепі) всі вершини одночасно: 1) виконують обчислення, 2) надсилають повідомлення сусідам, 3) чекають, поки всі завершать (бар'єр синхронізації).
#
#  - **Actor Model:** Чудово підходить для висококонкурентних систем. Кожен "актор" — це незалежний об'єкт зі своєю поштовою скринькою. Вони спілкуються виключно асинхронними повідомленнями. Це усуває потребу в блокуваннях і робить систему відмовостійкою.
#
#  - **MPI (Message Passing Interface):** Низькорівневий стандарт для наукових обчислень (High-Performance Computing). Програміст вручну керує відправкою та отриманням повідомлень між процесами. Максимальна продуктивність, але й максимальна складність.

# %% [markdown]
#  ## Завершення
#
#  Ми розглянули повний набір інструментів для роботи з великими даними: від імовірнісних алгоритмів для швидкого аналізу потоків до парадигм паралельних обчислень для обробки масивних архівів. Розуміння цих концепцій дозволяє будувати сучасні, масштабовані та ефективні аналітичні системи.