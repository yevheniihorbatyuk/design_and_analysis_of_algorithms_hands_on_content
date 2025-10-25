# 🚀 Quick Start Guide

## Швидкий старт за 5 хвилин

### 1️⃣ Розпакувати та встановити

```bash
# Перейти в директорію проекту
cd blended5

# Створити віртуальне середовище
python3 -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Встановити залежності
pip install -r requirements.txt
```

### 2️⃣ Перевірити базову функціональність

```bash
# Перевірити імпорти
python3 -c "from src.core import Timer, summarize; print('✓ Core modules OK')"
python3 -c "from src.problems import tsp_euclidean; print('✓ Problems modules OK')"
```

### 3️⃣ Згенерувати тестову задачу

```python
# test_basic.py
from src.problems import tsp_euclidean
from src.core import tour_length
import math

# Згенерувати TSP з 20 міст
coords = tsp_euclidean(n=20, seed=42)

# Створити матрицю відстаней
n = len(coords)
dist = [[0.0] * n for _ in range(n)]
for i in range(n):
    for j in range(i+1, n):
        dx = coords[i][0] - coords[j][0]
        dy = coords[i][1] - coords[j][1]
        d = math.sqrt(dx*dx + dy*dy)
        dist[i][j] = dist[j][i] = d

# Випадковий тур
tour = list(range(n))
length = tour_length(tour, dist)

print(f"✓ TSP instance: {n} cities")
print(f"✓ Tour length: {length:.2f}")
```

```bash
python test_basic.py
```

### 4️⃣ Завантажити приклади датасетів

```bash
python src/problems/download_datasets.py
```

### 5️⃣ Тестувати завантажувачі

```python
# test_loaders.py
from src.problems import load_tsplib

# Завантажити TSPLIB інстанс
instance = load_tsplib('data/tsp/gr17.tsp')

print(f"Name: {instance['name']}")
print(f"Dimension: {instance['dimension']}")
print(f"Coordinates: {len(instance['coords'])} cities")
print(f"✓ TSPLIB loader OK")
```

```bash
python test_loaders.py
```

---

## 📝 Наступні кроки

Після успішної перевірки базової інфраструктури:

1. **Фаза 2**: Реалізувати локальний пошук (HC, SD, RLS)
2. **Фаза 3**: Додати Simulated Annealing
3. **Фаза 4**: Наближені алгоритми
4. **Фаза 8**: Драйвери експериментів

---

## 🐛 Troubleshooting

### Помилка: ModuleNotFoundError
```bash
# Переконайтесь, що ви в правильній директорії
pwd  # має бути /path/to/blended5

# Переконайтесь, що активовано venv
which python  # має вказувати на .venv/bin/python
```

### Помилка: No module named 'numpy'
```bash
# Переустановити залежності
pip install --upgrade pip
pip install -r requirements.txt
```

### Помилка завантаження датасетів
```bash
# Перевірити інтернет з'єднання
ping comopt.ifi.uni-heidelberg.de

# Або завантажити вручну:
cd data/tsp
wget http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/gr17.tsp
```

---

## 📚 Корисні команди

```bash
# Використовуючи Makefile
make venv          # Створити venv
make install       # Встановити залежності
make download-data # Завантажити датасети
make clean         # Очистити cache

# Або вручну
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/problems/download_datasets.py
```

---

## 💡 Приклади коду

### Створити та оцінити TSP розв'язок

```python
from src.problems import tsp_euclidean
from src.core import tour_length, two_opt_delta, apply_two_opt
import random

# Генерувати інстанс
coords = tsp_euclidean(50, seed=42)

# Відстані
def dist(i, j):
    dx = coords[i][0] - coords[j][0]
    dy = coords[i][1] - coords[j][1]
    return (dx*dx + dy*dy)**0.5

# Випадковий тур
tour = list(range(len(coords)))
random.shuffle(tour)

# Обчислити довжину
initial_length = tour_length(tour, dist)
print(f"Initial tour length: {initial_length:.2f}")

# Спробувати 2-opt хід
i, k = 1, 10
delta = two_opt_delta(tour, i, k, dist)
print(f"2-opt delta: {delta:.2f}")

if delta < 0:
    apply_two_opt(tour, i, k)
    new_length = tour_length(tour, dist)
    print(f"Improved to: {new_length:.2f}")
```

### Використати метрики

```python
from src.core.metrics import summarize, gap_to_optimal

# Результати з 5 прогонів
results = [1245.3, 1198.7, 1267.1, 1210.5, 1232.8]

# Статистика
stats = summarize(results)
print(f"Best: {stats['best']:.2f}")
print(f"Median: {stats['median']:.2f}")
print(f"Mean: {stats['mean']:.2f}")
print(f"Std: {stats['std']:.2f}")

# Gap до оптимуму
optimal = 1150.0
gap = gap_to_optimal(stats['best'], optimal)
print(f"Gap to optimal: {gap:.2f}%")
```

---

**Готово!** Базова інфраструктура працює, можна переходити до реалізації алгоритмів.
