# Blended-5: Локальний пошук, метаевристики, наближення та рандомізація

Повний навчальний репозиторій для вивчення алгоритмів оптимізації та дистрибутивних обчислень.

## 📚 Теми курсу

### Модуль A: Локальний пошук та градієнтні методи
- Hill Climbing (HC)
- Steepest Descent (SD)
- Randomized Local Search (RLS)
- Градієнтний спуск (GD/SGD/Adam)

### Модуль B: Метаевристики
- Simulated Annealing (SA)
- Tabu Search (TS)
- Variable Neighborhood Search (VNS)
- Iterated Local Search (ILS)
- GRASP
- Large Neighborhood Search (LNS/ALNS)
- Late Acceptance Hill Climbing (LAHC)

### Модуль C: Теорія складності та наближені алгоритми
- Vertex Cover (2-approximation)
- Set Cover ((1+ln n)-approximation)
- TSP (MST-doubletree, Christofides)
- Knapsack (FPTAS)

### Модуль D: Рандомізовані алгоритми
- Karger Min-Cut
- Miller-Rabin primality test
- Freivalds matrix verification

### Модуль E: Марковські ланцюги
- PageRank
- SA діагностика

## 🚀 Швидкий старт

```bash
# 1. Створити віртуальне середовище
make venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# 2. Встановити залежності
make install

# 3. Завантажити приклади датасетів
make download-data

# 4. Запустити локальний експеримент
make run-local
```

## 📁 Структура проекту

```
blended5/
├── src/
│   ├── core/              # Базові утиліти та локальний пошук
│   ├── meta/              # Метаевристики (SA, TS, VNS, ILS, GRASP, LNS)
│   ├── approx/            # Наближені алгоритми
│   ├── randomized/        # Рандомізовані алгоритми
│   ├── markov/            # Марковські ланцюги
│   └── problems/          # Датасети та генератори
├── experiments/           # Драйвери експериментів
│   ├── driver.py          # Локальний драйвер
│   ├── ray_driver.py      # Ray паралелізація
│   ├── dask_driver.py     # Dask паралелізація
│   └── configs/           # YAML конфігурації
├── frameworks/            # Docker/K8s конфігурації
├── data/                  # Бенчмарк датасети
│   ├── tsp/              # TSPLIB інстанси
│   ├── knapsack/         # OR-Library
│   ├── coloring/         # DIMACS graphs
│   └── maxsat/           # SATLIB + MaxSAT
└── docs/                  # Документація
```

## 📊 Датасети

Проект підтримує стандартні бенчмарки:

- **TSP**: TSPLIB95 (http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/)
- **Knapsack**: OR-Library (https://people.brunel.ac.uk/~mastjjb/jeb/orlib/)
- **Graph Coloring**: DIMACS (https://mat.tepper.cmu.edu/COLOR/instances.html)
- **Max-SAT**: SATLIB (https://www.cs.ubc.ca/~hoos/SATLIB/benchm.html)

## 🧪 Приклади експериментів

### Локальний пошук на TSP
```bash
python experiments/driver.py --config experiments/configs/tsp_ls.yml
```

### Grid search SA параметрів з Ray
```bash
python experiments/ray_driver.py --config experiments/configs/tsp_sa_grid.yml
```

### Approx → Polish (MST-doubletree → 2-opt)
```python
from src.approx.tsp_mst_doubletree import tsp_mst_doubletree
from src.core.local_search import ls_tsp_2opt, LSConfig

# 1. Наближене рішення
coords = [...] # TSP координати
tour = tsp_mst_doubletree(coords)

# 2. Полірування локальним пошуком
result = ls_tsp_2opt(coords, seed=42, time_budget_s=30, 
                     cfg=LSConfig(acceptance='best'))
```

## 🐳 Docker

```bash
cd frameworks/docker
docker build -t blended5:latest .
docker run --rm blended5:latest
```

## ☸️ Kubernetes (kind)

```bash
# Створити локальний кластер
kind create cluster --name blended5 --config frameworks/kind/kind-cluster.yaml

# Deploy Ray
kubectl apply -f frameworks/ray/ray-cluster.yaml

# Deploy Dask
helm install dask dask/dask --namespace dask --create-namespace
```

## 📖 Документація

- [Cheat Sheets](docs/cheat_sheets.md) — шпаргалки по алгоритмах
- [Instructor Guide](docs/instructor_guide.md) — гайд для викладача
- [Scenario](docs/scenario.md) — сценарій заняття

## 🧮 Метрики

Експерименти логують:
- `best_value` — найкраще знайдене значення
- `median` / `p95` — статистика по множині запусків
- `time_s` — час виконання
- `iters` — кількість ітерацій
- `accept_rate` — коефіцієнт прийняття (для SA)

## 📝 Цитування

Якщо використовуєте цей код у дослідженнях:
```
@misc{blended5-2025,
  title={Blended-5: Local Search, Metaheuristics, and Approximation Algorithms},
  author={Design and Analysis of Algorithms Course},
  year={2025},
  url={https://github.com/your-repo/blended5}
}
```

## 📄 Ліцензія

MIT License

## 🤝 Контрибуція

Вітаються pull requests з:
- Новими алгоритмами
- Покращеннями існуючих реалізацій
- Додатковими бенчмарками
- Виправленнями помилок
