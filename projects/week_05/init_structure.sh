#!/usr/bin/env bash
set -euo pipefail

# Використання:
#   ./init_structure.sh            # створить у папці blended5
#   ./init_structure.sh myproj     # створить у папці myproj
# Вкажи корінь проєкту (може бути ".")
PROJECT_DIR="."

# Допоміжна функція: створити порожній файл (із батьківськими каталогами)
mkempty() {
  local path="${1:-}"
  # Пропускаємо порожні рядки та коментарі
  [ -z "$path" ] && return 0
  case "$path" in
    \#*) return 0 ;;
  esac

  mkdir -p "$(dirname "$PROJECT_DIR/$path")"
  : > "$PROJECT_DIR/$path"
}

# 1) Каталоги
mkdir -p "$PROJECT_DIR"

# Головні каталоги
while read -r d; do
  mkdir -p "$PROJECT_DIR/$d"
done <<'DIRS'
src/core
src/meta
src/approx
src/randomized
src/markov
src/problems
experiments/configs
frameworks/docker
frameworks/k8s
data/tsp
data/knapsack
data/coloring
data/maxsat
docs
tests
DIRS

# 2) Порожні файли-пакети (__init__.py), верхньорівневі файли та модулі
while IFS= read -r f; do
  [[ -z "$f" || "$f" == \#* ]] && continue
  mkempty "$f"
done <<'FILES'
# Python пакети
src/__init__.py
src/core/__init__.py
src/meta/__init__.py
src/approx/__init__.py
src/randomized/__init__.py
src/markov/__init__.py
src/problems/__init__.py

# Core
src/core/utils.py
src/core/metrics.py
src/core/neighborhoods.py
src/core/local_search.py

# Metaheuristics
src/meta/sa.py
src/meta/tabu.py
src/meta/vns.py
src/meta/ils.py
src/meta/grasp.py
src/meta/lns.py

# Approximation
src/approx/vertex_cover.py
src/approx/set_cover.py
src/approx/tsp_mst_doubletree.py
src/approx/knapsack_fptas.py

# Randomized
src/randomized/karger_mincut.py
src/randomized/miller_rabin.py
src/randomized/freivalds.py

# Markov
src/markov/pagerank.py
src/markov/sa_diagnostics.py

# Problems (генератори/лоадери/завантаження датасетів)
src/problems/generators.py
src/problems/loaders.py
src/problems/download_datasets.py

# Експерименти: драйвери
experiments/driver.py
experiments/ray_driver.py
experiments/dask_driver.py

# Експерименти: 4 YAML конфіги
experiments/configs/tsp_ls.yml
experiments/configs/tsp_sa_grid.yml
experiments/configs/approx_polish.yml
experiments/configs/knapsack_fptas.yml

# Frameworks (Docker/K8s, порожні файли-заглушки)
frameworks/docker/Dockerfile
frameworks/docker/docker-compose.yml
frameworks/k8s/deployment.yaml
frameworks/k8s/service.yaml
frameworks/k8s/README.md

# Data readme
data/README.md

# Docs
docs/README.md

# Tests
tests/__init__.py
tests/test_placeholder.py

# Верхньорівневі файли
requirements.txt
Makefile
README.md
.gitignore
FILES

# 3) .gitkeep, щоб порожні каталоги гарантовано відслідковувалися git-ом
for sub in data/tsp data/knapsack data/coloring data/maxsat frameworks/docker frameworks/k8s tests; do
  mkempty "$sub/.gitkeep"
done

# 4) Опціонально показати підсумкову структуру
if command -v tree >/dev/null 2>&1; then
  echo
  tree -a "$PROJECT_DIR"
else
  echo
  echo "Створено структуру в: $PROJECT_DIR"
  echo "Порада: встановіть 'tree' для зручного перегляду дерева каталогів."
fi
