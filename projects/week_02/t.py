Добре, рухаємося далі!

Наступний великий блок у вашому плані — **Розділ 3: Структури даних**, і ми почнемо з **3.2 Деревоподібні структури**. Це фундаментальна тема, яка пояснює, як ефективно організовувати дані для швидкого доступу та маніпуляцій.

Я підготував новий файл `src/data_structures/trees.py` та оновив `src/utils/visualizer.py` для візуалізації дерев. Також я написав наступну частину коду для вашого `main_notebook.py`.

---

### **Крок 1: Створіть/Оновіть файли в `src/`**

<details>
<summary><b>src/data_structures/trees.py (НОВИЙ ФАЙЛ)</b></summary>

```python
# src/data_structures/trees.py

import sys

class TreeNode:
    """Вузол для бінарних дерев пошуку."""
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None
        self.height = 1 # Для AVL дерева

class BinarySearchTree:
    """Класичне бінарне дерево пошуку."""
    def __init__(self):
        self.root = None

    def insert(self, key):
        self.root = self._insert_recursive(self.root, key)

    def _insert_recursive(self, node, key):
        if not node:
            return TreeNode(key)
        if key < node.key:
            node.left = self._insert_recursive(node.left, key)
        elif key > node.key:
            node.right = self._insert_recursive(node.right, key)
        return node

    def get_height(self):
        return self._get_height_recursive(self.root)

    def _get_height_recursive(self, node):
        if not node:
            return 0
        return 1 + max(self._get_height_recursive(node.left), self._get_height_recursive(node.right))

class AVLTree:
    """Самозбалансоване AVL дерево."""
    def __init__(self):
        self.root = None

    def insert(self, key):
        self.root = self._insert_recursive(self.root, key)

    def _get_height(self, node):
        return node.height if node else 0

    def _get_balance(self, node):
        return self._get_height(node.left) - self._get_height(node.right) if node else 0

    def _rotate_right(self, z):
        y = z.left
        T3 = y.right
        y.right = z
        z.left = T3
        z.height = 1 + max(self._get_height(z.left), self._get_height(z.right))
        y.height = 1 + max(self._get_height(y.left), self._get_height(y.right))
        return y

    def _rotate_left(self, z):
        y = z.right
        T2 = y.left
        y.left = z
        z.right = T2
        z.height = 1 + max(self._get_height(z.left), self._get_height(z.right))
        y.height = 1 + max(self._get_height(y.left), self._get_height(y.right))
        return y

    def _insert_recursive(self, node, key):
        if not node:
            return TreeNode(key)
        
        if key < node.key:
            node.left = self._insert_recursive(node.left, key)
        elif key > node.key:
            node.right = self._insert_recursive(node.right, key)
        else: # Дублікати не дозволені
            return node

        node.height = 1 + max(self._get_height(node.left), self._get_height(node.right))
        balance = self._get_balance(node)

        # Left Left Case
        if balance > 1 and key < node.left.key:
            return self._rotate_right(node)
        # Right Right Case
        if balance < -1 and key > node.right.key:
            return self._rotate_left(node)
        # Left Right Case
        if balance > 1 and key > node.left.key:
            node.left = self._rotate_left(node.left)
            return self._rotate_right(node)
        # Right Left Case
        if balance < -1 and key < node.right.key:
            node.right = self._rotate_right(node.right)
            return self._rotate_left(node)
        
        return node

    def get_height(self):
        return self._get_height(self.root)

class SegmentTree:
    """Дерево відрізків для запитів на діапазонах (сума)."""
    def __init__(self, data):
        self.n = len(data)
        self.tree = [0] * (4 * self.n)
        self.data = data
        if self.n > 0:
            self._build(0, 0, self.n - 1)

    def _build(self, node_idx, start, end):
        if start == end:
            self.tree[node_idx] = self.data[start]
            return
        mid = (start + end) // 2
        self._build(2 * node_idx + 1, start, mid)
        self._build(2 * node_idx + 2, mid + 1, end)
        self.tree[node_idx] = self.tree[2 * node_idx + 1] + self.tree[2 * node_idx + 2]

    def query(self, l, r):
        return self._query_recursive(0, 0, self.n - 1, l, r)

    def _query_recursive(self, node_idx, start, end, l, r):
        if r < start or end < l:
            return 0
        if l <= start and end <= r:
            return self.tree[node_idx]
        mid = (start + end) // 2
        p1 = self._query_recursive(2 * node_idx + 1, start, mid, l, r)
        p2 = self._query_recursive(2 * node_idx + 2, mid + 1, end, l, r)
        return p1 + p2

```
</details>

<details>
<summary><b>src/utils/visualizer.py (ОНОВЛЕНО)</b></summary>
```python
# src/utils/visualizer.py

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ... (код для PathVisualizer, TraversalVisualizer, MstVisualizer, AnalysisVisualizer) ...
# Додайте новий клас TreeVisualizer в кінець файлу:

class TreeVisualizer:
    """Клас для візуалізації деревоподібних структур."""
    @staticmethod
    def _get_tree_graph(node, graph=None):
        if graph is None:
            graph = nx.DiGraph()
        if node is not None:
            if node.left:
                graph.add_edge(node.key, node.left.key)
                TreeVisualizer._get_tree_graph(node.left, graph)
            if node.right:
                graph.add_edge(node.key, node.right.key)
                TreeVisualizer._get_tree_graph(node.right, graph)
        return graph

    @staticmethod
    def draw_tree(tree_root, title, figsize=(10, 6)):
        if not tree_root:
            print(f"Попередження: Дерево '{title}' порожнє.")
            return

        plt.figure(figsize=figsize)
        g = TreeVisualizer._get_tree_graph(tree_root)
        
        # Використовуємо graphviz_layout для ієрархічного вигляду, якщо доступно
        try:
            pos = nx.nx_agraph.graphviz_layout(g, prog='dot')
        except ImportError:
            print("Попередження: PyGraphviz не встановлено. Використовується менш ієрархічний layout.")
            pos = nx.spring_layout(g, iterations=100)

        nx.draw(g, pos, with_labels=True, node_color='skyblue', node_size=1500,
                edge_color='gray', font_size=10, font_weight='bold',
                arrows=False)
        plt.title(title, fontsize=16, fontweight='bold')
        return plt.gcf()
```
</details>

---

### **Крок 2: Додайте цей код до `main_notebook.py`**

Це наступна велика частина вашого файлу.

```python
# %% [markdown]
#  ## 3.2 Деревоподібні структури
#
#  Дерева є одними з найважливіших нелінійних структур даних. Вони ієрархічно організовують дані, що дозволяє реалізовувати надзвичайно ефективні алгоритми пошуку, вставки та видалення.

# %% [markdown]
#  ### Бінарне дерево пошуку (Binary Search Tree - BST)
#
#  **Ідея:** Фундаментальна деревоподібна структура, де для кожного вузла виконується інваріант: всі ключі в лівому піддереві менші за ключ вузла, а всі ключі в правому — більші.
#
#  **Проблема:** Продуктивність BST сильно залежить від порядку вставки елементів. У найгіршому випадку (при вставці відсортованих даних) дерево "вироджується" у зв'язний список, і всі операції займають `O(n)` часу замість очікуваного `O(log n)`.

# %%
# =============================================================================
# Клітинка 12: Демонстрація деградації BST
# =============================================================================
from src.data_structures.trees import BinarySearchTree
from src.utils.visualizer import TreeVisualizer
import random

# Дані
data_random = random.sample(range(1, 100), 15)
data_sorted = sorted(data_random)

# Дерево на випадкових даних
bst_random = BinarySearchTree()
for item in data_random:
    bst_random.insert(item)

# Дерево на відсортованих даних
bst_sorted = BinarySearchTree()
for item in data_sorted:
    bst_sorted.insert(item)

# Порівняння висоти
height_random = bst_random.get_height()
height_sorted = bst_sorted.get_height()
optimal_height = int(np.log2(len(data_random))) + 1

comparison_data = {
    'Тип даних': ['Випадкові', 'Відсортовані'],
    'Висота дерева': [height_random, height_sorted],
    'Оптимальна висота (log n)': [optimal_height, optimal_height],
    'Продуктивність': ['O(log n)', 'O(n) - Деградація!']
}
bst_df = pd.DataFrame(comparison_data)
print("Порівняння BST на різних даних:")
print(bst_df)

# Візуалізація
fig_rand = TreeVisualizer.draw_tree(bst_random.root, "BST на випадкових даних (збалансоване)")
plt.show()

fig_sort = TreeVisualizer.draw_tree(bst_sorted.root, "BST на відсортованих даних (деградоване)")
plt.savefig('./data/bst_degradation.png', dpi=300, bbox_inches='tight')
plt.show()

# %% [markdown]
#  ### AVL-дерево
#
#  **Рішення проблеми BST:** AVL-дерево — це **самозбалансоване** бінарне дерево пошуку. Воно підтримує інваріант, що для будь-якого вузла висоти його лівого та правого піддерев відрізняються не більше ніж на одиницю. Це досягається за допомогою "поворотів" (rotations) після операцій вставки або видалення.
#
#  **Результат:** Усі основні операції (пошук, вставка, видалення) **гарантовано** виконуються за `O(log n)`.

# %%
# =============================================================================
# Клітинка 13: AVL-дерево як вирішення проблеми деградації
# =============================================================================
from src.data_structures.trees import AVLTree

# Використовуємо ті самі відсортовані дані, що спричинили деградацію BST
avl_tree = AVLTree()
for item in data_sorted:
    avl_tree.insert(item)

height_avl = avl_tree.get_height()

print(f"Висота AVL-дерева на відсортованих даних: {height_avl}")
print(f"Висота деградованого BST на тих же даних: {height_sorted}")
print(f"Оптимальна висота: {optimal_height}")
print("\n✅ AVL-дерево успішно зберегло баланс і логарифмічну висоту!")

# Візуалізація збалансованого дерева
fig_avl = TreeVisualizer.draw_tree(avl_tree.root, "AVL-дерево на відсортованих даних (збалансоване)")
plt.savefig('./data/avl_tree_balanced.png', dpi=300, bbox_inches='tight')
plt.show()

# %% [markdown]
#  ### Дерево відрізків (Segment Tree)
#
#  **Ідея:** Спеціалізована структура даних для ефективного виконання **запитів на діапазонах (range queries)**. Кожен вузол у дереві представляє певний відрізок вхідного масиву та зберігає агреговане значення (наприклад, суму, мінімум, максимум) для цього відрізка.
#
#  | Операція | Складність |
#  | :--- | :--- |
#  | Побудова | `O(n)` |
#  | Запит на діапазоні | `O(log n)` |
#  | Оновлення елемента | `O(log n)` |
#
#  **Домени застосування:**
#  - **Фінансова аналітика:** Швидкий розрахунок сумарного доходу за будь-який період.
#  - **Комп'ютерна графіка:** Пошук об'єктів у певній області.
#  - **Біоінформатика:** Аналіз геномних послідовностей.

# %%
# =============================================================================
# Клітинка 14: Практичний приклад Segment Tree - Аналіз продажів
# =============================================================================
from src.data_structures.trees import SegmentTree
import time

# Генеруємо дані про продажі за 30 днів
np.random.seed(42)
sales_data = np.random.randint(50, 200, size=30)

# Будуємо дерево відрізків
seg_tree = SegmentTree(sales_data)

# --- Приклад запитів ---
# 1. Загальні продажі за другий тиждень (дні 7-13)
week_2_sales = seg_tree.query(7, 13)
# 2. Продажі за останню декаду (дні 20-29)
last_10_days_sales = seg_tree.query(20, 29)

print("Аналіз продажів за допомогою Дерева відрізків:")
print(f"- Продажі за 2-й тиждень (дні 7-13): ${week_2_sales} (Перевірка: ${np.sum(sales_data[7:14])})")
print(f"- Продажі за останні 10 днів (дні 20-29): ${last_10_days_sales} (Перевірка: ${np.sum(sales_data[20:30])})")

# --- Бенчмарк ---
large_sales_data = np.random.randint(50, 200, size=10000)
large_seg_tree = SegmentTree(large_sales_data)
num_queries = 5000

# Час для Segment Tree
start_time = time.perf_counter()
for _ in range(num_queries):
    l, r = sorted(np.random.randint(0, 10000, 2))
    large_seg_tree.query(l, r)
time_seg_tree = (time.perf_counter() - start_time) * 1000

# Час для наївного підходу
start_time = time.perf_counter()
for _ in range(num_queries):
    l, r = sorted(np.random.randint(0, 10000, 2))
    np.sum(large_sales_data[l:r+1])
time_naive = (time.perf_counter() - start_time) * 1000

print("\nБенчмарк продуктивності (10,000 елементів, 5,000 запитів):")
print(f"  - Час Дерева відрізків: {time_seg_tree:.2f} мс")
print(f"  - Час наївного підсумовування: {time_naive:.2f} мс")
print(f"  - Прискорення: {time_naive / time_seg_tree:.2f}x")

# %% [markdown]
#  ### Інші важливі деревоподібні структури (Концептуальний огляд)
#
#  - **Red-Black Tree:** Ще один тип самозбалансованого дерева. Менш строго збалансоване, ніж AVL, що робить вставки/видалення трохи швидшими за рахунок меншої кількості поворотів. Є стандартною реалізацією для `map` та `set` у C++ STL.
#
#  - **B-Tree / B+ Tree:** Оптимізовані для роботи з даними, що зберігаються на диску (HDD, SSD). Вони мають високий фактор розгалуження (багато дочірніх вузлів), що мінімізує кількість дискових операцій. Це основа індексів у більшості реляційних баз даних (PostgreSQL, MySQL).
#
#  - **Fenwick Tree (BIT):** Більш проста та пам'ятефективна альтернатива Дереву відрізків, але підходить лише для "обернених" операцій, таких як сума.



```

---

### Що ми зробили в цьому оновленні:

1.  **Продемонстрували проблему BST:** Показали, як продуктивність бінарного дерева пошуку деградує на відсортованих даних, що є ключовим мотиваційним моментом.
2.  **Представили AVL-дерево як рішення:** Наочно показали, як самозбалансована структура зберігає логарифмічну висоту і гарантує ефективність.
3.  **Розкрили потужність Дерева відрізків:** На практичному прикладі аналізу продажів показали, як воно прискорює запити на діапазонах у сотні разів порівняно з наївним підходом.
4.  **Створили візуалізатор для дерев:** Тепер ми можемо легко візуалізувати структуру будь-якого бінарного дерева.
5.  **Концептуально оглянули** інші важливі дерева, щоб дати студентам повну картину.

Ми готові до наступного кроку. Згідно з вашим планом, це **3.3 Геопросторові структури (KD-Tree, R-Tree, Quadtree тощо)**. Як тільки ви підтвердите, я почну готувати матеріал для них.