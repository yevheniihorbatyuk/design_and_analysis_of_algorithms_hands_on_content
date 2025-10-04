
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