# src/data_structures/union_find.py

#
class UnionFind:
    """
    Структура даних 'Система неперетинних множин' (Disjoint Set Union - DSU).
    Ефективно відстежує компоненти зв'язності в динамічному графі.
    """
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.num_components = n

    def find(self, i):
        """Знаходить представника (корінь) множини для елемента i."""
        if self.parent[i] == i:
            return i
        # Path compression: робимо всі вузли на шляху до кореня його прямими нащадками
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        """Об'єднує множини, що містять елементи i та j."""
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            # Union by rank: приєднуємо дерево меншого рангу до дерева більшого рангу
            if self.rank[root_i] > self.rank[root_j]:
                self.parent[root_j] = root_i
            else:
                self.parent[root_i] = root_j
                if self.rank[root_i] == self.rank[root_j]:
                    self.rank[root_j] += 1
            self.num_components -= 1
            return True
        return False
        return False