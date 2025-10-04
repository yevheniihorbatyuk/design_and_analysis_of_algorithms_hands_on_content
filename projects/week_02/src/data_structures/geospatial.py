# src/data_structures/geospatial.py

import numpy as np

# --- KD-Tree Implementation ---
class KDNode:
    """Вузол KD-дерева."""
    def __init__(self, point, axis, left=None, right=None):
        self.point = point
        self.axis = axis
        self.left = left
        self.right = right

class KDTree:
    """
    KD-дерево для ефективного просторового пошуку (наприклад, найближчих сусідів).
    """
    def __init__(self, points):
        self.k = len(points[0]) if points else 0
        self.root = self._build_tree(points, depth=0)

    def _build_tree(self, points, depth):
        if not points:
            return None
        
        axis = depth % self.k
        points.sort(key=lambda p: p[axis])
        median_idx = len(points) // 2
        
        return KDNode(
            point=points[median_idx],
            axis=axis,
            left=self._build_tree(points[:median_idx], depth + 1),
            right=self._build_tree(points[median_idx + 1:], depth + 1)
        )

    def find_nearest_neighbor(self, query_point):
        best = [None, float('inf')] # [point, distance_sq]

        def search(node):
            if node is None:
                return

            dist_sq = sum((query_point[i] - node.point[i]) ** 2 for i in range(self.k))
            if dist_sq < best[1]:
                best[0] = node.point
                best[1] = dist_sq

            axis = node.axis
            diff = query_point[axis] - node.point[axis]
            
            near, far = (node.left, node.right) if diff < 0 else (node.right, node.left)
            
            search(near)
            # Прунінг: перевіряємо, чи може бути точка в іншому піддереві
            if diff**2 < best[1]:
                search(far)
        
        search(self.root)
        return best[0], np.sqrt(best[1])

# --- Quadtree Implementation ---
class QuadTreeNode:
    """Вузол Quadtree."""
    def __init__(self, boundary, capacity=4):
        # boundary is a tuple (x, y, width, height)
        self.boundary = boundary
        self.capacity = capacity
        self.points = []
        self.divided = False
        self.children = [] # NW, NE, SW, SE

class QuadTree:
    """
    Quadtree для ефективного пошуку в діапазоні (range query).
    """
    def __init__(self, boundary, capacity=4):
        self.root = QuadTreeNode(boundary, capacity)
    
    def insert(self, point):
        self._insert_recursive(self.root, point)

    def _insert_recursive(self, node, point):
        x, y, w, h = node.boundary
        px, py = point
        if not (x <= px < x + w and y <= py < y + h):
            return False

        if len(node.points) < node.capacity and not node.divided:
            node.points.append(point)
            return True

        if not node.divided:
            self._subdivide(node)
        
        for child in node.children:
            if self._insert_recursive(child, point):
                return True

    def _subdivide(self, node):
        x, y, w, h = node.boundary
        hw, hh = w / 2, h / 2
        
        nw = QuadTreeNode((x, y, hw, hh), node.capacity)
        ne = QuadTreeNode((x + hw, y, hw, hh), node.capacity)
        sw = QuadTreeNode((x, y + hh, hw, hh), node.capacity)
        se = QuadTreeNode((x + hw, y + hh, hw, hh), node.capacity)
        node.children = [nw, ne, sw, se]
        node.divided = True

        for p in node.points:
            self.insert(p)
        node.points = []

    def query_range(self, query_boundary):
        found = []
        self._query_recursive(self.root, query_boundary, found)
        return found
    
    def _query_recursive(self, node, query_boundary, found):
        if not self._intersects(node.boundary, query_boundary):
            return

        for p in node.points:
            qx, qy, qw, qh = query_boundary
            px, py = p
            if qx <= px < qx + qw and qy <= py < qy + qh:
                found.append(p)
        
        if node.divided:
            for child in node.children:
                self._query_recursive(child, query_boundary, found)

    def _intersects(self, b1, b2):
        x1, y1, w1, h1 = b1
        x2, y2, w2, h2 = b2
        return not (x1 + w1 < x2 or x2 + w2 < x1 or y1 + h1 < y2 or y2 + h2 < y1)