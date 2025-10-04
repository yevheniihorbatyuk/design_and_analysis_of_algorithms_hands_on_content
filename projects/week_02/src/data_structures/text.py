# src/data_structures/text.py

class TrieNode:
    """Вузол префіксного дерева (Trie)."""
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False
        self.count = 0  # Для підрахунку частоти слів

class Trie:
    """
    Префіксне дерево для ефективного зберігання та пошуку рядків.
    """
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True
        node.count += 1

    def search(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end_of_word

    def starts_with(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True

    def autocomplete(self, prefix, limit=5):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]
        
        suggestions = []
        self._collect_words(node, prefix, suggestions)
        
        # Сортуємо за частотою (спадання), потім за абеткою
        suggestions.sort(key=lambda x: (-x[1], x[0]))
        
        return [word for word, count in suggestions[:limit]]

    def _collect_words(self, node, current_prefix, suggestions):
        if node.is_end_of_word:
            suggestions.append((current_prefix, node.count))
        
        for char, child_node in sorted(node.children.items()):
            self._collect_words(child_node, current_prefix + char, suggestions)