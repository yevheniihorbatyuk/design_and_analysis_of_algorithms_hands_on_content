"""
Модуль реалізації жадібних алгоритмів для демонстрації
Author: Educational Tutorial
Python version: 3.8+
"""

from typing import List, Tuple, Dict, Optional
import heapq
from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class Activity:
    """Представлення заходу для задачі про вибір заявок"""
    name: str
    start_time: int
    end_time: int
    
    def __lt__(self, other):
        return self.end_time < other.end_time


@dataclass 
class Item:
    """Представлення предмета для задачі про рюкзак"""
    name: str
    weight: float
    value: float
    
    @property
    def value_per_weight(self) -> float:
        """Обчислення ціності на одиницю ваги"""
        return self.value / self.weight if self.weight > 0 else 0
    
    def __lt__(self, other):
        return self.value_per_weight > other.value_per_weight


class GreedyAlgorithms:
    """Клас для демонстрації жадібних алгоритмів"""
    
    @staticmethod
    def activity_selection(activities: List[Activity]) -> Tuple[List[Activity], List[int]]:
        """
        Розв'язує задачу про вибір заявок (Activity Selection Problem)
        
        Args:
            activities: Список заходів з часом початку та кінця
            
        Returns:
            Tuple[List[Activity], List[int]]: Обрані заходи та їх індекси
        """
        if not activities:
            return [], []
        
        # Сортуємо за часом закінчення
        sorted_activities = sorted(enumerate(activities), key=lambda x: x[1].end_time)
        
        selected = []
        selected_indices = []
        last_end_time = -1
        
        for i, activity in sorted_activities:
            if activity.start_time >= last_end_time:
                selected.append(activity)
                selected_indices.append(i)
                last_end_time = activity.end_time
        
        return selected, selected_indices
    
    @staticmethod
    def fractional_knapsack(items: List[Item], capacity: float) -> Tuple[float, List[Tuple[Item, float]]]:
        """
        Розв'язує задачу про дробовий рюкзак (Fractional Knapsack)
        
        Args:
            items: Список предметів з вагою та цінністю
            capacity: Місткість рюкзака
            
        Returns:
            Tuple[float, List[Tuple[Item, float]]]: Максимальна цінність та список (предмет, частка)
        """
        if not items:
            return 0.0, []
        
        # Сортуємо за спаданням ціності на одиницю ваги
        sorted_items = sorted(items, reverse=True)
        
        total_value = 0.0
        selected_items = []
        remaining_capacity = capacity
        
        for item in sorted_items:
            if remaining_capacity <= 0:
                break
                
            if item.weight <= remaining_capacity:
                # Беремо предмет цілком
                selected_items.append((item, 1.0))
                total_value += item.value
                remaining_capacity -= item.weight
            else:
                # Беремо частину предмета
                fraction = remaining_capacity / item.weight
                selected_items.append((item, fraction))
                total_value += item.value * fraction
                remaining_capacity = 0
        
        return total_value, selected_items
    
    @staticmethod
    def greedy_coin_change(coins: List[int], amount: int) -> Tuple[List[int], int]:
        """
        Жадібний розв'язок задачі про розмін монет
        УВАГА: Не завжди дає оптимальний результат!
        
        Args:
            coins: Список номіналів монет (у порядку спадання)
            amount: Сума для розміну
            
        Returns:
            Tuple[List[int], int]: Список використаних монет та їх кількість
        """
        coins_sorted = sorted(coins, reverse=True)
        result = []
        remaining = amount
        
        for coin in coins_sorted:
            while remaining >= coin:
                result.append(coin)
                remaining -= coin
        
        return result, len(result)


class HuffmanCoding:
    """Клас для демонстрації алгоритму Гаффмана"""
    
    @dataclass
    class Node:
        """Вузол дерева Гаффмана"""
        char: Optional[str]
        freq: int
        left: Optional['HuffmanCoding.Node'] = None
        right: Optional['HuffmanCoding.Node'] = None
        
        def __lt__(self, other):
            return self.freq < other.freq
    
    def __init__(self, text: str):
        """
        Ініціалізація з текстом для кодування
        
        Args:
            text: Текст для кодування
        """
        self.text = text
        self.freq_table = self._build_frequency_table()
        self.root = self._build_huffman_tree()
        self.codes = self._generate_codes()
    
    def _build_frequency_table(self) -> Dict[str, int]:
        """Будує таблицю частот символів"""
        freq_table = {}
        for char in self.text:
            freq_table[char] = freq_table.get(char, 0) + 1
        return freq_table
    
    def _build_huffman_tree(self) -> Node:
        """Будує дерево Гаффмана жадібним способом"""
        heap = [self.Node(char, freq) for char, freq in self.freq_table.items()]
        heapq.heapify(heap)
        
        while len(heap) > 1:
            # Жадібно об'єднуємо два найменші вузли
            left = heapq.heappop(heap)
            right = heapq.heappop(heap)
            
            merged = self.Node(None, left.freq + right.freq, left, right)
            heapq.heappush(heap, merged)
        
        return heap[0] if heap else self.Node(None, 0)
    
    def _generate_codes(self) -> Dict[str, str]:
        """Генерує коди для символів"""
        codes = {}
        
        def traverse(node, code=""):
            if node.char is not None:
                codes[node.char] = code or "0"  # Для одного символа
            else:
                if node.left:
                    traverse(node.left, code + "0")
                if node.right:
                    traverse(node.right, code + "1")
        
        if self.root:
            traverse(self.root)
        return codes
    
    def encode(self) -> str:
        """Кодує текст"""
        return ''.join(self.codes.get(char, '') for char in self.text)
    
    def get_compression_stats(self) -> Dict[str, float]:
        """Повертає статистику стиснення"""
        original_bits = len(self.text) * 8  # ASCII
        encoded_bits = len(self.encode())
        compression_ratio = encoded_bits / original_bits if original_bits > 0 else 0
        
        return {
            'original_bits': original_bits,
            'encoded_bits': encoded_bits,
            'compression_ratio': compression_ratio,
            'space_saved': 1 - compression_ratio
        }


class GreedyVisualizer:
    """Клас для візуалізації жадібних алгоритмів"""
    
    @staticmethod
    def visualize_activity_selection(activities: List[Activity], selected: List[Activity]):
        """Візуалізує розв'язок задачі про вибір заявок"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Всі заходи
        for i, activity in enumerate(activities):
            color = 'green' if activity in selected else 'lightgray'
            alpha = 1.0 if activity in selected else 0.5
            ax.barh(i, activity.end_time - activity.start_time, 
                   left=activity.start_time, color=color, alpha=alpha,
                   label='Обрано' if activity in selected and i == 0 else 
                         'Не обрано' if activity not in selected and i == 0 else "")
            ax.text(activity.start_time + (activity.end_time - activity.start_time)/2, i,
                   activity.name, ha='center', va='center', fontweight='bold')
        
        ax.set_xlabel('Час')
        ax.set_ylabel('Заходи')
        ax.set_title('Задача про вибір заявок (Activity Selection)\nЖадібний алгоритм: сортування за часом закінчення')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def visualize_knapsack(items: List[Item], selected: List[Tuple[Item, float]], capacity: float):
        """Візуалізує розв'язок задачі про дробовий рюкзак"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Лівий графік: порівняння ціності/ваги
        names = [item.name for item in items]
        ratios = [item.value_per_weight for item in items]
        colors = ['green' if any(item.name == sel[0].name for sel in selected) else 'lightblue' 
                 for item in items]
        
        bars1 = ax1.bar(names, ratios, color=colors)
        ax1.set_title('Цінність на одиницю ваги')
        ax1.set_ylabel('Цінність/Вага')
        ax1.tick_params(axis='x', rotation=45)
        
        # Правий графік: використання рюкзака
        used_weight = sum(item.weight * fraction for item, fraction in selected)
        ax2.pie([used_weight, capacity - used_weight], 
               labels=[f'Використано: {used_weight:.1f}', f'Вільно: {capacity - used_weight:.1f}'],
               autopct='%1.1f%%', startangle=90)
        ax2.set_title(f'Використання рюкзака (місткість: {capacity})')
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def visualize_huffman_tree(huffman: HuffmanCoding):
        """Візуалізує дерево Гаффмана та статистику"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Лівий графік: частоти символів
        chars = list(huffman.freq_table.keys())
        freqs = list(huffman.freq_table.values())
        
        bars = ax1.bar(chars, freqs, color='skyblue')
        ax1.set_title('Частоти символів')
        ax1.set_xlabel('Символи')
        ax1.set_ylabel('Частота')
        
        # Додаємо коди на стовпчики
        for i, (char, bar) in enumerate(zip(chars, bars)):
            code = huffman.codes.get(char, '')
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"'{code}'", ha='center', va='bottom', fontsize=8)
        
        # Правий графік: статистика стиснення
        stats = huffman.get_compression_stats()
        categories = ['Оригінал\n(ASCII)', 'Гаффман']
        sizes = [stats['original_bits'], stats['encoded_bits']]
        colors = ['lightcoral', 'lightgreen']
        
        ax2.bar(categories, sizes, color=colors)
        ax2.set_title(f"Стиснення: {stats['space_saved']:.1%} економії")
        ax2.set_ylabel('Біти')
        
        # Додаємо значення на стовпчики
        for i, size in enumerate(sizes):
            ax2.text(i, size + max(sizes) * 0.01, str(size), 
                    ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
