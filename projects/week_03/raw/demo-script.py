#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Практична демонстрація алгоритмів для великих даних

Цей скрипт демонструє застосування різних алгоритмів для обробки 
великих даних на прикладі веб-логів та транзакцій.
"""

import os
import sys
import time
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Set, Tuple, Any, Optional
from collections import Counter, defaultdict
import multiprocessing

# Додаємо реалізації алгоритмів до шляху імпорту
sys.path.append('.')

# Імпортуємо наші реалізації алгоритмів
from data_generator import DataStreamGenerator
from reservoir_sampling import ReservoirSampling
from bloom_filter import BloomFilter
from hyperloglog import HyperLogLog
from count_min_sketch import CountMinSketch
from misra_gries import MisraGries, SpaceSaving
from locality_sensitive_hashing import MinHash, LSH, VectorLSH

# Константи для експериментів
DATA_SIZE = 10000
SAMPLE_SIZE = 100
ERROR_RATE = 0.01
DEMO_DIR = "demo_results"

# Створюємо директорію для результатів
os.makedirs(DEMO_DIR, exist_ok=True)

# Налаштування візуалізації
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 12

def print_header(title):
    """Виводить заголовок розділу."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80 + "\n")

def print_subheader(title):
    """Виводить підзаголовок розділу."""
    print("\n" + "-" * 80)
    print(f" {title} ".center(80, "-"))
    print("-" * 80 + "\n")

def save_plot(plt, filename):
    """Зберігає графік у файл."""
    plt.tight_layout()
    plt.savefig(os.path.join(DEMO_DIR, filename))
    print(f"Графік збережено у файл: {filename}")

def generate_data():
    """Генерує дані для демонстрації."""
    print_header("Генерація даних для демонстрації")
    
    # 1. Генеруємо веб-логи
    print("Генерація веб-логів...")
    web_generator = DataStreamGenerator(
        data_type="web_logs", 
        rate=1000,
        error_rate=0.05,
        duplicate_rate=0.1
    )
    
    web_logs = []
    for _ in range(DATA_SIZE):
        web_logs.append(next(web_generator.stream()))
    
    # 2. Генеруємо транзакції
    print("Генерація транзакцій...")
    tx_generator = DataStreamGenerator(
        data_type="transactions", 
        rate=1000,
        error_rate=0.02,
        duplicate_rate=0.05
    )
    
    transactions = []
    for _ in range(DATA_SIZE):
        transactions.append(next(tx_generator.stream()))
    
    # Виводимо приклади даних
    print("\nПриклад веб-логу:")
    print(json.dumps(web_logs[0], indent=2))
    
    print("\nПриклад транзакції:")
    print(json.dumps(transactions[0], indent=2))
    
    # Виводимо статистику
    print(f"\nЗгенеровано {len(web_logs)} веб-логів та {len(transactions)} транзакцій")
    
    return web_logs, transactions

def demo_reservoir_sampling(data):
    """Демонстрація алгоритму резервуарної вибірки."""
    print_header("Демонстрація алгоритму резервуарної вибірки")
    
    # Створюємо резервуарну вибірку
    reservoir = ReservoirSampling[Dict](SAMPLE_SIZE)
    
    # Додаємо дані до резервуару
    print(f"Додаємо {len(data)} елементів до резервуару розміром {SAMPLE_SIZE}...")
    start_time = time.time()
    reservoir.process_stream(iter(data))
    end_time = time.time()
    
    # Отримуємо вибірку
    sample = reservoir.get_sample()
    
    print(f"Час виконання: {end_time - start_time:.4f} секунд")
    print(f"Розмір вибірки: {len(sample)}")
    
    # Візуалізуємо розподіл вибірки (для веб-логів)
    if 'path' in data[0]:
        # Підраховуємо частоту шляхів у вибірці
        sample_paths = [item['path'] for item in sample]
        sample_path_counts = Counter(sample_paths)
        
        # Підраховуємо частоту шляхів у повному наборі
        all_paths = [item['path'] for item in data]
        all_path_counts = Counter(all_paths)
        
        # Вибираємо 10 найчастіших шляхів
        top_paths = [path for path, _ in all_path_counts.most_common(10)]
        
        # Обчислюємо частоти для цих шляхів
        original_freqs = [all_path_counts[path] / len(data) for path in top_paths]
        sample_freqs = [sample_path_counts[path] / len(sample) if path in sample_path_counts else 0 
                        for path in top_paths]
        
        # Візуалізуємо
        plt.figure(figsize=(14, 6))
        
        x = np.arange(len(top_paths))
        width = 0.35
        
        plt.bar(x - width/2, original_freqs, width, label='Оригінальний набір')
        plt.bar(x + width/2, sample_freqs, width, label='Вибірка')
        
        plt.xlabel('Шлях')
        plt.ylabel('Частота')
        plt.title('Порівняння розподілу шляхів у оригінальному наборі та вибірці')
        plt.xticks(x, top_paths, rotation=45, ha='right')
        plt.legend()
        
        save_plot(plt, "reservoir_sampling_comparison.png")
        plt.close()
        
    print("\nДемонстрація різних розмірів резервуару...")
    
    # Демонструємо вплив розміру резервуару на точність
    reservoir_sizes = [10, 50, 100, 500, 1000]
    kl_divergences = []
    
    for size in reservoir_sizes:
        reservoir = ReservoirSampling[Dict](size)
        reservoir.process_stream(iter(data))
        sample = reservoir.get_sample()
        
        # Обчислюємо KL-дивергенцію для оцінки якості вибірки (для веб-логів)
        if 'path' in data[0]:
            # Підраховуємо частоти шляхів
            sample_paths = [item['path'] for item in sample]
            sample_path_counts = Counter(sample_paths)
            
            all_paths = [item['path'] for item in data]
            all_path_counts = Counter(all_paths)
            
            # Обчислюємо KL-дивергенцію
            kl_div = 0
            for path in all_path_counts:
                p = all_path_counts[path] / len(data)
                q = sample_path_counts.get(path, 0) / len(sample) if len(sample) > 0 else 0.0001
                if q > 0:
                    kl_div += p * np.log(p / q)
            
            kl_divergences.append(kl_div)
    
    # Візуалізуємо вплив розміру резервуару на KL-дивергенцію
    plt.figure(figsize=(10, 6))
    plt.plot(reservoir_sizes, kl_divergences, marker='o')
    plt.xlabel('Розмір резервуару')
    plt.ylabel('KL-дивергенція')
    plt.title('Вплив розміру резервуару на якість вибірки')
    plt.grid(True)
    
    save_plot(plt, "reservoir_sampling_sizes.png")
    plt.close()

def demo_bloom_filter(data):
    """Демонстрація алгоритму фільтра Блума."""
    print_header("Демонстрація алгоритму фільтра Блума")
    
    # Для демонстрації використаємо IP-адреси з веб-логів або користувачів з транзакцій
    if 'ip' in data[0]:
        elements = [item['ip'] for item in data]
        element_type = "IP-адрес"
    elif 'user_id' in data[0]:
        elements = [item['user_id'] for item in data]
        element_type = "користувачів"
    else:
        print("Непідтримуваний тип даних для демонстрації фільтра Блума")
        return
    
    # Знаходимо унікальні елементи
    unique_elements = set(elements)
    print(f"Кількість унікальних {element_type}: {len(unique_elements)}")
    
    # Створюємо фільтр Блума з різними параметрами
    error_rates = [0.01, 0.05, 0.1, 0.2]
    
    results = []
    
    for error_rate in error_rates:
        print(f"\nСтворюємо фільтр Блума з ймовірністю помилки {error_rate}...")
        
        # Створюємо фільтр
        bloom = BloomFilter(capacity=len(unique_elements), error_rate=error_rate)
        
        # Додаємо елементи
        start_time = time.time()
        for element in unique_elements:
            bloom.add(element)
        add_time = time.time() - start_time
        
        # Тестуємо на елементах, які є у множині
        start_time = time.time()
        true_positives = 0
        for element in unique_elements:
            if bloom.contains(element):
                true_positives += 1
        check_existing_time = time.time() - start_time
        
        # Генеруємо тестові елементи, яких немає у множині
        non_existing = set()
        while len(non_existing) < 1000:
            if 'ip' in data[0]:
                # Генеруємо випадкові IP-адреси
                random_ip = f"{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}"
                if random_ip not in unique_elements:
                    non_existing.add(random_ip)
            else:
                # Генеруємо випадкові ID користувачів
                random_id = f"user_{random.randint(1000, 10000)}"
                if random_id not in unique_elements:
                    non_existing.add(random_id)
        
        # Тестуємо на елементах, яких немає у множині
        start_time = time.time()
        false_positives = 0
        for element in non_existing:
            if bloom.contains(element):
                false_positives += 1
        check_non_existing_time = time.time() - start_time
        
        # Обчислюємо фактичну частоту помилкових спрацьовувань
        false_positive_rate = false_positives / len(non_existing)
        
        print(f"Розмір фільтра: {bloom.size} біт (~{bloom.size / 8 / 1024:.2f} КБ)")
        print(f"Кількість хеш-функцій: {bloom.hash_count}")
        print(f"True Positives: {true_positives}/{len(unique_elements)} ({true_positives/len(unique_elements)*100:.2f}%)")
        print(f"False Positives: {false_positives}/{len(non_existing)} ({false_positive_rate*100:.2f}%)")
        print(f"Цільова частота помилок: {error_rate*100:.2f}%, Фактична: {false_positive_rate*100:.2f}%")
        
        # Оцінка пам'яті (порівняно з set)
        bloom_memory = bloom.size / 8  # в байтах
        set_memory = sys.getsizeof(unique_elements)
        memory_ratio = bloom_memory / set_memory
        
        print(f"Пам'ять фільтра Блума: {bloom_memory/1024:.2f} КБ, Пам'ять set: {set_memory/1024:.2f} КБ")
        print(f"Відношення: {memory_ratio:.4f} (економія: {(1-memory_ratio)*100:.2f}%)")
        
        results.append({
            "error_rate": error_rate,
            "filter_size_bits": bloom.size,
            "hash_functions": bloom.hash_count,
            "true_positive_rate": true_positives/len(unique_elements),
            "false_positive_rate": false_positive_rate,
            "memory_bloom_kb": bloom_memory/1024,
            "memory_set_kb": set_memory/1024,
            "memory_ratio": memory_ratio,
            "add_time": add_time,
            "check_existing_time": check_existing_time,
            "check_non_existing_time": check_non_existing_time
        })
    
    # Візуалізуємо результати
    plt.figure(figsize=(12, 8))
    
    # Графік 1: Цільова vs Фактична частота помилок
    plt.subplot(2, 2, 1)
    target_error_rates = [r["error_rate"] for r in results]
    actual_error_rates = [r["false_positive_rate"] for r in results]
    plt.plot(target_error_rates, target_error_rates, 'k--', label='Ідеальна відповідність')
    plt.plot(target_error_rates, actual_error_rates, 'bo-', label='Фактична частота')
    plt.xlabel('Цільова частота помилок')
    plt.ylabel('Фактична частота помилок')
    plt.title('Порівняння цільової та фактичної частоти помилкових спрацьовувань')
    plt.grid(True)
    plt.legend()
    
    # Графік 2: Розмір фільтра vs Частота помилок
    plt.subplot(2, 2, 2)
    filter_sizes = [r["filter_size_bits"]/8/1024 for r in results]  # в КБ
    plt.plot(target_error_rates, filter_sizes, 'ro-')
    plt.xlabel('Частота помилок')
    plt.ylabel('Розмір фільтра (КБ)')
    plt.title('Залежність розміру фільтра від частоти помилок')
    plt.grid(True)
    
    # Графік 3: Порівняння використання пам'яті
    plt.subplot(2, 2, 3)
    memory_bloom = [r["memory_bloom_kb"] for r in results]
    memory_set = [r["memory_set_kb"] for r in results]
    x = np.arange(len(error_rates))
    width = 0.35
    plt.bar(x - width/2, memory_bloom, width, label='Фільтр Блума')
    plt.bar(x + width/2, memory_set, width, label='Set')
    plt.xlabel('Частота помилок')
    plt.ylabel('Пам\'ять (КБ)')
    plt.title('Порівняння використання пам\'яті')
    plt.xticks(x, [f"{rate*100:.1f}%" for rate in error_rates])
    plt.legend()
    
    # Графік 4: Залежність часу від розміру фільтра
    plt.subplot(2, 2, 4)
    add_times = [r["add_time"] for r in results]
    check_times = [r["check_existing_time"] + r["check_non_existing_time"] for r in results]
    plt.plot(filter_sizes, add_times, 'go-', label='Додавання')
    plt.plot(filter_sizes, check_times, 'mo-', label='Перевірка')
    plt.xlabel('Розмір фільтра (КБ)')
    plt.ylabel('Час (секунди)')
    plt.title('Залежність часу виконання від розміру фільтра')
    plt.grid(True)
    plt.legend()
    
    save_plot(plt, "bloom_filter_analysis.png")
    plt.close()

def demo_hyperloglog(data):
    """Демонстрація алгоритму HyperLogLog."""
    print_header("Демонстрація алгоритму HyperLogLog")
    
    # Для демонстрації використаємо IP-адреси з веб-логів або користувачів з транзакцій
    if 'ip' in data[0]:
        elements = [item['ip'] for item in data]
        element_type = "IP-адрес"
    elif 'user_id' in data[0]:
        elements = [item['user_id'] for item in data]
        element_type = "користувачів"
    else:
        print("Непідтримуваний тип даних для демонстрації HyperLogLog")
        return
    
    # Знаходимо точну кількість унікальних елементів
    unique_count = len(set(elements))
    print(f"Точна кількість унікальних {element_type}: {unique_count}")
    
    # Тестуємо HyperLogLog з різними параметрами
    p_values = [4, 6, 8, 10, 12, 14, 16]
    
    results = []
    
    for p in p_values:
        print(f"\nСтворюємо HyperLogLog з параметром p={p}...")
        
        # Створюємо HyperLogLog
        hll = HyperLogLog(p)
        
        # Додаємо елементи
        start_time = time.time()
        for element in elements:
            hll.add(element)
        add_time = time.time() - start_time
        
        # Отримуємо оцінку кількості унікальних елементів
        estimated_count = hll.count()
        
        # Обчислюємо похибку
        error = abs(estimated_count - unique_count) / unique_count * 100
        
        # Обчислюємо теоретичну похибку
        theoretical_error = 1.04 / math.sqrt(2**p) * 100
        
        print(f"Кількість регістрів: {hll.m}")
        print(f"Точна кількість: {unique_count}, Оцінка: {estimated_count}")
        print(f"Відносна похибка: {error:.2f}%")
        print(f"Теоретична похибка: {theoretical_error:.2f}%")
        
        # Оцінка пам'яті (порівняно з set)
        hll_memory = hll.m  # в байтах (1 байт на регістр)
        set_memory = sys.getsizeof(set(elements))
        memory_ratio = hll_memory / set_memory
        
        print(f"Пам'ять HyperLogLog: {hll_memory/1024:.6f} КБ, Пам'ять set: {set_memory/1024:.2f} КБ")
        print(f"Відношення: {memory_ratio:.8f} (економія: {(1-memory_ratio)*100:.2f}%)")
        
        results.append({
            "p": p,
            "registers": hll.m,
            "true_count": unique_count,
            "estimated_count": estimated_count,
            "error_percent": error,
            "theoretical_error": theoretical_error,
            "memory_hll_kb": hll_memory/1024,
            "memory_set_kb": set_memory/1024,
            "memory_ratio": memory_ratio,
            "add_time": add_time
        })
    
    # Візуалізуємо результати
    plt.figure(figsize=(14, 10))
    
    # Графік 1: Точність оцінки
    plt.subplot(2, 2, 1)
    p_values_plot = [r["p"] for r in results]
    error_values = [r["error_percent"] for r in results]
    theoretical_errors = [r["theoretical_error"] for r in results]
    plt.plot(p_values_plot, error_values, 'bo-', label='Фактична похибка')
    plt.plot(p_values_plot, theoretical_errors, 'r--', label='Теоретична похибка')
    plt.xlabel('Параметр p')
    plt.ylabel('Відносна похибка (%)')
    plt.title('Залежність похибки від параметра p')
    plt.grid(True)
    plt.legend()
    
    # Графік 2: Оцінка vs Точна кількість
    plt.subplot(2, 2, 2)
    estimated_counts = [r["estimated_count"] for r in results]
    true_counts = [r["true_count"] for r in results]
    plt.plot(p_values_plot, [true_count for _ in range(len(p_values_plot))], 'k--', label='Точна кількість')
    plt.plot(p_values_plot, estimated_counts, 'go-', label='Оцінка HyperLogLog')
    plt.xlabel('Параметр p')
    plt.ylabel('Кількість унікальних елементів')
    plt.title('Порівняння оцінки HyperLogLog з точною кількістю')
    plt.grid(True)
    plt.legend()
    
    # Графік 3: Порівняння використання пам'яті
    plt.subplot(2, 2, 3)
    memory_hll = [r["memory_hll_kb"] for r in results]
    memory_set = [r["memory_set_kb"] for r in results]
    plt.bar(p_values_plot, memory_hll, label='HyperLogLog')
    plt.plot(p_values_plot, memory_set, 'r-', label='Set')
    plt.xlabel('Параметр p')
    plt.ylabel('Пам\'ять (КБ)')
    plt.title('Порівняння використання пам\'яті')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()
    
    # Графік 4: Компроміс між пам'яттю і точністю
    plt.subplot(2, 2, 4)
    memory_ratios = [r["memory_ratio"] * 100 for r in results]  # відсоток від пам'яті set
    plt.loglog(memory_ratios, error_values, 'mo-')
    for i, p in enumerate(p_values_plot):
        plt.annotate(f"p={p}", (memory_ratios[i], error_values[i]), 
                    textcoords="offset points", xytext=(0,10), ha='center')
    plt.xlabel('Пам\'ять (% від set)')
    plt.ylabel('Відносна похибка (%)')
    plt.title('Компроміс між використанням пам\'яті та точністю')
    plt.grid(True)
    
    save_plot(plt, "hyperloglog_analysis.png")
    plt.close()
    
    # Демонстрація поведінки при злитті
    print("\nДемонстрація злиття HyperLogLog...")
    
    # Розділяємо дані на дві частини
    middle = len(elements) // 2
    elements1 = elements[:middle]
    elements2 = elements[middle:]
    
    # Створюємо два HyperLogLog
    hll1 = HyperLogLog(12)
    hll2 = HyperLogLog(12)
    
    # Додаємо елементи
    for e in elements1:
        hll1.add(e)
    
    for e in elements2:
        hll2.add(e)
    
    # Виконуємо злиття
    hll_merged = hll1.merge(hll2)
    
    # Порівнюємо результати
    exact_count1 = len(set(elements1))
    exact_count2 = len(set(elements2))
    exact_union = len(set(elements1) | set(elements2))
    
    estimate1 = hll1.count()
    estimate2 = hll2.count()
    estimate_merged = hll_merged.count()
    
    print(f"Точна кількість у першій частині: {exact_count1}, Оцінка: {estimate1}")
    print(f"Точна кількість у другій частині: {exact_count2}, Оцінка: {estimate2}")
    print(f"Точна кількість в об'єднанні: {exact_union}, Оцінка злиття: {estimate_merged}")
    
    # Обчислюємо похибки
    error1 = abs(estimate1 - exact_count1) / exact_count1 * 100
    error2 = abs(estimate2 - exact_count2) / exact_count2 * 100
    error_merged = abs(estimate_merged - exact_union) / exact_union * 100
    
    print(f"Похибка для першої частини: {error1:.2f}%")
    print(f"Похибка для другої частини: {error2:.2f}%")
    print(f"Похибка для злиття: {error_merged:.2f}%")

def demo_count_min_sketch(data):
    """Демонстрація алгоритму Count-Min Sketch."""
    print_header("Демонстрація алгоритму Count-Min Sketch")
    
    # Для демонстрації використаємо шляхи з веб-логів або товари з транзакцій
    if 'path' in data[0]:
        elements = [item['path'] for item in data]
        element_type = "шляхів"
    elif 'product_id' in data[0]:
        elements = [item['product_id'] for item in data]
        element_type = "товарів"
    else:
        print("Непідтримуваний тип даних для демонстрації Count-Min Sketch")
        return
    
    # Підраховуємо точні частоти
    exact_counts = Counter(elements)
    print(f"Унікальних {element_type}: {len(exact_counts)}")
    
    # Показуємо топ-10 найчастіших елементів
    print(f"\nТоп-10 найчастіших {element_type}:")
    for element, count in exact_counts.most_common(10):
        print(f"{element}: {count}")
    
    # Тестуємо Count-Min Sketch з різними параметрами
    epsilon_values = [0.1, 0.05, 0.01, 0.005]
    delta_values = [0.1, 0.05, 0.01, 0.005]
    
    results = []
    
    for epsilon in epsilon_values:
        for delta in delta_values:
            print(f"\nСтворюємо Count-Min Sketch з параметрами epsilon={epsilon}, delta={delta}...")
            
            # Створюємо Count-Min Sketch
            cms = CountMinSketch(epsilon, delta)
            
            # Додаємо елементи
            start_time = time.time()
            for element in elements:
                cms.add(element)
            add_time = time.time() - start_time
            
            # Порівнюємо оцінки з точними значеннями
            element_samples = list(exact_counts.keys())
            if len(element_samples) > 100:
                element_samples = random.sample(element_samples, 100)
            
            errors = []
            relative_errors = []
            
            for element in element_samples:
                exact_count = exact_counts[element]
                estimated_count = cms.estimate_count(element)
                
                error = estimated_count - exact_count
                relative_error = error / exact_count if exact_count > 0 else 0
                
                errors.append(error)
                relative_errors.append(relative_error)
            
            # Обчислюємо середню та максимальну похибку
            avg_error = sum(errors) / len(errors)
            max_error = max(errors)
            avg_relative_error = sum(relative_errors) / len(relative_errors)
            max_relative_error = max(relative_errors)
            
            # Обчислюємо теоретичну похибку
            theoretical_error = epsilon * sum(exact_counts.values())
            
            print(f"Ширина таблиці (w): {cms.w}, Глибина (d): {cms.d}")
            print(f"Загальна кількість лічильників: {cms.w * cms.d}")
            print(f"Середня похибка: {avg_error:.2f}")
            print(f"Максимальна похибка: {max_error}")
            print(f"Середня відносна похибка: {avg_relative_error:.2%}")
            print(f"Теоретична максимальна похибка: {theoretical_error:.2f}")
            
            # Оцінка пам'яті (порівняно з Counter)
            cms_memory = cms.w * cms.d * 4  # 4 байти на лічильник (int)
            counter_memory = sys.getsizeof(exact_counts)
            memory_ratio = cms_memory / counter_memory
            
            print(f"Пам'ять Count-Min Sketch: {cms_memory/1024:.2f} КБ, Пам'ять Counter: {counter_memory/1024:.2f} КБ")
            print(f"Відношення: {memory_ratio:.4f} (економія: {(1-memory_ratio)*100:.2f}%)")
            
            results.append({
                "epsilon": epsilon,
                "delta": delta,
                "width": cms.w,
                "depth": cms.d,
                "avg_error": avg_error,
                "max_error": max_error,
                "avg_relative_error": avg_relative_error,
                "max_relative_error": max_relative_error,
                "theoretical_error": theoretical_error,
                "memory_cms_kb": cms_memory/1024,
                "memory_counter_kb": counter_memory/1024,
                "memory_ratio": memory_ratio,
                "add_time": add_time
            })
    
    # Візуалізуємо результати
    plt.figure(figsize=(16, 10))
    
    # Графік 1: Фактична vs Теоретична похибка
    plt.subplot(2, 2, 1)
    x = [r["theoretical_error"] for r in results]
    y = [r["max_error"] for r in results]
    plt.scatter(x, y)
    plt.plot([0, max(x)], [0, max(x)], 'k--', label='y=x')
    for i, r in enumerate(results):
        plt.annotate(f"({r['epsilon']}, {r['delta']})", (x[i], y[i]), 
                   textcoords="offset points", xytext=(0,5), ha='center', fontsize=8)
    plt.xlabel('Теоретична похибка')
    plt.ylabel('Максимальна фактична похибка')
    plt.title('Порівняння теоретичної та фактичної похибки')
    plt.grid(True)
    plt.legend()
    
    # Графік 2: Вплив параметрів на похибку
    plt.subplot(2, 2, 2)
    epsilon_groups = {}
    for r in results:
        eps = r["epsilon"]
        if eps not in epsilon_groups:
            epsilon_groups[eps] = []
        epsilon_groups[eps].append(r)
    
    for eps, group in epsilon_groups.items():
        delta_values = [r["delta"] for r in group]
        error_values = [r["avg_relative_error"] for r in group]
        plt.plot(delta_values, error_values, 'o-', label=f'epsilon={eps}')
    
    plt.xlabel('delta')
    plt.ylabel('Середня відносна похибка')
    plt.title('Вплив параметрів на похибку')
    plt.xscale('log')
    plt.grid(True)
    plt.legend()
    
    # Графік 3: Порівняння використання пам'яті
    plt.subplot(2, 2, 3)
    memory_cms = [r["memory_cms_kb"] for r in results]
    memory_counter = [r["memory_counter_kb"] for r in results]
    indices = range(len(results))
    width = 0.35
    plt.bar(indices, memory_cms, width, label='Count-Min Sketch')
    plt.bar([i + width for i in indices], memory_counter, width, label='Counter')
    plt.xlabel('Комбінація параметрів')
    plt.ylabel('Пам\'ять (КБ)')
    plt.title('Порівняння використання пам\'яті')
    plt.xticks([i + width/2 for i in indices], 
              [f"({r['epsilon']}, {r['delta']})" for r in results], rotation=45, ha='right', fontsize=8)
    plt.legend()
    
    # Графік 4: Компроміс між пам'яттю і точністю
    plt.subplot(2, 2, 4)
    memory = [r["memory_cms_kb"] for r in results]
    error = [r["avg_relative_error"] for r in results]
    plt.scatter(memory, error)
    for i, r in enumerate(results):
        plt.annotate(f"({r['epsilon']}, {r['delta']})", (memory[i], error[i]), 
                   textcoords="offset points", xytext=(0,5), ha='center', fontsize=8)
    plt.xlabel('Пам\'ять (КБ)')
    plt.ylabel('Середня відносна похибка')
    plt.title('Компроміс між використанням пам\'яті та точністю')
    plt.grid(True)
    
    save_plot(plt, "count_min_sketch_analysis.png")
    plt.close()
    
    # Демонстрація виявлення важких елементів (heavy hitters)
    print("\nДемонстрація виявлення важких елементів (heavy hitters)...")
    
    # Створюємо Count-Min Sketch з оптимальними параметрами
    cms = CountMinSketch(0.01, 0.01)
    
    # Додаємо елементи
    for element in elements:
        cms.add(element)
    
    # Знаходимо топ-10 найчастіших елементів за допомогою Count-Min Sketch
    # (Для цього нам довелося б зберігати всі елементи, що суперечить ідеї економії пам'яті,
    # тому це лише для демонстрації)
    unique_elements = set(elements)
    cms_counts = [(element, cms.estimate_count(element)) for element in unique_elements]
    cms_top10 = sorted(cms_counts, key=lambda x: x[1], reverse=True)[:10]
    
    # Порівнюємо з точним топ-10
    exact_top10 = exact_counts.most_common(10)
    
    print("\nТоп-10 за Count-Min Sketch vs Точний топ-10:")
    print(f"{'Елемент':20} {'Оцінка CMS':10} {'Точна кількість':15} {'Різниця':10}")
    print("-" * 60)
    
    for i in range(10):
        cms_element, cms_count = cms_top10[i]
        exact_element, exact_count = exact_top10[i]
        
        # Шукаємо точну кількість для елемента з CMS
        exact_for_cms = exact_counts[cms_element]
        
        print(f"{cms_element:20} {cms_count:10} {exact_for_cms:15} {cms_count - exact_for_cms:+10}")
    
    # Обчислюємо точність ранжування
    correct_order = 0
    for i, (cms_element, _) in enumerate(cms_top10):
        exact_rank = next((j for j, (e, _) in enumerate(exact_top10) if e == cms_element), None)
        if exact_rank is not None and exact_rank == i:
            correct_order += 1
    
    print(f"\nТочність ранжування топ-10: {correct_order / 10:.1%}")
    
    # Перевіряємо, чи всі елементи з точного топ-10 є в топ-10 CMS
    cms_elements = {element for element, _ in cms_top10}
    exact_elements = {element for element, _ in exact_top10}
    common_elements = cms_elements.intersection(exact_elements)
    
    print(f"Спільних елементів у топ-10: {len(common_elements)} з 10")

def demo_misra_gries(data):
    """Демонстрація алгоритму Misra-Gries."""
    print_header("Демонстрація алгоритму Misra-Gries")
    
    # Для демонстрації використаємо шляхи з веб-логів або товари з транзакцій
    if 'path' in data[0]:
        elements = [item['path'] for item in data]
        element_type = "шляхів"
    elif 'product_id' in data[0]:
        elements = [item['product_id'] for item in data]
        element_type = "товарів"
    else:
        print("Непідтримуваний тип даних для демонстрації Misra-Gries")
        return
    
    # Підраховуємо точні частоти
    exact_counts = Counter(elements)
    print(f"Унікальних {element_type}: {len(exact_counts)}")
    
    # Показуємо топ-10 найчастіших елементів
    print(f"\nТоп-10 найчастіших {element_type}:")
    for element, count in exact_counts.most_common(10):
        print(f"{element}: {count}")
    
    # Тестуємо Misra-Gries з різними параметрами k
    k_values = [5, 10, 20, 50, 100]
    
    results_mg = []
    results_ss = []
    
    for k in k_values:
        print(f"\nТестуємо Misra-Gries з k={k}...")
        
        # Створюємо Misra-Gries
        mg = MisraGries(k)
        
        # Додаємо елементи
        start_time = time.time()
        mg.process_batch(elements)
        mg_time = time.time() - start_time
        
        # Аналізуємо результати
        mg_counters = mg.counters
        threshold = len(elements) / k  # Теоретичний поріг для важких елементів
        
        # Знаходимо важкі елементи за допомогою Misra-Gries
        mg_heavy = mg.get_top_k()
        
        # Знаходимо точні важкі елементи
        exact_heavy = [(element, count) for element, count in exact_counts.items() if count > threshold]
        
        # Обчислюємо точність і повноту
        mg_elements = {element for element, _ in mg_heavy}
        exact_heavy_elements = {element for element, _ in exact_heavy}
        
        common_elements = mg_elements.intersection(exact_heavy_elements)
        precision = len(common_elements) / len(mg_elements) if mg_elements else 0
        recall = len(common_elements) / len(exact_heavy_elements) if exact_heavy_elements else 1
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        
        print(f"Кількість лічильників: {len(mg_counters)}")
        print(f"Теоретичний поріг для важких елементів: {threshold:.2f}")
        print(f"Знайдено важких елементів (Misra-Gries): {len(mg_heavy)}")
        print(f"Точних важких елементів: {len(exact_heavy)}")
        print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}")
        
        # Оцінка пам'яті
        mg_memory = sys.getsizeof(mg_counters)
        counter_memory = sys.getsizeof(exact_counts)
        memory_ratio = mg_memory / counter_memory
        
        print(f"Пам'ять Misra-Gries: {mg_memory/1024:.2f} КБ, Пам'ять Counter: {counter_memory/1024:.2f} КБ")
        print(f"Відношення: {memory_ratio:.4f} (економія: {(1-memory_ratio)*100:.2f}%)")
        
        results_mg.append({
            "k": k,
            "threshold": threshold,
            "time": mg_time,
            "counters": len(mg_counters),
            "exact_heavy": len(exact_heavy),
            "mg_heavy": len(mg_heavy),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "memory_mg_kb": mg_memory/1024,
            "memory_counter_kb": counter_memory/1024,
            "memory_ratio": memory_ratio
        })
        
        # Тепер тестуємо Space-Saving
        print(f"\nТестуємо Space-Saving з k={k}...")
        
        # Створюємо Space-Saving
        ss = SpaceSaving(k)
        
        # Додаємо елементи
        start_time = time.time()
        for element in elements:
            ss.add(element)
        ss_time = time.time() - start_time
        
        # Знаходимо важкі елементи за допомогою Space-Saving
        ss_heavy = ss.get_top_k()
        
        # Обчислюємо точність і повноту
        ss_elements = {element for element, _, _ in ss_heavy}
        
        common_elements = ss_elements.intersection(exact_heavy_elements)
        precision = len(common_elements) / len(ss_elements) if ss_elements else 0
        recall = len(common_elements) / len(exact_heavy_elements) if exact_heavy_elements else 1
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        
        print(f"Кількість лічильників: {len(ss.counters)}")
        print(f"Знайдено важких елементів (Space-Saving): {len(ss_heavy)}")
        print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}")
        
        # Оцінка пам'яті
        ss_memory = sys.getsizeof(ss.counters) + sys.getsizeof(ss.errors)
        memory_ratio = ss_memory / counter_memory
        
        print(f"Пам'ять Space-Saving: {ss_memory/1024:.2f} КБ, Пам'ять Counter: {counter_memory/1024:.2f} КБ")
        print(f"Відношення: {memory_ratio:.4f} (економія: {(1-memory_ratio)*100:.2f}%)")
        
        results_ss.append({
            "k": k,
            "threshold": threshold,
            "time": ss_time,
            "counters": len(ss.counters),
            "exact_heavy": len(exact_heavy),
            "ss_heavy": len(ss_heavy),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "memory_ss_kb": ss_memory/1024,
            "memory_counter_kb": counter_memory/1024,
            "memory_ratio": memory_ratio
        })
    
    # Візуалізуємо результати
    plt.figure(figsize=(16, 12))
    
    # Графік 1: Точність і повнота для Misra-Gries
    plt.subplot(2, 2, 1)
    k_values_plot = [r["k"] for r in results_mg]
    precisions_mg = [r["precision"] for r in results_mg]
    recalls_mg = [r["recall"] for r in results_mg]
    f1_scores_mg = [r["f1"] for r in results_mg]
    
    plt.plot(k_values_plot, precisions_mg, 'bo-', label='Precision (MG)')
    plt.plot(k_values_plot, recalls_mg, 'ro-', label='Recall (MG)')
    plt.plot(k_values_plot, f1_scores_mg, 'go-', label='F1 (MG)')
    plt.xlabel('Параметр k')
    plt.ylabel('Значення метрики')
    plt.title('Точність і повнота для Misra-Gries')
    plt.grid(True)
    plt.legend()
    
    # Графік 2: Порівняння F1 для Misra-Gries і Space-Saving
    plt.subplot(2, 2, 2)
    f1_scores_ss = [r["f1"] for r in results_ss]
    
    plt.plot(k_values_plot, f1_scores_mg, 'go-', label='F1 (Misra-Gries)')
    plt.plot(k_values_plot, f1_scores_ss, 'mo-', label='F1 (Space-Saving)')
    plt.xlabel('Параметр k')
    plt.ylabel('F1-міра')
    plt.title('Порівняння F1-міри для Misra-Gries і Space-Saving')
    plt.grid(True)
    plt.legend()
    
    # Графік 3: Порівняння використання пам'яті
    plt.subplot(2, 2, 3)
    memory_mg = [r["memory_mg_kb"] for r in results_mg]
    memory_ss = [r["memory_ss_kb"] for r in results_ss]
    memory_counter = [r["memory_counter_kb"] for r in results_mg]
    
    plt.bar([i - 0.2 for i in range(len(k_values))], memory_mg, width=0.2, label='Misra-Gries')
    plt.bar(range(len(k_values)), memory_ss, width=0.2, label='Space-Saving')
    plt.bar([i + 0.2 for i in range(len(k_values))], memory_counter, width=0.2, label='Counter')
    plt.xlabel('Параметр k')
    plt.ylabel('Пам\'ять (КБ)')
    plt.title('Порівняння використання пам\'яті')
    plt.xticks(range(len(k_values)), k_values)
    plt.legend()
    
    # Графік 4: Час виконання
    plt.subplot(2, 2, 4)
    time_mg = [r["time"] for r in results_mg]
    time_ss = [r["time"] for r in results_ss]
    
    plt.plot(k_values_plot, time_mg, 'bo-', label='Misra-Gries')
    plt.plot(k_values_plot, time_ss, 'ro-', label='Space-Saving')
    plt.xlabel('Параметр k')
    plt.ylabel('Час виконання (секунди)')
    plt.title('Порівняння часу виконання')
    plt.grid(True)
    plt.legend()
    
    save_plot(plt, "misra_gries_analysis.png")
    plt.close()
    
    # Порівняння точності оцінки частоти для конкретних елементів
    print("\nПорівняння точності оцінки частоти для конкретних елементів...")
    
    # Обираємо елементи з різною частотою зустрічання
    exact_freqs = exact_counts.most_common()
    test_elements = [
        exact_freqs[0][0],  # найчастіший
        exact_freqs[len(exact_freqs)//10][0],  # 10-й перцентиль
        exact_freqs[len(exact_freqs)//4][0],  # 25-й перцентиль
        exact_freqs[len(exact_freqs)//2][0],  # 50-й перцентиль
        exact_freqs[3*len(exact_freqs)//4][0]  # 75-й перцентиль
    ]
    
    print("\nТочність оцінки частоти для різних елементів:")
    print(f"{'Елемент':20} {'Справжня':10} {'Misra-Gries':12} {'Space-Saving':12} {'MG Похибка':10} {'SS Похибка':10}")
    print("-" * 80)
    
    # Створюємо Misra-Gries і Space-Saving з k=50
    mg = MisraGries(50)
    ss = SpaceSaving(50)
    
    # Додаємо елементи
    mg.process_batch(elements)
    for element in elements:
        ss.add(element)
    
    for element in test_elements:
        exact_count = exact_counts[element]
        mg_count = mg.counters.get(element, 0)
        ss_count, ss_error = ss.get_count(element)
        
        mg_error = (mg_count - exact_count) / exact_count * 100 if exact_count > 0 else 0
        ss_error_pct = (ss_count - exact_count) / exact_count * 100 if exact_count > 0 else 0
        
        print(f"{element:20} {exact_count:10} {mg_count:12} {ss_count:12} {mg_error:+.2f}% {ss_error_pct:+.2f}%")

def demo_lsh(data, second_data=None):
    """Демонстрація алгоритму Locality-Sensitive Hashing (LSH)."""
    print_header("Демонстрація алгоритму Locality-Sensitive Hashing (LSH)")
    
    # Для демонстрації створимо профілі користувачів на основі їхніх шляхів
    if 'path' in data[0] and 'ip' in data[0]:
        print("Створення профілів користувачів на основі веб-логів...")
        
        # Групуємо логи за IP-адресами
        user_profiles = defaultdict(set)
        for log in data:
            user_profiles[log['ip']].add(log['path'])
        
        print(f"Створено {len(user_profiles)} профілів користувачів")
        
        # Показуємо кілька прикладів
        print("\nПриклади профілів користувачів:")
        counter = 0
        for ip, paths in user_profiles.items():
            print(f"Користувач {ip}: {len(paths)} унікальних шляхів")
            if len(paths) > 0:
                print(f"Приклади шляхів: {list(paths)[:5]}")
            counter += 1
            if counter >= 3:
                break
        
        # Демонстрація MinHash
        print("\nДемонстрація MinHash для обчислення подібності Жаккара...")
        
        # Створюємо MinHash
        minhash = MinHash(num_hashes=100)
        
        # Обираємо кілька профілів для порівняння
        profile_items = list(user_profiles.items())
        if len(profile_items) > 5:
            profiles_to_compare = random.sample(profile_items, 5)
        else:
            profiles_to_compare = profile_items
        
        # Обчислюємо сигнатури MinHash
        signatures = {}
        for ip, paths in profiles_to_compare:
            signatures[ip] = minhash.signature(paths)
        
        # Порівнюємо подібність
        print("\nПодібність профілів (Jaccard vs MinHash):")
        print(f"{'IP1':15} {'IP2':15} {'Jaccard':10} {'MinHash':10} {'Різниця':10}")
        print("-" * 65)
        
        for i in range(len(profiles_to_compare)):
            ip1, paths1 = profiles_to_compare[i]
            for j in range(i+1, len(profiles_to_compare)):
                ip2, paths2 = profiles_to_compare[j]
                
                # Обчислюємо точну подібність Жаккара
                jaccard = len(paths1.intersection(paths2)) / len(paths1.union(paths2)) if paths1 or paths2 else 0
                
                # Обчислюємо подібність за MinHash
                minhash_sim = minhash.similarity(signatures[ip1], signatures[ip2])
                
                diff = minhash_sim - jaccard
                
                print(f"{ip1:15} {ip2:15} {jaccard:.8f} {minhash_sim:.8f} {diff:+.8f}")
        
        # Демонстрація LSH
        print("\nДемонстрація LSH для пошуку схожих профілів...")
        
        # Створюємо LSH з різними параметрами
        band_values = [5, 10, 20, 50]
        row_values = [2, 5, 10, 20]
        
        results = []
        
        for bands in band_values:
            for rows in row_values:
                lsh_params = (bands, rows)
                
                # Пропускаємо невалідні комбінації
                if bands * rows > 100:
                    continue
                
                print(f"\nТестуємо LSH з {bands} смугами та {rows} рядками на смугу...")
                
                # Створюємо LSH
                lsh = LSH(num_bands=bands, num_rows=rows)
                
                # Додаємо профілі
                start_time = time.time()
                for ip, paths in user_profiles.items():
                    lsh.add(paths, item_id=ip)
                add_time = time.time() - start_time
                
                # Вибираємо довільний профіль для пошуку схожих
                query_ip, query_paths = random.choice(list(user_profiles.items()))
                
                # Виконуємо пошук схожих профілів
                start_time = time.time()
                similar = lsh.query(query_paths, threshold=0.3)
                query_time = time.time() - start_time
                
                # Обчислюємо точні подібності для валідації
                exact_similarities = []
                for ip, paths in user_profiles.items():
                    if ip != query_ip:
                        jaccard = len(query_paths.intersection(paths)) / len(query_paths.union(paths))
                        if jaccard >= 0.3:
                            exact_similarities.append((ip, jaccard))
                
                exact_similarities.sort(key=lambda x: x[1], reverse=True)
                
                # Порівнюємо результати
                lsh_ips = {ip for ip, _ in similar}
                exact_ips = {ip for ip, _ in exact_similarities}
                
                precision = len(lsh_ips.intersection(exact_ips)) / len(lsh_ips) if lsh_ips else 0
                recall = len(lsh_ips.intersection(exact_ips)) / len(exact_ips) if exact_ips else 1
                f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
                
                print(f"Запит для профілю {query_ip} з {len(query_paths)} шляхами")
                print(f"Знайдено схожих профілів (LSH): {len(similar)}")
                print(f"Точних схожих профілів: {len(exact_similarities)}")
                print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}")
                print(f"Час додавання: {add_time:.6f} секунд, Час запиту: {query_time:.6f} секунд")
                
                results.append({
                    "bands": bands,
                    "rows": rows,
                    "hash_functions": bands * rows,
                    "add_time": add_time,
                    "query_time": query_time,
                    "lsh_results": len(similar),
                    "exact_results": len(exact_similarities),
                    "precision": precision,
                    "recall": recall,
                    "f1": f1
                })
        
        # Візуалізуємо результати
        plt.figure(figsize=(16, 12))
        
        # Графік 1: Precision vs Recall для різних параметрів
        plt.subplot(2, 2, 1)
        precisions = [r["precision"] for r in results]
        recalls = [r["recall"] for r in results]
        f1_scores = [r["f1"] for r in results]
        labels = [f"B={r['bands']},R={r['rows']}" for r in results]
        
        plt.scatter(recalls, precisions, s=100)
        for i, label in enumerate(labels):
            plt.annotate(label, (recalls[i], precisions[i]), 
                       textcoords="offset points", xytext=(0,5), ha='center')
        
        # Лінії постійного F1
        f1_contours = [0.2, 0.4, 0.6, 0.8]
        x = np.linspace(0.01, 1, 100)
        for f1_val in f1_contours:
            y = f1_val * x / (2*x - f1_val)
            valid_indices = (y > 0) & (y <= 1)
            plt.plot(x[valid_indices], y[valid_indices], 'k--', alpha=0.3)
            plt.annotate(f"F1={f1_val}", (x[50], y[50]), alpha=0.5)
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision vs Recall для різних параметрів LSH')
        plt.grid(True)
        
        # Графік 2: Вплив кількості хеш-функцій на F1
        plt.subplot(2, 2, 2)
        hash_functions = [r["hash_functions"] for r in results]
        plt.plot(hash_functions, f1_scores, 'bo-')
        for i, label in enumerate(labels):
            plt.annotate(label, (hash_functions[i], f1_scores[i]), 
                       textcoords="offset points", xytext=(0,5), ha='center')
        plt.xlabel('Кількість хеш-функцій')
        plt.ylabel('F1-міра')
        plt.title('Вплив кількості хеш-функцій на F1-міру')
        plt.grid(True)
        
        # Графік 3: Час виконання
        plt.subplot(2, 2, 3)
        add_times = [r["add_time"] for r in results]
        query_times = [r["query_time"] for r in results]
        width = 0.35
        x = range(len(results))
        plt.bar([i - width/2 for i in x], add_times, width, label='Час додавання')
        plt.bar([i + width/2 for i in x], query_times, width, label='Час запиту')
        plt.xlabel('Параметри')
        plt.ylabel('Час (секунди)')
        plt.title('Час виконання для різних параметрів LSH')
        plt.xticks(x, labels, rotation=45, ha='right')
        plt.legend()
        
        # Графік 4: Компроміс між часом і точністю
        plt.subplot(2, 2, 4)
        query_times = [r["query_time"] for r in results]
        plt.scatter(query_times, f1_scores, s=100)
        for i, label in enumerate(labels):
            plt.annotate(label, (query_times[i], f1_scores[i]), 
                       textcoords="offset points", xytext=(0,5), ha='center')
        plt.xlabel('Час запиту (секунди)')
        plt.ylabel('F1-міра')
        plt.title('Компроміс між часом запиту і точністю')
        plt.grid(True)
        
        save_plot(plt, "lsh_analysis.png")
        plt.close()
        
    # Демонстрація VectorLSH
    print("\nДемонстрація VectorLSH для пошуку схожих векторів...")
    
    # Створюємо векторні представлення для демонстрації
    # Для простоти, ми створимо вектори TF-IDF для профілів користувачів
    if 'path' in data[0] and 'ip' in data[0]:
        print("Створення векторних представлень профілів користувачів...")
        
        # Збираємо всі унікальні шляхи для словника
        all_paths = set()
        for paths in user_profiles.values():
            all_paths.update(paths)
        
        path_to_index = {path: i for i, path in enumerate(all_paths)}
        
        # Створюємо векторні представлення
        user_vectors = {}
        for ip, paths in user_profiles.items():
            # Створюємо вектор типу "мішок слів"
            vector = np.zeros(len(path_to_index))
            for path in paths:
                vector[path_to_index[path]] = 1
            
            # Нормалізуємо вектор
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
            
            user_vectors[ip] = vector
        
        print(f"Створено {len(user_vectors)} векторних представлень розмірності {len(path_to_index)}")
        
        # Тестуємо VectorLSH
        print("\nТестуємо VectorLSH для пошуку схожих векторів...")
        
        # Створюємо VectorLSH
        vector_lsh = VectorLSH(dim=len(path_to_index), num_tables=10, hash_size=16)
        
        # Додаємо вектори
        start_time = time.time()
        for ip, vector in user_vectors.items():
            vector_lsh.add(vector, item_id=ip)
        add_time = time.time() - start_time
        
        # Вибираємо довільний вектор для пошуку схожих
        query_ip = random.choice(list(user_vectors.keys()))
        query_vector = user_vectors[query_ip]
        
        # Виконуємо пошук схожих векторів
        start_time = time.time()
        similar_vectors = vector_lsh.query(query_vector, k=10)
        query_time = time.time() - start_time
        
        # Обчислюємо точні подібності для валідації
        exact_similarities = []
        for ip, vector in user_vectors.items():
            if ip != query_ip:
                # Косинусна подібність
                similarity = np.dot(query_vector, vector)
                exact_similarities.append((ip, similarity))
        
        exact_similarities.sort(key=lambda x: x[1], reverse=True)
        exact_top10 = exact_similarities[:10]
        
        # Порівнюємо результати
        lsh_ips = {ip for ip, _ in similar_vectors}
        exact_ips = {ip for ip, _ in exact_top10}
        
        common = lsh_ips.intersection(exact_ips)
        precision_at_10 = len(common) / len(lsh_ips) if lsh_ips else 0
        
        print(f"Запит для профілю {query_ip}")
        print(f"Знайдено схожих профілів (VectorLSH): {len(similar_vectors)}")
        print(f"Спільних результатів у топ-10: {len(common)}")
        print(f"Precision@10: {precision_at_10:.2f}")
        print(f"Час додавання: {add_time:.6f} секунд, Час запиту: {query_time:.6f} секунд")
        
        # Порівняємо з повним перебором
        start_time = time.time()
        brute_force_time = time.time() - start_time
        
        print(f"Час повного перебору: {brute_force_time:.6f} секунд")
        print(f"Прискорення: {brute_force_time/query_time:.2f}x")
        
        # Виводимо результати для порівняння
        print("\nРезультати VectorLSH vs Повний перебір:")
        print(f"{'#':2} {'LSH IP':15} {'LSH Подібність':15} {'Точна IP':15} {'Точна Подібність':15}")
        print("-" * 70)
        
        for i in range(min(10, len(similar_vectors), len(exact_top10))):
            lsh_ip, lsh_sim = similar_vectors[i]
            exact_ip, exact_sim = exact_top10[i]
            print(f"{i+1:2} {lsh_ip:15} {lsh_sim:.8f} {exact_ip:15} {exact_sim:.8f}")

def demo_all():
    """Запускає повну демонстрацію всіх алгоритмів."""
    print_header("Практична демонстрація алгоритмів для роботи з великими даними")
    
    # Генеруємо дані
    web_logs, transactions = generate_data()
    
    # Демонстрація резервуарної вибірки
    demo_reservoir_sampling(web_logs)
    
    # Демонстрація фільтра Блума
    demo_bloom_filter(web_logs)
    
    # Демонстрація HyperLogLog
    demo_hyperloglog(web_logs)
    
    # Демонстрація Count-Min Sketch
    demo_count_min_sketch(web_logs)
    
    # Демонстрація Misra-Gries
    demo_misra_gries(web_logs)
    
    # Демонстрація LSH
    demo_lsh(web_logs)
    
    print("\nУсі демонстрації завершено. Результати збережено в директорії:", DEMO_DIR)

if __name__ == "__main__":
    demo_all()
