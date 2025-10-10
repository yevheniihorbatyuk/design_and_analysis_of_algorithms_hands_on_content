from collections import defaultdict

# Функція map
def map_function(data):
    result = []
    for line in data:
        for word in line.split():
            result.append((word, 1))
    return result

# Функція shuffle (групування)
def shuffle(mapped_data):
    grouped_data = defaultdict(list)
    for key, value in mapped_data:
        grouped_data[key].append(value)
    return grouped_data

# Функція reduce
def reduce_function(grouped_data):
    reduced_data = {}
    for key, values in grouped_data.items():
        reduced_data[key] = sum(values)
    return reduced_data

if __name__ == "__main__":
    # Вхідні дані
    data = [
        "hello world",
        "hello mapreduce"
    ]

    # Виконання MapReduce
    mapped_data = map_function(data)
    grouped_data = shuffle(mapped_data)
    result = reduce_function(grouped_data)

    print(result)  # {'hello': 2, 'world': 1, 'mapreduce': 1}