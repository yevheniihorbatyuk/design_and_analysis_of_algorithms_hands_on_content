# src/utils/data_generator.py

import random
import uuid
import time
from typing import Iterator, Dict, Any, List, Set

def generate_social_media_stream(n: int) -> Iterator[Dict[str, Any]]:
    """Генерує потік подій з соціальної мережі."""
    users = [f"user_{i}" for i in range(1000)]
    hashtags = [f"#topic{i}" for i in range(50)]
    
    # Zipf-розподіл для хештегів
    hashtag_weights = [1.0 / (i**0.8) for i in range(1, 51)]

    for _ in range(n):
        event = {
            "user_id": random.choice(users),
            "timestamp": time.time(),
            "action": random.choice(["like", "post", "comment"]),
            "hashtags": random.choices(hashtags, weights=hashtag_weights, k=random.randint(1, 3))
        }
        yield event

def generate_documents(n: int, vocab_size: int, doc_size: int) -> Dict[int, Set[str]]:
    """Генерує документи як набори слів для LSH."""
    vocab = [f"word_{i}" for i in range(vocab_size)]
    documents = {}
    for i in range(n):
        documents[i] = set(random.sample(vocab, doc_size))
    # Створимо кілька майже дублікатів
    documents[n] = documents[0].copy()
    documents[n].remove(random.choice(list(documents[n])))
    documents[n].add(f"word_{vocab_size+1}")
    return documents