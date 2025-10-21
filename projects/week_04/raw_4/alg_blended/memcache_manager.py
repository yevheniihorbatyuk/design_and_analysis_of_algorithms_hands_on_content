import telnetlib
import sys

MEMCACHED_HOST = "localhost"  # Замініть на IP або hostname вашого Memcached сервера
MEMCACHED_PORT = 11211

def clear_memcache(host=MEMCACHED_HOST, port=MEMCACHED_PORT):
    """Очищає весь кеш Memcached."""
    try:
        tn = telnetlib.Telnet(host, port)
        tn.write(b"flush_all\r\n")
        response = tn.read_until(b"OK\r\n", timeout=1)
        if b"OK" in response:
            print(f"Успішно очищено весь кеш на {host}:{port}")
        else:
            print(f"Помилка при очищенні кешу на {host}:{port}: {response.decode().strip()}")
        tn.close()
    except ConnectionRefusedError:
        print(f"Помилка підключення до Memcached на {host}:{port}. Перевірте, чи запущено сервер.")
        sys.exit(1)
    except Exception as e:
        print(f"Сталася помилка: {e}")
        sys.exit(1)

def show_cache_stats(host=MEMCACHED_HOST, port=MEMCACHED_PORT):
    """Показує загальну статистику кешу Memcached."""
    try:
        tn = telnetlib.Telnet(host, port)
        tn.write(b"stats\r\n")
        response = tn.read_until(b"END\r\n", timeout=5)
        print(f"Статистика кешу на {host}:{port}:\n{response.decode()}")
        tn.close()
    except ConnectionRefusedError:
        print(f"Помилка підключення до Memcached на {host}:{port}. Перевірте, чи запущено сервер.")
        sys.exit(1)
    except Exception as e:
        print(f"Сталася помилка: {e}")
        sys.exit(1)

def show_cached_keys(host=MEMCACHED_HOST, port=MEMCACHED_PORT):
    """Намагається отримати список ключів у кеші (залежить від конфігурації Memcached).
    Примітка: Отримання всіх ключів може бути ресурсомістким на великих кешах
             і може бути вимкнено в конфігурації Memcached."""
    try:
        tn = telnetlib.Telnet(host, port)
        tn.write(b"stats items\r\n")
        items_response = tn.read_until(b"END\r\n", timeout=5).decode()

        slab_ids = []
        for line in items_response.splitlines():
            if line.startswith("ITEM"):
                parts = line.split()
                if len(parts) > 2:
                    slab_id = parts[1].split(":")[1]
                    if slab_id not in slab_ids:
                        slab_ids.append(slab_id)

        if not slab_ids:
            print(f"Не знайдено активних елементів у кеші на {host}:{port}.")
            tn.close()
            return

        print(f"Ключі в кеші на {host}:{port}:")
        for slab_id in slab_ids:
            tn.write(f"stats cachedump {slab_id} 100\r\n".encode()) # Отримуємо до 100 ключів з кожної slab
            dump_response = tn.read_until(b"END\r\n", timeout=5).decode()
            for line in dump_response.splitlines():
                if line.startswith("ITEM"):
                    parts = line.split()
                    if len(parts) > 1:
                        key = parts[1]
                        print(f"- {key}")

        tn.close()

    except ConnectionRefusedError:
        print(f"Помилка підключення до Memcached на {host}:{port}. Перевірте, чи запущено сервер.")
        sys.exit(1)
    except Exception as e:
        print(f"Сталася помилка: {e}")
        sys.exit(1)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Скрипт для очищення та перегляду вмісту Memcached.")
    parser.add_argument("--host", type=str, default=MEMCACHED_HOST, help=f"Хост Memcached сервера (за замовчуванням: {MEMCACHED_HOST})")
    parser.add_argument("--port", type=int, default=MEMCACHED_PORT, help=f"Порт Memcached сервера (за замовчуванням: {MEMCACHED_PORT})")
    parser.add_argument("--clear", action="store_true", help="Очистити весь кеш Memcached")
    parser.add_argument("--show_stats", action="store_true", help="Показати загальну статистику кешу Memcached")
    parser.add_argument("--show_keys", action="store_true", help="Спробувати показати ключі, що знаходяться в кеші Memcached")

    args = parser.parse_args()

    if args.clear:
        clear_memcache(args.host, args.port)

    if args.show_stats:
        show_cache_stats(args.host, args.port)

    if args.show_keys:
        show_cached_keys(args.host, args.port)

    if not any([args.clear, args.show_stats, args.show_keys]):
        print("Будь ласка, вкажіть одну з дій: --clear, --show_stats або --show_keys.")