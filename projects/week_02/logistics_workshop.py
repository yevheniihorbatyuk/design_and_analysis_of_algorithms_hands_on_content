# %% [markdown]
#  # Практичний воркшоп: Побудова логістичної системи з нуля
#
#  **Сценарій:** Ми — логістична компанія, яка має доставити 15 замовлень по Київській області з 3 складів, використовуючи 3 вантажівки різної місткості.
#
#  **Наша мета:** Побудувати повний цикл оптимізації: від отримання матриці реальних дорожніх відстаней до візуалізації оптимальних маршрутів на карті.
#
#  ### Архітектура нашого рішення
#  ```
#  [ Вхідні дані: CSV з точками ]
#      ↓
#  [ Крок 1: Матриця відстаней (OSRM) ]
#      ↓
#  [ Крок 2: VRP Оптимізація (VROOM) ]
#      ↓
#  [ Крок 3: Візуалізація та аналітика (Folium, Pandas) ]
#  ```

# %%
# =============================================================================
# Клітинка 1: Налаштування середовища та імпорти
# =============================================================================
import pandas as pd
import requests
import json
import folium
from folium.plugins import HeatMap
import time
from datetime import datetime, timedelta

# Налаштування Pandas для кращого відображення
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 120)

# URL до наших локальних сервісів (запущених через Docker Compose)
OSRM_URL = "http://localhost:5000"
VROOM_URL = "http://localhost:3000/optimize"

print("✅ Середовище та змінні налаштовано.")

# %% [markdown]
#  ## Підготовка: Запускаємо інфраструктуру
#
#  Перш ніж почати, нам потрібно запустити два ключові сервіси: **OSRM** (для розрахунку маршрутів) та **VROOM** (для VRP оптимізації).
#
#  **Інструкція:**
#  1. **Встановіть Docker та Docker Compose.**
#  2. **Завантажте карту:** Завантажте файл `ukraine-latest.osm.pbf` з [Geofabrik](https://download.geofabrik.de/europe/ukraine.html) і покладіть його в папку `data/osrm_data/`.
#  3. **Підготуйте OSRM дані:** В терміналі, у корені проєкту, виконайте наступні команди. **Це може зайняти 10-30 хвилин і потребувати >16GB RAM!**
#     ```bash
#     # Крок 1: Extract (витягнення графу доріг)
#     docker run -t -v "${PWD}/data/osrm_data:/data" osrm/osrm-backend:v5.27.1 osrm-extract -p /opt/car.lua /data/ukraine-latest.osm.pbf
#     # Крок 2: Partition and Customize (для MLD алгоритму)
#     docker run -t -v "${PWD}/data/osrm_data:/data" osrm/osrm-backend:v5.27.1 osrm-partition /data/ukraine-latest.osrm
#     docker run -t -v "${PWD}/data/osrm_data:/data" osrm/osrm-backend:v5.27.1 osrm-customize /data/ukraine-latest.osrm
#     ```
#  4. **Запустіть сервіси:**
#     ```bash
#     docker-compose up -d
#     ```
#  5. **Перевірте, чи сервіси працюють:**
#     - OSRM: [http://localhost:5000](http://localhost:5000)
#     - VROOM: [http://localhost:3000](http://localhost:3000)

# %%
# =============================================================================
# Клітинка 2: Завантаження та підготовка вхідних даних
# =============================================================================
# Завантажуємо дані з CSV
try:
    points_df = pd.read_csv('data/points.csv')
    depots_df = points_df[points_df['type'] == 'depot'].reset_index(drop=True)
    jobs_df = points_df[points_df['type'] == 'delivery'].reset_index(drop=True)
    print("✅ Дані успішно завантажено:")
    print("\nСклади (депо):")
    print(depots_df)
    print("\nТочки доставки (замовлення):")
    print(jobs_df)
except FileNotFoundError:
    print("❌ Помилка: файл 'data/points.csv' не знайдено. Будь ласка, створіть його.")

# Функція для перетворення часу у секунди від півночі
def time_to_seconds(time_str):
    h, m = map(int, time_str.split(':'))
    return h * 3600 + m * 60

# %% [markdown]
#  ## Крок 1: Розрахунок матриці відстаней та часу
#
#  **Задача:** Отримати точні час та відстань по дорогах між усіма нашими точками (складами та клієнтами). Це "паливо" для будь-якого VRP-оптимізатора.
#
#  **Інструмент:** **OSRM (Open Source Routing Machine)**. Це надзвичайно швидкий рушій для маршрутизації, який використовує алгоритм **Contraction Hierarchies (CH)** для досягнення мілісекундних відповідей на запити.

# %%
# =============================================================================
# Клітинка 3: Запит до OSRM для отримання матриці
# =============================================================================
def get_osrm_matrix(points):
    """Отримує матрицю відстаней та часу від OSRM."""
    locations = ";".join([f"{lon},{lat}" for lon, lat in points[['lon', 'lat']].values])
    url = f"{OSRM_URL}/table/v1/driving/{locations}?annotations=duration,distance"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if data['code'] == 'Ok':
            print("✅ Матрицю відстаней та часу успішно отримано від OSRM.")
            return np.array(data['durations']), np.array(data['distances'])
        else:
            print(f"❌ Помилка OSRM: {data['message']}")
            return None, None
    except requests.exceptions.RequestException as e:
        print(f"❌ Не вдалося підключитися до OSRM: {e}")
        print("   Переконайтеся, що Docker контейнери запущені.")
        return None, None

# Об'єднуємо всі точки для запиту до OSRM
all_points = pd.concat([depots_df, jobs_df], ignore_index=True)
durations_matrix, distances_matrix = get_osrm_matrix(all_points)

if durations_matrix is not None:
    # Конвертуємо у хвилини для кращої читабельності
    durations_df = pd.DataFrame(durations_matrix / 60, 
                                index=all_points['name'], 
                                columns=all_points['name'])
    print("\nМатриця часу в дорозі (хвилини):")
    print(durations_df.round(1))

# %% [markdown]
#  ## Крок 2: VRP Оптимізація
#
#  **Задача:** Розподілити всі замовлення між наявними вантажівками так, щоб мінімізувати загальний час у дорозі, враховуючи обмеження (місткість, часові вікна).
#
#  **Інструмент:** **VROOM (Vehicle Routing Open-source Optimization Machine)**. Це високопродуктивний оптимізатор, написаний на C++, який вирішує складні варіанти VRP. Ми будемо спілкуватися з ним через простий REST API.

# %%
# =============================================================================
# Клітинка 4: Формування запиту до VROOM та його відправка
# =============================================================================
def solve_vrp_with_vroom(depots, jobs, time_matrix):
    """Формує та відправляє запит до VROOM."""
    
    # 1. Описуємо наші вантажівки
    vehicles = []
    capacities = [20, 30, 25] # Місткість для 3-х машин
    for i, depot in depots.iterrows():
        vehicles.append({
            "id": i,
            "profile": "driving",
            "start": [depot['lon'], depot['lat']],
            "end": [depot['lon'], depot['lat']],
            "capacity": [capacities[i]],
            "time_window": [time_to_seconds(depot['time_window_start']), time_to_seconds(depot['time_window_end'])]
        })

    # 2. Описуємо наші замовлення (jobs)
    vroom_jobs = []
    for i, job in jobs.iterrows():
        vroom_jobs.append({
            "id": i + len(depots), # Важливо, щоб ID були унікальними і відповідали матриці
            "location": [job['lon'], job['lat']],
            "service": 900, # 15 хвилин на обслуговування
            "amount": [int(job['demand'])],
            "time_windows": [[time_to_seconds(job['time_window_start']), time_to_seconds(job['time_window_end'])]]
        })
        
    # 3. Формуємо повний запит, включаючи матрицю часу
    request_payload = {
        "vehicles": vehicles,
        "jobs": vroom_jobs,
        "matrices": {
            "driving": {
                "durations": time_matrix.tolist()
            }
        }
    }
    
    # 4. Відправляємо запит
    try:
        headers = {'Content-Type': 'application/json'}
        response = requests.post(VROOM_URL, data=json.dumps(request_payload), headers=headers, timeout=30)
        response.raise_for_status()
        solution = response.json()
        
        if solution['code'] == 0:
            print("✅ Оптимальне рішення успішно отримано від VROOM.")
            return solution
        else:
            print(f"❌ Помилка оптимізації VROOM: {solution.get('error')}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"❌ Не вдалося підключитися до VROOM: {e}")
        return None

# Вирішуємо VRP
vroom_solution = solve_vrp_with_vroom(depots_df, jobs_df, durations_matrix)

if vroom_solution:
    # Аналізуємо результат
    summary = vroom_solution['summary']
    routes = vroom_solution['routes']
    
    print("\n--- Загальна статистика рішення ---")
    print(f"  Вартість (загальний час): {timedelta(seconds=summary['cost'])}")
    print(f"  Кількість використаних машин: {summary['vehicles']}")
    print(f"  Загальний час в дорозі: {timedelta(seconds=summary['duration'])}")
    print(f"  Час обслуговування: {timedelta(seconds=summary['service'])}")
    
    print("\n--- Деталі маршрутів ---")
    for route in routes:
        vehicle_id = route['vehicle']
        depot_name = depots_df.loc[vehicle_id, 'name']
        num_stops = len(route['steps']) - 2 # мінус старт і кінець
        print(f"\n  🚚 Машина #{vehicle_id+1} (зі складу '{depot_name}')")
        print(f"     - Кількість доставок: {num_stops}")
        print(f"     - Завантаженість: {route['delivery'][0]} / {capacities[vehicle_id]} од.")
        print(f"     - Час у дорозі: {timedelta(seconds=route['duration'])}")

# %% [markdown]
#  ## Крок 3: Візуалізація та Аналітика
#
#  **Задача:** Представити отримані маршрути у зрозумілому вигляді та проаналізувати ключові метрики ефективності.
#
#  **Інструменти:**
#  - **Folium:** Для створення інтерактивних карт на основі Leaflet.js.
#  - **Pandas:** Для агрегації та аналізу даних.

# %%
# =============================================================================
# Клітинка 5: Візуалізація маршрутів на інтерактивній карті
# =============================================================================
def visualize_routes_on_map(solution, depots, jobs):
    """Візуалізує маршрути на карті Folium."""
    if not solution or 'routes' not in solution:
        print("Немає даних для візуалізації.")
        return None

    # Створюємо карту з центром у Києві
    map_center = [50.4501, 30.5234]
    m = folium.Map(location=map_center, zoom_start=9, tiles="cartodbpositron")

    colors = ['#FF5733', '#33FF57', '#3357FF', '#FF33A1', '#A133FF']
    
    # Додаємо депо на карту
    for i, depot in depots.iterrows():
        folium.Marker(
            location=[depot['lat'], depot['lon']],
            popup=f"<strong>Склад:</strong> {depot['name']}",
            icon=folium.Icon(color='black', icon='warehouse', prefix='fa')
        ).add_to(m)

    # Малюємо маршрути
    for i, route in enumerate(solution['routes']):
        vehicle_id = route['vehicle']
        color = colors[i % len(colors)]
        
        # Отримуємо координати для лінії маршруту
        route_points = []
        for step in route['steps']:
            # VROOM не повертає координати, ми маємо їх взяти з наших даних
            # Індекси в матриці відповідають індексам в all_points
            point_index = step['id']
            point_info = all_points.loc[point_index]
            route_points.append([point_info['lat'], point_info['lon']])
        
        # Малюємо лінію маршруту
        folium.PolyLine(
            locations=route_points,
            color=color,
            weight=4,
            opacity=0.8,
            popup=f"Машина #{vehicle_id+1}"
        ).add_to(m)

        # Додаємо точки доставки
        for step in route['steps']:
            if step['type'] == 'job':
                job_index = step['id'] - len(depots) # Adjust index for jobs_df
                job_info = jobs.loc[job_index]
                arrival_time = datetime.fromtimestamp(route['arrival'] + step['arrival']).strftime('%H:%M')
                
                folium.CircleMarker(
                    location=[job_info['lat'], job_info['lon']],
                    radius=8,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.9,
                    popup=(f"<strong>{job_info['name']}</strong><br>"
                           f"Прибуття: {arrival_time}<br>"
                           f"Вікно: {job_info['time_window_start']}-{job_info['time_window_end']}<br>"
                           f"Попит: {job_info['demand']} од.")
                ).add_to(m)
    
    print("✅ Інтерактивну карту з маршрутами згенеровано. Вона буде відображена нижче.")
    return m

# Генеруємо та відображаємо карту
if vroom_solution:
    routes_map = visualize_routes_on_map(vroom_solution, depots_df, jobs_df)
    # Зберігаємо карту в HTML файл
    routes_map.save("logistics_routes_map.html")
    display(routes_map)

# %% [markdown]
#  ## Системне мислення: від алгоритму до Production
#
#  Ми щойно виконали основний цикл оптимізації. У реальній продакшн-системі це лише один з багатьох компонентів.
#
#  - **Дані:** Замість CSV, дані надходили б з бази даних (PostgreSQL + PostGIS).
#  - **Сервіси:** OSRM та VROOM працювали б як окремі мікросервіси, які можна масштабувати незалежно.
#  - **API:** Наш Python-скрипт був би обгорнутий у FastAPI, приймаючи нові замовлення та повертаючи маршрути.
#  - **Оркестрація:** Щоденна пакетна оптимізація запускалася б за розкладом за допомогою Apache Airflow.
#  - **Реальний час:** Нові замовлення "на льоту" додавалися б до існуючих маршрутів, викликаючи часткову переоптимізацію.
#
#  Розуміння того, як окремі алгоритми вписуються в загальну архітектуру, є ключовим для побудови складних та ефективних систем.

# %% [markdown]
#  ## Висновки воркшопу
#
#  1. **Ми побудували повний пайплайн логістичної оптимізації**, використовуючи інструменти індустріального рівня.
#  2. **Побачили силу спеціалізованих алгоритмів:** Contraction Hierarchies (в OSRM) для миттєвого розрахунку відстаней та просунуті евристики (в VROOM) для вирішення складної NP-hard задачі VRP.
#  3. **Створили практичний результат:** Інтерактивну карту з оптимальними маршрутами, яку можна показати менеджеру чи водієві.
#  4. **Зрозуміли місце алгоритмів у системі:** Наш воркшоп — це "серце" великої логістичної платформи.


# %% [markdown]
#  # Розділ 2: Від Алгоритму до Системи
#
#  Ми успішно вирішили конкретну задачу оптимізації. Тепер давайте зробимо крок назад і подивимося, як подібні рішення інтегруються в складні продакшн-системи, які використовуються в компаніях рівня Uber, Amazon чи Glovo.

# %% [markdown]
#  ## Архітектура Реальної Production Системи
#
#  Наш воркшоп — це "серце" оптимізації. У реальному світі воно оточене десятками інших сервісів, які забезпечують збір даних, взаємодію з користувачами, моніторинг та масштабування.
#
#  ```mermaid
#  flowchart TD
#      subgraph "Frontend Layer"
#          WebApp[Web UI для диспетчерів]
#          MobileApp[Мобільний додаток для водіїв]
#          Dashboard[Дашборд аналітики]
#      end
#
#      subgraph "Backend Services (Microservices)"
#          APIGateway[API Gateway]
#          OrderService[Сервіс Замовлень]
#          OptimizationService[Наш воркшоп: Сервіс Оптимізації]
#          NotificationService[Сервіс Повідомлень]
#          UserService[Сервіс Користувачів]
#      end
#
#      subgraph "Routing & Optimization Engines"
#          OSRM[OSRM Engine]
#          VROOM[VROOM Engine]
#      end
#
#      subgraph "Data Layer"
#          Postgres[PostgreSQL + PostGIS]
#          Redis[Redis Cache]
#          Kafka[Kafka Event Stream]
#      end
#
#      Frontend Layer --> APIGateway
#      APIGateway --> OrderService
#      APIGateway --> UserService
#      OrderService -- надсилає подію --> Kafka
#      Kafka -- сповіщає --> OptimizationService
#      OptimizationService -- запит матриці --> OSRM
#      OptimizationService -- запит оптимізації --> VROOM
#      OptimizationService -- зберігає результат --> Postgres
#      OptimizationService -- надсилає подію --> Kafka
#      Kafka -- сповіщає --> NotificationService
#      NotificationService --> MobileApp
#      APIGateway -- читає дані --> Postgres
#      APIGateway -- читає кеш --> Redis
#  ```
#
#  **Роль наших інструментів у цій архітектурі:**
#  - **OSRM/VROOM:** Виділені, незалежні сервіси, які можна масштабувати окремо.
#  - **Python-скрипт (наш ноутбук):** Був би обгорнутий у мікросервіс (напр. на FastAPI), який слухає події про нові замовлення з Kafka, викликає OSRM/VROOM, і публікує результат.
#  - **PostGIS:** Зберігає гео-дані (координати, полігони зон доставки), що дозволяє робити ефективні просторові запити (напр., "знайти всіх водіїв у радіусі 2 км").

# %% [markdown]
#  ## Системне мислення: Реалії Production
#
#  | Теорія (що ми зробили) | Реальність (що відбувається в production) | Рішення в Production |
#  | :--- | :--- | :--- |
#  | **Статичний набір точок** | Нові замовлення надходять постійно, існуючі скасовуються. | **Динамічна переоптимізація:** Запускати VROOM кожні 1-5 хвилин для оновлення маршрутів. |
#  | **Ідеальні дані** | Водій застряг у заторі, клієнт не відповідає, GPS-трекер неточний. | **Event-driven архітектура:** Система реагує на події (затримка, скасування) і автоматично запускає перерахунок. |
#  | **Один критерій (час)** | Потрібно балансувати час, вартість, завантаженість, пріоритет клієнта, викиди CO₂. | **Багатокритеріальна оптимізація:** Використання вагової функції або пошук Парето-оптимальних рішень. |
#  | **Оптимальне рішення** | "Достатньо добре" рішення за 1 секунду набагато цінніше за оптимальне за 10 хвилин. | **Метаевристики та Anytime Algorithms:** Алгоритми, що швидко знаходять хороше рішення і поступово його покращують. |
#  | **Надійні сервіси** | OSRM може впасти, VROOM може не знайти рішення. | **Fallback-стратегії:** Якщо VROOM не відповів за 2 секунди, використовувати простішу евристику (напр., призначити найближчого водія). |

# %% [markdown]
#  ## Рефлексія та Обговорення
#
#  **Питання для закріплення матеріалу:**
#
#  1.  **Теоретичні:**
#      - Чому `Floyd-Warshall` (`O(V³)`), незважаючи на високу складність, корисний для створення матриць відстаней на невеликих графах?
#      - Чому `VRP` є `NP-hard` задачею, а пошук найкоротшого шляху (Dijkstra) — ні?
#      - Яку структуру даних використовує `PostGIS` для швидкого пошуку "точок у полігоні"? (Відповідь: R-Tree).
#
#  2.  **Практичні:**
#      - Що важливіше для логістичного стартапу: швидкість відповіді API чи оптимальність маршрутів? Як знайти баланс?
#      - Як би ви змінили архітектуру для підтримки "батчингу" замовлень (один кур'єр забирає кілька замовлень з одного ресторану)?
#      - Які дані потрібно збирати з мобільних додатків водіїв, щоб покращувати модель оптимізації в майбутньому?

# %% [markdown]
#  ## Домашнє Завдання та Подальші Кроки
#
#  ### Рівень 1: Розширення нашого воркшопу
#  1.  **Додайте обмеження по вазі:** Модифікуйте `points.csv`, додавши колонку `weight`. Оновіть VROOM-запит, щоб враховувати не тільки `amount` (об'єм), але й вагу.
#  2.  **Додайте пріоритети:** Додайте до замовлень поле `priority` (від 0 до 100). VROOM підтримує це поле "out of the box". Проаналізуйте, як змінилися маршрути.
#  3.  **Реалізуйте Fallback:** Напишіть просту Python-функцію, яка реалізує "жадібний" алгоритм (завжди відправляти найближчого вільного водія до найближчого замовлення). Порівняйте результат з VROOM.
#
#  ### Рівень 2: Новий домен — Таксі-сервіс
#  **Задача:** Створити симулятор простого таксі-сервісу.
#  1.  **Симуляція:** Напишіть скрипт, який генерує випадкові замовлення (пасажир з точки А в точку Б) кожні 10-30 секунд.
#  2.  **Matching:** Для кожного нового замовлення, знайдіть 5 найближчих вільних водіїв (використовуйте OSRM `table` запит).
#  3.  **Призначення:** Призначте водія, для якого сумарний час (доїзд до клієнта + поїздка з клієнтом) буде мінімальним.
#  4.  **Візуалізація:** Відобразіть на `Folium` мапі рух машин та нові замовлення в реальному часі (можна оновлювати карту в циклі).
#
#  ### Рівень 3: Дослідження
#  - **Прочитайте про алгоритм `Large Neighborhood Search (LNS)`**, який лежить в основі VROOM, і спробуйте написати його спрощену версію.
#  - **Дослідіть `OR-Tools` від Google:** Спробуйте вирішити нашу VRP-задачу за допомогою `pywrapcp` бібліотеки і порівняйте результат та складність коду з VROOM.
#  - **Проаналізуйте реальні дані:** Завантажте [NYC Taxi Dataset](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) і побудуйте теплову карту попиту в різний час доби.

# %% [markdown]
#  ## Завершення курсу
#
#  ### Ключові висновки:
#
#  1.  **Алгоритми — це інструменти, а не самоціль.** Розуміння задачі та контексту важливіше, ніж знання напам'ять складності кожного алгоритму.
#  2.  **Не бійтеся "чорних скриньок".** Такі інструменти, як OSRM та VROOM, є результатом десятиліть досліджень. Ваше завдання як інженера — навчитися їх правильно "готувати" та інтегрувати.
#  3.  **Системне мислення вирішує все.** Найкращий алгоритм марний, якщо він не вписаний у надійну, масштабовану та гнучку архітектуру.
#
#  > **Ви тепер маєте не просто набір знань, а дорожню карту для вирішення складних оптимізаційних задач. Шлях від теорії до практики пройдено. Тепер — час створювати власні системи!**

# %%
# =============================================================================
# Клітинка-довідник: Корисні команди та ресурси
# =============================================================================
print("--- Команди Docker ---")
print("Запустити всі сервіси: docker-compose up -d")
print("Зупинити всі сервіси: docker-compose down")
print("Переглянути логи OSRM: docker logs osrm_backend")
print("Переглянути логи VROOM: docker logs vroom_express")

print("\n--- Корисні посилання ---")
print("OSRM API документація: http://project-osrm.org/docs/v5.27.1/api/")
print("VROOM API документація: https://github.com/VROOM-Project/vroom/blob/master/docs/API.md")
print("OR-Tools VRP (Python): https://developers.google.com/optimization/routing/vrp")
print("Folium документація: https://python-visualization.github.io/folium/")