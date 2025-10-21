from locust import HttpUser, task, between
import random

class APILoadTest(HttpUser):
    # Час між запитами для кожного користувача
    wait_time = between(0.15, 0.35)  

    @task
    def test_zipcode_api(self):
        # Визначаємо можливі країни та діапазон поштових кодів
        countries = ["us"]  # США, Канада, Мексика
        zipcodes = range(33140, 33170)  # Діапазон поштових кодів

        # Генеруємо випадкові значення
        country_code = random.choice(countries)
        zipcode = random.choice(zipcodes)

        # Формуємо endpoint
        endpoint = f"/api/v1/zipcode/{country_code}/{zipcode}"

        # Виконуємо GET запит і перевіряємо статус
        with self.client.get(endpoint, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed with status code: {response.status_code}")