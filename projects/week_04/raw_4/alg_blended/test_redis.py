import redis

class RedisTest:
    def __init__(self, host='localhost', port=6379, db=0):
        self.redis_client = redis.StrictRedis(host=host, port=port, db=db)

    def get_all_keys(self):
        # Get all keys from Redis
        return self.redis_client.keys()

    def get_all_data(self):
        # Retrieve all key-value pairs from Redis
        data = {}
        for key in self.get_all_keys():
            data[key.decode('utf-8')] = self.redis_client.get(key).decode('utf-8')
        return data

if __name__ == "__main__":
    # Initialize RedisTest instance
    redis_test = RedisTest()

    # Get all data from Redis
    all_data = redis_test.get_all_data()

    # Print the results
    print("All data in Redis:")
    for key, value in all_data.items():
        print(f"{key}: {value}")