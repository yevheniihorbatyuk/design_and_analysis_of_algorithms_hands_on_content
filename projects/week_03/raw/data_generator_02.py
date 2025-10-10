import random
import string
import time
from dataclasses import dataclass
from typing import Generator

@dataclass
class UsageEvent:
    user_id: str
    music_id: str
    timestamp: str
    duration_point: str
    completed: str
    device: str

def random_string(length: int) -> str:
    """Generate a random string of uppercase letters and digits."""
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))

def generate_usage_events() -> Generator[UsageEvent, None, None]:
    """A generator that yields random UsageEvent objects."""
    devices = ["mobile", "desktop", "tablet", "smart_tv"]

    while True:
        user_id = random_string(10)
        music_id = random_string(8)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        duration_point = str(random.randint(30, 300))  # Duration in seconds (30 to 300 seconds)
        completed = random.choice(["true", "false"])
        device = random.choice(devices)

        yield UsageEvent(
            user_id=user_id,
            music_id=music_id,
            timestamp=timestamp,
            duration_point=duration_point,
            completed=completed,
            device=device
        )

# Example of using the generator
if __name__ == "__main__":
    event_generator = generate_usage_events()
    
    for _ in range(5):  # Generate 5 events for demonstration
        event = next(event_generator)
        print(event)
