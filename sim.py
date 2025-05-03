# test_agent.py
import requests
import time
import random

def simulate_agent():
    data = {
        "user_id": "test_user",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "features": {
            "file_operations": random.randint(1, 10),
            "http_requests": random.randint(10, 50),
            "usb_usage": random.randint(0, 1),
            "process_count": random.randint(5, 25),
            "websites_visited": random.randint(0, 15)
        }
    }
    r = requests.post("http://127.0.0.1:8000/predict", json=data)
    print("Server response:", r.json())

if __name__ == "__main__":
    simulate_agent()
