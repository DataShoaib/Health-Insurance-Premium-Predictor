import requests
import random
import time

API_URL = "http://51.21.161.214:8000/predict"

SAMPLE_INPUTS = [
    {"age": 19, "gender": "male",   "bmi": 27.9, "bloodpressure": 130, "diabetic": "No",  "children": 0, "smoker": "Yes", "region": "southeast"},
    {"age": 18, "gender": "female", "bmi": 33.7, "bloodpressure": 80,  "diabetic": "No",  "children": 1, "smoker": "No",  "region": "northwest"},
    {"age": 28, "gender": "male",   "bmi": 33.0, "bloodpressure": 90,  "diabetic": "Yes", "children": 3, "smoker": "No",  "region": "southwest"},
    {"age": 33, "gender": "female", "bmi": 22.7, "bloodpressure": 70,  "diabetic": "No",  "children": 0, "smoker": "No",  "region": "northeast"},
    {"age": 45, "gender": "male",   "bmi": 30.1, "bloodpressure": 110, "diabetic": "Yes", "children": 2, "smoker": "Yes", "region": "southeast"},
    {"age": 60, "gender": "female", "bmi": 25.8, "bloodpressure": 140, "diabetic": "Yes", "children": 0, "smoker": "No",  "region": "northwest"},
    {"age": 25, "gender": "male",   "bmi": 26.2, "bloodpressure": 75,  "diabetic": "No",  "children": 0, "smoker": "No",  "region": "northeast"},
    {"age": 52, "gender": "female", "bmi": 30.7, "bloodpressure": 120, "diabetic": "Yes", "children": 1, "smoker": "No",  "region": "southwest"},
    {"age": 37, "gender": "male",   "bmi": 29.8, "bloodpressure": 95,  "diabetic": "No",  "children": 2, "smoker": "No",  "region": "northwest"},
    {"age": 31, "gender": "female", "bmi": 25.7, "bloodpressure": 85,  "diabetic": "No",  "children": 0, "smoker": "Yes", "region": "southeast"},
    {"age": 46, "gender": "male",   "bmi": 33.4, "bloodpressure": 100, "diabetic": "Yes", "children": 1, "smoker": "No",  "region": "northeast"},
    {"age": 23, "gender": "female", "bmi": 17.4, "bloodpressure": 65,  "diabetic": "No",  "children": 1, "smoker": "No",  "region": "northwest"},
    {"age": 56, "gender": "male",   "bmi": 39.8, "bloodpressure": 150, "diabetic": "Yes", "children": 0, "smoker": "Yes", "region": "southeast"},
    {"age": 62, "gender": "female", "bmi": 26.2, "bloodpressure": 130, "diabetic": "Yes", "children": 0, "smoker": "No",  "region": "southwest"},
    {"age": 34, "gender": "male",   "bmi": 31.9, "bloodpressure": 105, "diabetic": "No",  "children": 0, "smoker": "Yes", "region": "southeast"},
    {"age": 27, "gender": "female", "bmi": 42.1, "bloodpressure": 88,  "diabetic": "No",  "children": 0, "smoker": "No",  "region": "northwest"},
    {"age": 59, "gender": "male",   "bmi": 27.7, "bloodpressure": 135, "diabetic": "Yes", "children": 3, "smoker": "No",  "region": "northeast"},
    {"age": 30, "gender": "female", "bmi": 35.3, "bloodpressure": 92,  "diabetic": "No",  "children": 0, "smoker": "Yes", "region": "southwest"},
    {"age": 63, "gender": "male",   "bmi": 23.8, "bloodpressure": 145, "diabetic": "Yes", "children": 0, "smoker": "No",  "region": "southeast"},
    {"age": 22, "gender": "female", "bmi": 28.5, "bloodpressure": 72,  "diabetic": "No",  "children": 0, "smoker": "No",  "region": "northeast"},
    {"age": 40, "gender": "male",   "bmi": 36.0, "bloodpressure": 118, "diabetic": "Yes", "children": 2, "smoker": "Yes", "region": "southwest"},
    {"age": 55, "gender": "female", "bmi": 32.7, "bloodpressure": 128, "diabetic": "Yes", "children": 2, "smoker": "No",  "region": "northwest"},
    {"age": 29, "gender": "male",   "bmi": 24.6, "bloodpressure": 78,  "diabetic": "No",  "children": 1, "smoker": "No",  "region": "northeast"},
    {"age": 48, "gender": "female", "bmi": 38.2, "bloodpressure": 112, "diabetic": "Yes", "children": 3, "smoker": "Yes", "region": "southeast"},
    {"age": 35, "gender": "male",   "bmi": 21.3, "bloodpressure": 82,  "diabetic": "No",  "children": 1, "smoker": "No",  "region": "southwest"},
]

def simulate(n_requests=200):
    print(f"Starting simulation — {n_requests} requests to {API_URL}")
    print("-" * 55)

    success = 0
    failed  = 0

    for i in range(n_requests):
        payload = random.choice(SAMPLE_INPUTS)
        try:
            r = requests.post(API_URL, json=payload, timeout=5)
            if r.status_code == 200:
                success += 1
                print(f"[{i+1:3}] OK     | claim: {r.json()['claim']:>10.2f}")
            else:
                failed += 1
                print(f"[{i+1:3}] ERROR  | status: {r.status_code} | {r.text}")
        except Exception as e:
            failed += 1
            print(f"[{i+1:3}] FAILED | {e}")

        time.sleep(random.uniform(0.1, 0.4))

    print("-" * 55)
    print(f"Done!  Success: {success}  |  Failed: {failed}")

if __name__ == "__main__":
    simulate()