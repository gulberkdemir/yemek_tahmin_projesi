import requests

url = "http://127.0.0.1:5000/predict_by_flightno"
data = {"flight_no": "301"}  # örnek flight_no değeri (FLT_NO kolonundaki gerçek bir değer olmalı)

response = requests.post(url, json=data)

print("✅ Status Code:", response.status_code)
print("🛬 Response:", response.json())
