import requests

url = "http://127.0.0.1:5000/predict_manual"

data = {
    'FLIGHT_TIME_HOURS': 3.5,
    'CAPACITY': 250,
    'FF_UYE_SAYISI': 40,
    'GROUP_SALES_AD_PAX_COUNT': 10,
    'GROUP_SALES_CH_PAX_COUNT': 4,
    'TURKISH_M_AD_PAX_CNT': 22,
    'TURKISH_F_AD_PAX_CNT': 18,
    'BUNDLE_CATERING_CNT': 12,
    'PREORDER_CATERING_CNT': 3,
    'SALE_COUNT': 15,
    'DISCOUNT_SALE_COUNT': 5,
    'CREW_DISCOUNT_SALE_COUNT': 10,
    'TOT_SALE_COUNT': 100,
    'PURCHASE_SALE_RATIO': 0.72,
    'POINTS': 82,
    'amount': 6
}

response = requests.post(url, json=data)
print("✅ Status Code:", response.status_code)
print("🍱 Tahmin:", response.json())
