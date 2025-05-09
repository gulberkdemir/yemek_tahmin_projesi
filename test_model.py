import os
import joblib
import pandas as pd

# ✅ Model dosyası var mı kontrol et
if not os.path.exists('yemek_yukleme_model.pkl'):
    raise FileNotFoundError("❌ Model dosyası bulunamadı. Lütfen önce model_training.py dosyasını çalıştır.")

# ✅ Modeli yükle
model = joblib.load('yemek_yukleme_model.pkl')

# ✅ Test verisi (manuel örnek)
sample_input = pd.DataFrame([{
    'FLIGHT_TIME_HOURS': 2.5,
    'CAPACITY': 180,
    'FF_UYE_SAYISI': 40,
    'GROUP_SALES_AD_PAX_COUNT': 10,
    'GROUP_SALES_CH_PAX_COUNT': 4,
    'TURKISH_M_AD_PAX_CNT': 22,
    'TURKISH_F_AD_PAX_CNT': 18,
    'BUNDLE_CATERING_CNT': 12,
    'PREORDER_CATERING_CNT': 3,
    'SALE_COUNT': 95,
    'DISCOUNT_SALE_COUNT': 15,
    'CREW_DISCOUNT_SALE_COUNT': 10,
    'TOT_SALE_COUNT': 120,
    'PURCHASE_SALE_RATIO': 0.72,
    'POINTS': 82,
    'amount': 6
}])

# ✅ Tahmin yap
predicted_yemek = model.predict(sample_input)[0]
print(f"🍱 Tahmin edilen yemek yükleme miktarı: {predicted_yemek:.2f} birim")
