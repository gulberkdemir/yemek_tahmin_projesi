import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Verileri yükleme
df_flights = pd.read_csv('data/flights.csv', sep=';', encoding='latin1')
df_flights_product = pd.read_csv('data/flights_product.csv', sep=';')

# flights.csv'e anlamlı sütun isimleri verelim
df_flights.columns = [
    'Scheduled_Departure_Local', 'Scheduled_Arrival_Local', 'Flight_No', 'Aircraft_Type',
    'Flight_Time_Hours', 'Block_Time_Hours', 'Leg_Isn_Core', 'Leg_Isn_Pax', 'Flight_Set_ID',
    'Reporting_Group_Name', 'Flight_Way', 'Dep_Port', 'Dep_Country', 'Arr_Port', 'Arr_Country',
    'Delay_Minutes', 'Capacity', 'FF_Uye_Sayisi', 'Group_Adult_Pax', 'Group_Child_Pax', 'Group_Infant_Pax',
    'Turkish_M_Adult', 'NonTurkish_M_Adult', 'Turkish_F_Adult', 'NonTurkish_F_Adult',
    'Turkish_Child', 'NonTurkish_Child', 'Turkish_Infant', 'NonTurkish_Infant',
    'Bundle_Catering', 'Preorder_Catering'
]

# flights_product.csv'de tahmin edilecek sütunu yeniden adlandır
df_flights_product = df_flights_product.rename(columns={
    df_flights_product.columns[-1]: 'ProductCount'
})

# Dönüştür (en önemli adım)
df_flights['Leg_Isn_Core'] = df_flights['Leg_Isn_Core'].astype(int)
df_flights_product[df_flights_product.columns[0]] = df_flights_product[df_flights_product.columns[0]].astype(int)

# Doğru eşleşme ile birleştir
merged_df = pd.merge(
    df_flights,
    df_flights_product,
    left_on='Leg_Isn_Core',
    right_on=df_flights_product.columns[0]
)

print("✅ Satır sayısı (merged_df):", len(merged_df))

# Eksik verileri temizle
merged_df.dropna(inplace=True)

# Özellikleri seçme
X = merged_df[[
    'Flight_Time_Hours', 'Capacity', 'FF_Uye_Sayisi',
    'Group_Adult_Pax', 'Group_Child_Pax', 'Preorder_Catering'
]]

y = merged_df['ProductCount']

# Veriyi eğitim ve test olarak ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modeli kur ve eğit
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Tahmin ve değerlendirme
y_pred = rf_model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f'Model RMSE: {rmse}')
print(f'Model R^2 Skoru: {r2}')

# Modeli kaydet
joblib.dump(rf_model, 'yemek_tahmin_modeli.pkl')
