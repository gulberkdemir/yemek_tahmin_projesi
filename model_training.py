import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import numpy as np


df_flights = pd.read_csv('data/flights.csv', sep=';', encoding='latin1', header=None)

df_flights.columns = [
    'STD', 'STD_LOCAL', 'FLT_NO', 'AC_TYPE',
    'FLIGHT_TIME_HOURS', 'BLOCK_TIME_HOURS', 'LEG_ISN_CORE', 'LEG_ISN_PAX', 'FLIGHT_SET_ID',
    'REPORTING_GROUP_NAME', 'FLIGHT_WAY', 'DEP_PORT', 'DEP_COUNTRY', 'ARR_PORT', 'ARR_COUNTRY',
    'DELAY_MIN', 'CAPACITY', 'FF_UYE_SAYISI',
    'GROUP_SALES_AD_PAX_COUNT', 'GROUP_SALES_CH_PAX_COUNT', 'GROUP_SALES_INF_PAX_COUNT',
    'TURKISH_M_AD_PAX_CNT', 'NON_TURKISH_M_AD_PAX_CNT',
    'TURKISH_F_AD_PAX_CNT', 'NON_TURKISH_F_AD_PAX_CNT',
    'TURKISH_CH_PAX_CNT', 'NON_TURKISH_CH_PAX_CNT',
    'TURKISH_INF_PAX_CNT', 'NON_TURKISH_INF_PAX_CNT',
    'BUNDLE_CATERING_CNT', 'PREORDER_CATERING_CNT'
]

df_flights_product = pd.read_csv('data/flights_product.csv', sep=';', header=None)

df_flights_product.columns = [
    'LEG_ISN_CORE',
    'LEG_ISN_PAX',
    'PRODUCT_ID',
    'SALE_COUNT',
    'DISCOUNT_SALE_COUNT',
    'CREW_DISCOUNT_SALE_COUNT',
    'TOT_SALE_COUNT'
]

df_flightset_load = pd.read_csv('data/flightset_load.csv', sep=';', header=None)

df_flightset_load.columns = [
    'FLIGHT_SET_ID',
    'PRODUCT_ID',
    'LOAD_COUNT'
]

df_cabin_crew = pd.read_csv('data/cabin_crew.csv', sep=';', header=None)

df_cabin_crew.columns = [
    'LEG_ISN_CORE',
    'LEG_ISN_PAX',
    'COMPANY_ID'
]

df_cabin_crew_perf = pd.read_csv('data/cabin_crew_performance.csv', sep=';', header=None)

df_cabin_crew_perf.columns = [
    'COMPANY_ID',
    'POINTS'
]

df_stock_out = pd.read_csv('data/stock_out.csv', sep=';', header=None)

df_stock_out.columns = [
    'ac_type',
    'ac_reg_code',
    'flight_isn',
    'sch_dep_dt',
    'departure_port',
    'arrival_port',
    'flight_way',
    'flight_no',
    'product_id',
    'amount'
]


df_purchase_sale_ratio = pd.read_csv('data/purchase_sale_ratio.csv', sep=';', header=None)

df_purchase_sale_ratio.columns = [
    'PRODUCT_ID',
    'PURCHASE_SALE_RATIO'
]


merged_df = pd.merge(df_flights, df_flights_product, on=['LEG_ISN_CORE', 'LEG_ISN_PAX'], how='inner')


merged_df = pd.merge(merged_df, df_flightset_load, on=['FLIGHT_SET_ID', 'PRODUCT_ID'], how='left')


merged_df = pd.merge(merged_df, df_purchase_sale_ratio, on='PRODUCT_ID', how='left')


merged_df = pd.merge(merged_df, df_cabin_crew, on=['LEG_ISN_CORE', 'LEG_ISN_PAX'], how='left')


merged_df = pd.merge(merged_df, df_cabin_crew_perf, on='COMPANY_ID', how='left')


df_stock_out['flight_isn'] = df_stock_out['flight_isn'].astype(int)
merged_df = pd.merge(merged_df, df_stock_out, left_on='LEG_ISN_CORE', right_on='flight_isn', how='left')

print(f"✅ Birleşmiş veri satır sayısı: {len(merged_df)}")
print("🧹 Eksik veri var mı:", merged_df.isnull().any().sum(), "adet sütunda eksik veri var")


numeric_cols = merged_df.select_dtypes(include='number').columns
merged_df[numeric_cols] = merged_df[numeric_cols].fillna(merged_df[numeric_cols].median())

categorical_cols = merged_df.select_dtypes(include='object').columns
merged_df[categorical_cols] = merged_df[categorical_cols].fillna('Unknown')

print(f"✅ Birleşmiş veri satır sayısı: {len(merged_df)}")
print("🧹 Eksik veri var mı:", merged_df.isnull().any().sum(), "adet sütunda eksik veri var")





merged_df.dropna(inplace=True)


y = merged_df['LOAD_COUNT']


feature_columns = [
    'FLIGHT_TIME_HOURS',
    'CAPACITY',
    'FF_UYE_SAYISI',
    'GROUP_SALES_AD_PAX_COUNT',
    'GROUP_SALES_CH_PAX_COUNT',
    'TURKISH_M_AD_PAX_CNT',
    'TURKISH_F_AD_PAX_CNT',
    'BUNDLE_CATERING_CNT',
    'PREORDER_CATERING_CNT',
    'SALE_COUNT',
    'DISCOUNT_SALE_COUNT',
    'CREW_DISCOUNT_SALE_COUNT',
    'TOT_SALE_COUNT',
    'PURCHASE_SALE_RATIO',
    'POINTS',
    'amount'  # stock_out tablosundan
]


X = merged_df[feature_columns]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"📉 RMSE: {rmse:.2f}")
print(f"📈 R^2: {r2:.4f}")


joblib.dump(model, 'yemek_yukleme_model.pkl')
joblib.dump(merged_df, 'merged_df.pkl')

print("✅ Model başarıyla kaydedildi.")