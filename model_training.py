import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Verileri yükleme
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

