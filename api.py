from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)


model = joblib.load('yemek_yukleme_model.pkl')
merged_df = joblib.load('merged_df.pkl')



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
    'amount'
]

@app.route("/predict_manual", methods=["POST"])
def predict_manual():
    try:
        data = request.json

   
        for col in feature_columns:
            if col not in data:
                return jsonify({"error": f"Eksik alan: {col}"}), 400


        df_input = pd.DataFrame([data])
        
   
        prediction = model.predict(df_input)[0]
        return jsonify({"predicted_load_count": round(prediction, 2)})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route("/predict_by_flightno", methods=["POST"])
def predict_by_flightno():
    try:
        flight_no = str(request.json.get("flight_no")).strip()
        flight_data = merged_df[merged_df["FLT_NO"].astype(str).str.strip() == flight_no]



        if flight_data.empty:
            return jsonify({"error": "Flight Number not found"}), 404

        
        input_data = flight_data[feature_columns].iloc[0:1]
        prediction = model.predict(input_data)[0]

        return jsonify({"predicted_load_count": round(prediction, 2)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)