from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd

app = Flask(__name__)

# Muat model saat aplikasi dimulai
model = load_model('collaborative_filtering.h5', compile=False)

@app.route('/predict', methods=['POST'])
def predict():
    # Ambil input dari request
    input_data = request.get_json()
    phone_count = input_data['phone_count']
    user_click = input_data['user_click']
    user_rating = input_data['user_rating']

    user_click_input = np.reshape(user_click, (1, phone_count))
    user_rating_input = np.reshape(user_rating, (1, phone_count))
    user_rating_input = user_rating_input / 5
    result = model.predict([user_rating_input, user_click_input])
    result = result[0]

    probability = np.array(result)
    df_result = pd.DataFrame()
    df_result['probability'] = np.reshape(probability, phone_count).astype(float)
    df_result['id'] = range(1, phone_count + 1)
    sorted_df_result = df_result.sort_values(by='probability', ascending=False)
    sorted_df_result['probability'] = sorted_df_result['probability'] / sorted_df_result['probability'].max()
    top_10_results = sorted_df_result.head(10)['id']
   
    top_10_results_json = top_10_results.to_json(orient='records')

    return top_10_results_json

if __name__ == '__main__':
    app.run(debug=True)

# host='0.0.0.0', port=8080, debug=True