from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
import pandas as pd

app = Flask(__name__)

# Muat model saat aplikasi dimulai
model = load_model('content_based.h5', compile=False)
brand_choice_list = ['Oppo', 'Asus', 'Infinix', 'Samsung', 'Vivo', 'Huawei', 'Apple', 'Realme', 'Xiaomi', 'Poco', 'lainnya/tidak ada']

def preprocess_user(x):
  one_hot = x[-1]
  one_hot = tf.one_hot(brand_choice_list.index(one_hot), len(brand_choice_list))
  x = tf.cast(tf.convert_to_tensor(x[:-1]), tf.float32)
  return tf.expand_dims(tf.concat([x, one_hot], 0), 0)

@app.route('/predict', methods=['POST'])
def predict():
    # Ambil input dari request
    
    input_data = request.get_json()
    user_survey = preprocess_user(input_data['user_survey'])
    phone_count = input_data['phone_count']

    u_vector = model.predict(user_survey)
    p_vector = pd.read_csv('phone_vector.csv', header=None)

    p_vector = tf.convert_to_tensor(p_vector)
    p_vector = tf.cast(p_vector, tf.float32)
    rating_pred_list = []
    for p in p_vector:
        rating_pred_list.append(tf.tensordot(u_vector, tf.expand_dims(p, 0), axes=(1,1)) * 5)
        
    df_result = pd.DataFrame()
    df_result['probability'] = np.reshape(rating_pred_list, phone_count).astype(float)
    df_result['id'] = range(1, phone_count + 1)
    df_result.sort_values(by='probability', ascending=False)
    sorted_df_result = df_result.sort_values(by='probability', ascending=False)
    sorted_df_result['probability'] = sorted_df_result['probability'] / sorted_df_result['probability'].max()
    top_10_results = sorted_df_result.head(10)['id']
    top_10_results_json = top_10_results.to_json(orient='records')
    return top_10_results_json
        
if __name__ == '__main__':
    app.run(debug=True)