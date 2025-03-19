from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the Logistic Regression model and TF-IDF Vectorizer
try:
    model = joblib.load("logistic_regression_imdb.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    print("✅ Model and Vectorizer loaded successfully")
except Exception as e:
    print(f"❌ Error loading files: {e}")
    raise

# Single review prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        review = data['review']
        review_vector = vectorizer.transform([review])
        prediction = model.predict(review_vector)[0]
        result = 'positive' if prediction == 1 else 'negative'
        return jsonify({'prediction': result})
    except KeyError:
        return jsonify({'error': 'Review text is missing'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Batch analysis endpoint
@app.route('/analyze-batch', methods=['POST'])
def analyze_batch():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '' or not file.filename.endswith('.csv'):
            return jsonify({'error': 'Invalid file. Please upload a CSV file'}), 400
        
        df = pd.read_csv(file)
        if 'review' not in df.columns:
            return jsonify({'error': 'CSV must contain a "review" column'}), 400
        
        reviews = df['review'].dropna().astype(str).tolist()
        if not reviews:
            return jsonify({'error': 'No valid reviews found in the CSV'}), 400
        
        review_vectors = vectorizer.transform(reviews)
        predictions = model.predict(review_vectors)
        
        positive_count = np.sum(predictions == 1)
        negative_count = np.sum(predictions == 0)
        total_count = len(reviews)
        
        return jsonify({'positive': int(positive_count), 'negative': int(negative_count), 'total': int(total_count)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Generate CSV with predictions endpoint
@app.route('/generate-csv', methods=['POST'])
def generate_csv():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '' or not file.filename.endswith('.csv'):
            return jsonify({'error': 'Invalid file. Please upload a CSV file'}), 400
        
        df = pd.read_csv(file)
        if 'review' not in df.columns:
            return jsonify({'error': 'CSV must contain a "review" column'}), 400
        
        reviews = df['review'].dropna().astype(str).tolist()
        if not reviews:
            return jsonify({'error': 'No valid reviews found in the CSV'}), 400
        
        review_vectors = vectorizer.transform(reviews)
        predictions = model.predict(review_vectors)
        df['label'] = ['Good' if pred == 1 else 'Bad' for pred in predictions]
        
        output_path = "predicted_reviews.csv"
        df.to_csv(output_path, index=False)
        
        return send_file(output_path, as_attachment=True)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
