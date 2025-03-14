from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np

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
        # Check if a file is uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not file.filename.endswith('.csv'):
            return jsonify({'error': 'File must be a CSV'}), 400

        # Read the CSV file
        df = pd.read_csv(file)
        
        # Check if 'review' column exists
        if 'review' not in df.columns:
            return jsonify({'error': 'CSV must contain a "review" column'}), 400
        
        # Handle missing or empty reviews
        reviews = df['review'].dropna().astype(str).tolist()
        if not reviews:
            return jsonify({'error': 'No valid reviews found in the CSV'}), 400

        # Preprocess all reviews using the TF-IDF Vectorizer
        review_vectors = vectorizer.transform(reviews)

        # Make predictions for all reviews
        predictions = model.predict(review_vectors)

        # Count positive and negative reviews
        positive_count = np.sum(predictions == 1)  # Assuming 1 = positive
        negative_count = np.sum(predictions == 0)  # Assuming 0 = negative
        total_count = len(reviews)

        # Return the result as JSON
        return jsonify({
            'positive': int(positive_count),
            'negative': int(negative_count),
            'total': int(total_count)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)