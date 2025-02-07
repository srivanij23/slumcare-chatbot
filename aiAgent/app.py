from flask import Flask, request, jsonify
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load the trained models and vectorizer
category_model = joblib.load('category_model.pkl')
authority_model = joblib.load('authority_model.pkl')
category_encoder = joblib.load('category_encoder.pkl')
authority_encoder = joblib.load('authority_encoder.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# AI Agent Prediction Function
def ai_agent_predict(complaint_text):
    complaint_tfidf = vectorizer.transform([complaint_text])
    predicted_category = category_model.predict(complaint_tfidf)
    predicted_authority = authority_model.predict(complaint_tfidf)
    category_name = category_encoder.inverse_transform(predicted_category)[0]
    authority_name = authority_encoder.inverse_transform(predicted_authority)[0]
    return category_name, authority_name

# API Endpoint
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    complaint_text = data.get('complaint')
    
    if not complaint_text:
        return jsonify({'error': 'Complaint text is required'}), 400

    category, authority = ai_agent_predict(complaint_text)
    return jsonify({
        'category': category,
        'authority': authority
    })

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, port=5000)
