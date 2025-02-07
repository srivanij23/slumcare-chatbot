import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the data
df = pd.read_csv('C:/Users/JADAV SRIVANI/Desktop/SLUMCARE/aiAgent/complaint_data.csv')

# Check the first few rows of the data
df.head()

# Split the data into features (X) and labels (y)
X = df['Complaint Description']
y_category = df['Category']
y_authority = df['Authority']

# Use TF-IDF Vectorizer to convert text into numerical format
vectorizer = TfidfVectorizer(stop_words='english')
X_tfidf = vectorizer.fit_transform(X)

# Label encode the 'Category' and 'Authority' columns for classification
category_encoder = LabelEncoder()
authority_encoder = LabelEncoder()

y_category_encoded = category_encoder.fit_transform(y_category)
y_authority_encoded = authority_encoder.fit_transform(y_authority)

# Split the data into training and testing sets
X_train, X_test, y_train_category, y_test_category, y_train_authority, y_test_authority = train_test_split(
    X_tfidf, y_category_encoded, y_authority_encoded, test_size=0.2, random_state=42
)

# Check the shape of the split data
X_train.shape, X_test.shape

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Train the model for Category prediction
category_model = RandomForestClassifier(n_estimators=100, random_state=42)
category_model.fit(X_train, y_train_category)

# Train the model for Authority prediction
authority_model = RandomForestClassifier(n_estimators=100, random_state=42)
authority_model.fit(X_train, y_train_authority)

# Predictions on the test data
y_pred_category = category_model.predict(X_test)
y_pred_authority = authority_model.predict(X_test)

# Evaluate the models
category_accuracy = accuracy_score(y_test_category, y_pred_category)
authority_accuracy = accuracy_score(y_test_authority, y_pred_authority)

# ✅ Fix: Handle missing classes in the test set
# Get unique labels from the test set
unique_category_labels = np.unique(y_test_category)
unique_authority_labels = np.unique(y_test_authority)

# Map these labels back to the original class names
category_target_names = category_encoder.inverse_transform(unique_category_labels)
authority_target_names = authority_encoder.inverse_transform(unique_authority_labels)

# Classification reports with corrected labels
category_report = classification_report(
    y_test_category, 
    y_pred_category, 
    labels=unique_category_labels, 
    target_names=category_target_names
)

authority_report = classification_report(
    y_test_authority, 
    y_pred_authority, 
    labels=unique_authority_labels, 
    target_names=authority_target_names
)

# Display the results
print(f"Category Accuracy: {category_accuracy}")
print(category_report)
print(f"Authority Accuracy: {authority_accuracy}")
print(authority_report)
import joblib

# Save the trained models
joblib.dump(category_model, 'category_model.pkl')
joblib.dump(authority_model, 'authority_model.pkl')

# Save the encoders and vectorizer
joblib.dump(category_encoder, 'category_encoder.pkl')
joblib.dump(authority_encoder, 'authority_encoder.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print("Models and encoders saved successfully!")
