import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
from keras import layers, Sequential
import os

# Function to preprocess features for ML
def preprocess_features(features_df):
    # Handle missing values if any
    features_df.fillna(0, inplace=True)
    
    # Label encoding for categorical features
    label_encoder = LabelEncoder()
    categorical_columns = ['src', 'dst']
    for column in categorical_columns:
        if features_df[column].dtype == 'object':
            features_df[column] = label_encoder.fit_transform(features_df[column])
    
    # Scaling numerical features
    numerical_features = ['rssi', 'timestamp']
    scaler = StandardScaler()
    features_df[numerical_features] = scaler.fit_transform(features_df[numerical_features])
    
    return features_df


# Function for pattern analysis using machine learning
def analyze_patterns_with_ml(X_train, y_train, X_test, y_test):
    model_path = 'deauth_detection_model.h5'
    
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
    else:
        model = Sequential([
            layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
            layers.Dropout(0.5),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=1)
        model.save(model_path)
    
    # Model predictions
    predictions = model.predict(X_test)
    predictions = (predictions > 0.5).astype(int)
    
    # Evaluation metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    return predictions


# Example data preparation for training and evaluation
def prepare_and_train_model(data):
    # Assuming 'data' is a DataFrame containing the extracted features and labels
    features = preprocess_features(data.drop(columns=['label']))
    labels = data['label']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    
    # Train and evaluate the model
    analyze_patterns_with_ml(X_train, y_train, X_test, y_test)
