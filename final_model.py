import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
from sklearn.utils import resample
import numpy as np


# Load the CSV file into a DataFrame
df = pd.read_csv('file_path')

# Preprocessing
df['Time'] = pd.to_datetime(df['Time'])  # Convert 'Time' column to datetime
df['Hour'] = df['Time'].dt.hour  # Extract hour of the day
df['Day_of_week'] = df['Time'].dt.dayofweek  # Extract day of the week (Monday=0, Sunday=6)

# Fill missing values in 'Info' column with an empty string
df['Info'] = df['Info'].fillna('')

# Encode the target variable
df['Is_Handshake'] = df['Info'].apply(lambda x: 1 if 'Key (Message 4 of 4)' in x else 0)

# Upsample the minority class
df_majority = df[df['Is_Handshake'] == 0]
df_minority = df[df['Is_Handshake'] == 1]
# Calculate the number of samples needed to triple the size of the minority class
desired_minority_samples = len(df_minority) * 3

# Upsample the minority class
df_minority_upsampled = resample(df_minority, 
                                 replace=True,     
                                 n_samples=desired_minority_samples)
df_upsampled = pd.concat([df_majority, df_minority_upsampled])

# Selecting features
features = ['Hour', 'Day_of_week', 'Length']
X = df_upsampled[features]
y = df_upsampled['Is_Handshake']

# Split the upsampled dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train a random forest classifier
rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train, y_train)

# Predict probabilities for the positive class
y_proba = rf_classifier.predict_proba(X_test)[:, 1]

# Predictions
y_pred = rf_classifier.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Feature Importance
feature_importances = rf_classifier.feature_importances_
print("Feature Importance:")
for name, importance in zip(features, feature_importances):
    print(name, ":", importance)

# Calculate ROC AUC score
roc_auc = roc_auc_score(y_test, y_proba)
print("ROC AUC Score:", roc_auc)

# Plot ROC curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")


# Learning Curve
train_sizes, train_scores, test_scores = learning_curve(
    rf_classifier, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10), scoring='accuracy')

plt.figure(figsize=(8, 6))
plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label='Training score')
plt.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', label='Cross-validation score')
plt.title('Learning Curve')
plt.xlabel('Training examples')
plt.ylabel('Score')
plt.legend(loc='best')
plt.grid()

# Residual Plot
residuals = y_test - y_proba
plt.figure(figsize=(8, 6))
plt.scatter(y_proba, residuals)
plt.axhline(y=0, color='red', linestyle='--')
plt.title('Residual Plot')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')

# Group by day and hour, and count the number of handshakes
handshake_counts = df.groupby(['Day_of_week', 'Hour']).size().reset_index(name='Count')

# Plot scatter plot with counts and labels
plt.figure(figsize=(10, 6))
scatter = plt.scatter(handshake_counts['Day_of_week'], handshake_counts['Hour'], s=handshake_counts['Count']*1.5, alpha=0.5, color='limegreen')

plt.title('Handshake Distribution by Day and Hour', fontsize=16)
plt.xlabel('Day of Week', fontsize=14)
plt.ylabel('Hour of Day', fontsize=14)
plt.xticks(ticks=range(7), labels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
plt.yticks(range(24), [str(i%12) + (' AM' if i < 12 else ' PM') for i in range(24)])  # Convert 24-hour format to 12-hour format
plt.grid(True)

# Feature Importance
plt.subplot(1, 2, 2)
bar_color = 'limegreen'  # Bright green color for bars

plt.bar(features, feature_importances, color=bar_color)
plt.title('Feature Importance', color='black')  # Title color white
plt.xlabel('Feature', color='black')  # X-axis label color white
plt.ylabel('Importance', color='black')  # Y-axis label color white
plt.xticks(color='black')  # X-axis tick labels color white
plt.yticks(color='black')  # Y-axis tick labels color white
plt.gca().set_facecolor('black')  # Set background color to black
plt.grid(axis='y', linestyle='--', color='gray')  # Add gridlines
plt.tight_layout()

# Show all plots
plt.show()
