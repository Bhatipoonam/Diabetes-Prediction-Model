# Diabetes-Prediction-Model
Diabetes prediction models leveraging machine learning are revolutionizing early detection and management.
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score 
from sklearn.preprocessing import StandardScaler

# Load the diabetes dataset
diabetes = pd.read_csv("C:\\Users\\Poonam Bhati\\Downloads\\diabetes.csv")
print(diabetes.head())

# Display dataset shape and information
print("Dataset shape:", diabetes.shape)
print(diabetes.info())

# Check the distribution of the target variable
print(diabetes['Outcome'].value_counts())

# Separate features and target variable
X = diabetes.drop('Outcome', axis=1)
y = diabetes['Outcome']

# Standardize the feature data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.23, random_state=1)

# Initialize and train the SVM model
svc = svm.SVC(kernel='linear')
svc.fit(X_train, y_train)

# Make predictions on the training and testing sets
train_y_pred = svc.predict(X_train) 
test_y_pred = svc.predict(X_test)

# Print the lengths of the datasets and predictions
print(f"y_train length: {len(y_train)}, train_y_pred length: {len(train_y_pred)}")
print(f"y_test length: {len(y_test)}, test_y_pred length: {len(test_y_pred)}")

# Calculate and print accuracy scores
train_accuracy = accuracy_score(y_train, train_y_pred)
test_accuracy = accuracy_score(y_test, test_y_pred)

print("Train set accuracy:", train_accuracy)
print("Test set accuracy:", test_accuracy)

# Example input data for prediction
input_data = (5, 116, 74, 0, 0, 25.6, 0.201, 30)  
np_array_data = np.asarray(input_data).reshape(1, -1)

# Standardize the input data
input_data_scaled = scaler.transform(np_array_data)

# Make a prediction for the input data
prediction = svc.predict(input_data_scaled)

# Print the prediction result
if prediction[0] == 1:
    print("This person has diabetes.")
else:
    print("This person does not have diabetes.")

# Display the first five rows of the dataset
print(diabetes.head(5))
