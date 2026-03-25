import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("Training.csv")

# Separate input and output
X = data.drop("prognosis", axis=1)
y = data["prognosis"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Check accuracy
accuracy = accuracy_score(y_test, predictions)
print("Model Accuracy:", accuracy)




# -------- Disease Prediction from User Input --------

symptoms_list = list(X.columns)

print("\nAvailable symptoms:")
print(symptoms_list)

user_input = input("\nEnter symptoms separated by comma: ").lower()
user_symptoms = user_input.split(",")

# create input vector
input_vector = [0] * len(symptoms_list)

for symptom in user_symptoms:
    symptom = symptom.strip()
    if symptom in symptoms_list:
        index = symptoms_list.index(symptom)
        input_vector[index] = 1
    else:
        print(symptom, "not found in dataset")

# prediction
prediction = model.predict([input_vector])

print("\nPredicted Disease:", prediction[0])