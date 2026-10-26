
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Generate a demonstration student dataset
data = pd.DataFrame({
    "study_hours": [
        1, 2, 3, 4, 5, 6, 7, 8, 2, 3,
        4, 5, 6, 7, 8, 1, 2, 3, 6, 7,
        5, 4, 8, 9, 2, 3, 5, 6, 7, 9
    ],
    "attendance": [
        40, 45, 50, 55, 60, 65, 70, 90, 42, 52,
        58, 62, 68, 75, 85, 35, 48, 53, 72, 80,
        65, 57, 88, 95, 43, 55, 67, 73, 82, 98
    ],
    "previous_score": [
        30, 35, 40, 45, 50, 55, 60, 85, 32, 42,
        48, 52, 58, 65, 80, 25, 38, 43, 68, 75,
        55, 47, 83, 90, 33, 45, 62, 70, 78, 92
    ],
    "result": [
        0, 0, 0, 0, 1, 1, 1, 1, 0, 0,
        0, 1, 1, 1, 1, 0, 0, 0, 1, 1,
        1, 0, 1, 1, 0, 0, 1, 1, 1, 1
    ]
})

# Save the demonstration dataset
data.to_csv("student_results.csv", index=False)

# Separate features and target
X = data.drop("result", axis=1)
y = data["result"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Calculate evaluation metrics
metrics = {
    "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
    "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
    "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
    "f1_score": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
    "training_records": int(len(X_train)),
    "testing_records": int(len(X_test))
}

# Save the trained model
joblib.dump(model, "student_result_model.pkl")

# Save evaluation metrics
with open("metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)

print("Model training completed successfully!")
print("Evaluation metrics:", metrics)
print("Generated: student_result_model.pkl, metrics.json, student_results.csv")
