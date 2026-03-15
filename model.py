import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
from preprocess import load_and_preprocess

df = load_and_preprocess("data/insurance_data.csv")

X = df.drop('Response', axis=1)
y = df['Response']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("==========================================")
print("         MODEL EVALUATION RESULTS        ")
print("==========================================")
print(f"Accuracy Score  : {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

joblib.dump(model, "model/model.pkl")
print("Model saved → model/model.pkl ✅")