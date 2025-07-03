import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load the dataset (you can replace with local path if needed)
df = pd.read_csv('heart.csv')  # Or give full path if outside project folder

# Prepare data
X = df.drop("target", axis=1)
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'heart_disease_model.pkl')
print("Model saved as heart_disease_model.pkl")

