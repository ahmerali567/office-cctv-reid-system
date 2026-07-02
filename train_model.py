# train_model.py
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
import os

# Agar models folder nahi hai toh banao
os.makedirs("models", exist_ok=True)

df = pd.read_csv("office_data.csv")
X = df.drop("label", axis=1)
y = df["label"]

model = RandomForestClassifier(
    n_estimators=200, max_depth=15, min_samples_leaf=2,
    class_weight='balanced', random_state=42
)
model.fit(X, y)

joblib.dump(model, "models/office_action_model.pkl")
print(f"✅ Model trained on {len(df)} samples")
print(f"Classes: {y.unique()}")