import pandas as pd
import joblib
from xgboost import XGBClassifier

print("⚡ Reading data from local CSV...")
# Load local dataset
try:
    df = pd.read_csv("processed_balanced_dataset.csv")
except FileNotFoundError:
    print("❌ Error: 'processed_balanced_dataset.csv' not found.")
    print("   Please ensure the CSV file is in the same folder as this script.")
    exit()

# Prepare Features (X) and Target (y)
y = df['target_is_delayed']
X = df.drop(columns=['target_is_delayed'])

print(f"🔥 Training new XGBoost model on {len(df)} records...")
# Train model locally (Ensures OS compatibility)
model = XGBClassifier(
    n_estimators=100, 
    use_label_encoder=False, 
    eval_metric='logloss',
    random_state=42
)
model.fit(X, y)

print("💾 Saving best_model.pkl (Windows Compatible Version)...")
# Overwrite the old model file
joblib.dump(model, "best_model.pkl")

print("✅ DONE! Model artifact updated. You can run 'streamlit run app.py' now.")