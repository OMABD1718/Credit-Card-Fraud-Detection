import joblib
import os

model_path = "artifacts/final_fraud_detection_bundle.joblib"

print("Checking model path...")

if not os.path.exists(model_path):
    print(f"ERROR: Model not found at {model_path}")
else:
    print("Loading model...")
    bundle = joblib.load(model_path)

    print("Model loaded successfully!")
    print("Bundle keys:", bundle.keys())
    print("Number of features:", len(bundle["features"]))
    print("Model name:", bundle.get("model_name"))