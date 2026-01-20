import joblib

data = {"test": 123}
joblib.dump(data, "test.pkl")

print("TEST PKL SAVED")
