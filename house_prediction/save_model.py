import joblib
from sklearn.ensemble import RandomForestRegressor

# Best model (jo aapke result se best aaya)
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

# IMPORTANT: Model ko train karo (FAST)
# Training yahan lightweight hai
# (Dataset chhota hai, seconds me ho jaayega)
import pandas as pd
dataset = pd.read_csv("Housing.csv")

dataset.drop("furnishingstatus", axis=1, inplace=True)
dataset = pd.get_dummies(dataset, drop_first=True)

X = dataset.drop("price", axis=1)
y = dataset["price"]

model.fit(X, y)

joblib.dump(model, "house_price_model.pkl")
print("house_price_model.pkl CREATED SUCCESSFULLY")
