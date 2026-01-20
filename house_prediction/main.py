# First step : Import necessary libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# =========================
# Step 1: Load Dataset
# =========================
dataset = pd.read_csv("Housing.csv")

print(dataset.head(5))
print(dataset.columns)
print(dataset.shape)


# =========================
# Step 2: Data Preprocessing
# =========================
obj = (dataset.dtypes == 'object')
object_cols = list(obj[obj].index)
print("Categorical variables:", len(object_cols))

intt = (dataset.dtypes == "int")
int_Cols = list(intt[intt].index)
print("Integer variables:", len(int_Cols))

fl = (dataset.dtypes == "float")
fl_Col = list(fl[fl].index)
print("FFloat variables:", len(fl_Col))

print(dataset.isnull().sum())


# =========================
# Step 3: Exploratory Data Analysis
# =========================
numerical_dataset = dataset.select_dtypes(include=["number"])

plt.figure(figsize=(12,6))
sns.heatmap(
    numerical_dataset.corr(),
    cmap='BrBG',
    annot=True,
    fmt='.2f',
    linewidths=2
)
plt.show()


unique_values = []
for col in object_cols:
    unique_values.append(dataset[col].unique().size)

plt.figure(figsize=(10,6))
plt.title('No. Unique values of Categorical Features')
plt.xticks(rotation=90)
sns.barplot(x=object_cols, y=unique_values)
plt.show()


# =========================
# Step 4: Data Cleaning & Encoding
# =========================
dataset.drop("furnishingstatus", axis=1, inplace=True)
dataset = pd.get_dummies(dataset, drop_first=True)


# =========================
# Step 5: Train-Test Split
# =========================
X = dataset.drop("price", axis=1)
y = dataset["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# =========================
# Step 6: Models & Hyperparameter Tuning
# =========================
models = {
    "Linear Regression": {
        "model": LinearRegression(),
        "params": {}
    },
    "Ridge": {
        "model": Ridge(),
        "params": {
            "alpha": [0.1, 1, 10, 50]
        }
    },
    "Lasso": {
        "model": Lasso(),
        "params": {
            "alpha": [0.01, 0.1, 1, 10]
        }
    },
    "Random Forest": {
        "model": RandomForestRegressor(),
        "params": {
            "n_estimators": [100, 200],
            "max_depth": [None, 10, 20]
        }
    },
    "Gradient Boosting": {
        "model": GradientBoostingRegressor(),
        "params": {
            "n_estimators": [100, 200],
            "learning_rate": [0.05, 0.1],
            "max_depth": [3, 5, 10]
        }
    }
}


results = []

for name, config in models.items():
    print(f"\nTraining {name}...")
    
    grid = GridSearchCV(
        config["model"],
        config["params"],
        cv=5,
        scoring="r2",
        n_jobs=-1
    )

    print("Starting GridSearch...")
    grid.fit(X_train, y_train)
    print("GridSearch finished")

    results.append({
        "model": name,
        "best_score": grid.best_score_,
        "best_params": grid.best_params_,
        "estimator": grid.best_estimator_
    })


# =========================
# Step 7: Model Comparison
# =========================
results_df = pd.DataFrame(results)
print("\nModel Comparison Results:")
print(results_df)


# =========================
# Step 8: Final Model Evaluation
# =========================
best_model = max(results, key=lambda x: x["best_score"])["estimator"]

y_pred = best_model.predict(X_test)

print("\nFinal Model Performance:")
print("R2:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

print("BEFORE SAVE")

import joblib
joblib.dump(best_model, "house_price_model.pkl")

print("AFTER SAVE")
