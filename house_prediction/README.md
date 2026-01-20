# House_predicton
A Machine Learning based House Price Prediction Web Application that predicts house prices based on various features like area, bedrooms, bathrooms, parking, amenities, etc.
The project uses multiple regression models, selects the best-performing model, and deploys it using Flask with an interactive web interface.
**📌 Project Features**

📊 Exploratory Data Analysis (EDA)

🔍 Feature Engineering & Preprocessing
🤖 Multiple ML Models:
Linear Regression
Ridge Regression
Lasso Regression
Random Forest Regressor
Gradient Boosting Regressor
⚙️ Hyperparameter Tuning using GridSearchCV
🏆 Best Model Selection using R² Score
💰 Price formatting in ₹ Lakhs / Crores

🧠 Machine Learning Workflow

Data Collection (Housing.csv)

Data Cleaning & Preprocessing

Feature Encoding

Model Training

Hyperparameter Tuning

Model Comparison

Best Model Selection

Model Saving (.pkl)

Web Deployment
Deployment

🛠️ Technologies Used
Category	Tools
Programming Language	Python
ML Libraries	Scikit-learn, NumPy, Pandas
Visualization	Matplotlib, Seaborn, Chart.js
Web Framework	Flask
Model Saving	Joblib
Frontend	HTML, CSS, JavaScript

house_prediction/
│
├── main.py                # Model training & evaluation
├── save_model.py          # Saves trained model
├── app.py                 # Flask application
├── house_price_model.pkl  # Trained ML model
├── Housing.csv            # Dataset
│
├── house_price_web/
│   ├── templates/
│   │   └── index.html
│   ├── static/
│   │   ├── style.css
│   │   └── script.js
│
├── screenshots/           # Project screenshots
├── requirements.txt
└── README.md
