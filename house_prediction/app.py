from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(
    __name__,
    template_folder="house_price_web/templates",
    static_folder="house_price_web/static"
)

# Load trained ML model
model = joblib.load("house_price_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    print("PREDICT ROUTE HIT ")
    try:
        area = float(request.form["area"])
        bedrooms = int(request.form["bedrooms"])
        bathrooms = int(request.form["bathrooms"])
        stories = int(request.form["stories"])
        parking = int(request.form["parking"])

        mainroad = int(request.form["mainroad"])
        guestroom = int(request.form["guestroom"])
        basement = int(request.form["basement"])
        airconditioning = int(request.form["airconditioning"])

        # agar form me nahi hai to default 0 rakho
        hotwaterheating = int(request.form.get("hotwaterheating", 0))
        prefarea = int(request.form.get("prefarea", 0))
        furnishingstatus = int(request.form.get("furnishingstatus", 0))

        input_data = np.array([[
            area,
            bedrooms,
            bathrooms,
            stories,
            mainroad,
            guestroom,
            basement,
            hotwaterheating,
            airconditioning,
            parking,
            prefarea,
            furnishingstatus
        ]])

        print("INPUT DATA:", input_data)
        print("SHAPE:", input_data.shape)

        prediction = model.predict(input_data)[0]

        return render_template("index.html", prediction=f"Predicted Price: ₹ {prediction:,.2f}")

    except Exception as e:
        print("ERROR:", e)
        return render_template("index.html", prediction=str(e))


if __name__ == "__main__":
    app.run(debug=True)
