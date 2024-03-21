from flask import Flask, render_template, request, Response
import pickle
import numpy as np

app = Flask(__name__)

# Load the model and scaler
model = pickle.load(open("pumpmodel.pkl", "rb"))
scaler = pickle.load(open("pumpscaler.pkl", "rb"))

@app.route("/pump_action", methods=["GET", "POST"])
def prediction():
    if request.method == "POST":
        # Retrieve data from the form
        temp = float(request.form["temp"])
        moisture = float(request.form["moisture"])
        input_data = [temp, moisture]
        input_data = np.array(input_data).reshape(1, -1)
        scale_input_data = scaler.transform(input_data)
        # Make prediction using the model
        act = model.predict(scale_input_data)[0]
        if act==0:
            action = "ON"
        else:
            action = "OFF"
        return render_template("result.html", action=action)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
