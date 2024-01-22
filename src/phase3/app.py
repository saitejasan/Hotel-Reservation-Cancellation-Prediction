from flask import Flask, request, jsonify,render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the logistic regression model
with open('logistic_regression_model.pkl', 'rb') as file:
    model = pickle.load(file)


@app.route('/')
def display_page():
    return render_template('index.html')

@app.route('/booking_status',methods=['POST'])
def get_data():
    adults=""
    children=""
    weekend_nights="",
    week_nights="",
    meal_plan="",
    car_parking="",
    room_type="",
    lead_time="",
    market_seg="",
    repeated_guest="",
    cancellations="",
    non_cancellations="",
    avg_price="",
    special_requests=""
    if request.method == 'POST':

        adults = request.form.get('adults')
        children = request.form.get('children')
        weekend_nights = request.form.get('weekend_nights')
        week_nights = request.form.get('week_nights')
        meal_plan = request.form.get('meal_plan')
        car_parking = request.form.get('car_parking')
        room_type = request.form.get('room_type')
        lead_time = request.form.get('lead_time')
        market_seg = request.form.get('market_seg')
        repeated_guest = request.form.get('repeated_guest')
        cancellations = request.form.get('cancellations')
        non_cancellations = request.form.get('non_cancellations')
        avg_price = request.form.get('avg_price')
        special_requests = request.form.get('special_requests')


    features = pd.DataFrame([[adults,children,weekend_nights,week_nights,meal_plan,car_parking,room_type,lead_time,market_seg,repeated_guest,cancellations,non_cancellations,avg_price,special_requests]])


    prediction = model.predict(features)
    probability = model.predict_proba(features)[:, 1]


    return render_template('booking.html',val=[int(prediction[0]),float(probability[0])])
    
@app.route("/get_vis")
def show_vis():
    return render_template("dashboard.html")

if __name__ == '__main__':
    app.run()

