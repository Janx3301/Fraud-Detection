
from flask import Flask, render_template, url_for, request, redirect, session
import sqlite3
import os
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score, accuracy_score
from scipy.stats import ttest_ind
from sklearn.model_selection import train_test_split
from math import radians, sin, cos, sqrt, atan2
from sklearn.metrics import f1_score
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
sns.set(font_scale=1.3)
import joblib
import bcrypt

app = Flask(__name__)
app.secret_key = 'your_secret_key'

def create_table():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()

    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                 first_name TEXT NOT NULL,
                 last_name TEXT NOT NULL,
                 email TEXT NOT NULL,
                 password TEXT NOT NULL)''')

    conn.commit()
    conn.close()

app.config['UPLOAD_FOLDER'] = 'uploads'

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        conn = sqlite3.connect('users.db')
        c = conn.cursor()

        c.execute("SELECT * FROM users WHERE email=?", (email,))
        user = c.fetchone()

        if user and bcrypt.checkpw(password.encode('utf-8'), user[4].encode('utf-8')):
            session['username'] = user[1]  # Store the username in the session
            return redirect(url_for('fraud_detection'))
        else:
            error_message = "Invalid email or password."
            return render_template('login.html', error_message=error_message)

    return render_template('login.html')


from flask_mail import Mail, Message
import random
import string
import time

app.config['MAIL_SERVER']='smtp.gmail.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USERNAME'] = 'janx3301@gmail.com'
app.config['MAIL_PASSWORD'] = 'vhky qpbv dmyl kpln'
app.config['MAIL_USE_TLS'] = False
app.config['MAIL_USE_SSL'] = True

mail = Mail(app)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        first_name = request.form.get('first-name')  
        last_name = request.form.get('last-name')  
        email = request.form.get('email')
        password = request.form.get('password')
        if not password:
            error_message = "Password is required."
            return render_template('register.html', error_message=error_message)

        conn = sqlite3.connect('users.db')
        c = conn.cursor()

        c.execute("SELECT * FROM users WHERE email=?", (email,))
        existing_user = c.fetchone()

        if existing_user:
            error_message = "Email already exists. Please choose a different email."
            return render_template('register.html', error_message=error_message)

        # Hash and salt the password
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

        # Generate OTP
        otp = ''.join(random.choices(string.digits, k=6))  

        # Save the user data and OTP in the session
        session['otp'] = otp
        session['first_name'] = first_name
        session['last_name'] = last_name
        session['email'] = email
        session['hashed_password'] = hashed_password.decode('utf-8')

        # Send OTP
        msg = Message('OTP', sender = 'janx3301@gmail.com', recipients = [email])
        msg.body = 'Your OTP is ' + otp
        mail.send(msg)

        return render_template('verify.html')

    return render_template('register.html')


@app.route('/verify', methods=['GET', 'POST'])
def verify():
    if request.method == 'POST':
        user_otp = request.form['otp']

        if 'otp' in session and session['otp'] == user_otp:
            conn = sqlite3.connect('users.db')
            c = conn.cursor()

            c.execute("INSERT INTO users (first_name, last_name, email, password) VALUES (?, ?, ?, ?)",
                      (session['first_name'], session['last_name'], session['email'], session['hashed_password']))

            conn.commit()
            conn.close()

            # Clear the session
            session.pop('otp', None)
            session.pop('first_name', None)
            session.pop('last_name', None)
            session.pop('email', None)
            session.pop('hashed_password', None)

            return redirect('/login')

        else:
            error_message = "Invalid OTP. Please try again."
            return render_template('verify.html', error_message=error_message)

    # Handle OTP resend logic
    if request.method == 'GET' and 'email' in session:
        email = session['email']
        
        # Check if enough time has passed since the last OTP was sent (30 seconds cooldown)
        if 'last_resend_time' in session and time.time() - session['last_resend_time'] < 30:
            cooldown_remaining = int(30 - (time.time() - session['last_resend_time']))
            return render_template('verify.html', cooldown_remaining=cooldown_remaining)

        # Check if OTP has already been sent in the current session
        if 'otp_sent' in session and session['otp_sent']:
            return render_template('verify.html', error_message="OTP has already been sent. Please check your email.")
            
        # Generate a new OTP
        new_otp = ''.join(random.choices(string.digits, k=6))
        
        # Update the session with the new OTP and the current time
        session['otp'] = new_otp
        session['last_resend_time'] = time.time()

        # Send the new OTP
        msg = Message('New OTP', sender='your_email@gmail.com', recipients=[email])
        msg.body = 'Your new OTP is ' + new_otp
        mail.send(msg)

        return render_template('verify.html', error_message="New OTP sent. Please check your email.")

    else:
        return render_template('verify.html')  # Render the verification form


@app.route('/fraud-detection')
def fraud_detection():
    if 'username' in session:
        username = session['username']
        return render_template('fraud_detection.html', username=username)
    else:
        return redirect('/login')

@app.route('/logout', methods=['GET', 'POST'])
def logout():
    if request.method == 'POST':
        session.pop('username', None)
        return redirect('/')
    return redirect('/')



@app.route('/predict', methods=['POST'])
def predict():
    if 'csv-file' not in request.files:
        return redirect(url_for('fraud_detection'))

    file = request.files['csv-file']
    if file.filename == '':
        return redirect(url_for('fraud_detection'))

    # Save the uploaded file
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Load the ensemble model
    ensemble_model_filename = 'ensemble_model.pkl'
    loaded_ensemble_model = joblib.load(ensemble_model_filename)

    # Read the CSV file
    data = pd.read_csv(file_path).drop('Unnamed: 0', axis=1)
    # Read the CSV file to extract trans_num values
    trans_num_data = pd.read_csv(file_path, usecols=['trans_num'])
    trans_num_values = trans_num_data['trans_num'].values.tolist()

        
    data['full_name'] = data['first'] + ' ' + data['last']
    data.drop(['first', 'last', 'trans_num', 'unix_time'], axis=1, inplace=True)
    data['trans_date_trans_time'] = pd.to_datetime(data['trans_date_trans_time'])
    data['year'] = data['trans_date_trans_time'].dt.year
    data['month'] = data['trans_date_trans_time'].dt.month
    data['day'] = data['trans_date_trans_time'].dt.day
    data['hour'] = data['trans_date_trans_time'].dt.hour
    data['day_of_week'] = data['trans_date_trans_time'].dt.day_name()
    data['is_weekend'] = (data['trans_date_trans_time'].dt.weekday // 5).map({0: 'No', 1: 'Yes'})
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    data['month'] = data['month'].apply(lambda x: months[x - 1])
    def get_credit_card_type(card_number):
        card_number = str(card_number)
        if card_number[0] == '4':
            return 'Visa'
        elif card_number[0] == '5' or card_number[0] == '2':
            return 'MasterCard'
        elif card_number[0] == '3':
            return 'American Express'
        elif card_number[0] == '6':
            return 'Discover'
        else:
            return 'Unknown'
    data['cc_type'] = data['cc_num'].apply(get_credit_card_type)
    freq = data['cc_num'].value_counts()
    data['category'] = data['category'].apply(lambda x: x.title().replace('_', ' '))
    data['gender'] = data['gender'].map({'F': 'Female', 'M': 'Male'})
    data['address'] = data['street'] + ', ' + data['city'] + ', ' + data['state'] + ' ' + data['zip'].apply(str)
    def haversine(lat1, lon1, lat2, lon2):
        # Radius of the Earth in kilometers
        R = 6371
        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)
        a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
        distance = 2 * R * atan2(sqrt(a), sqrt(1 - a))
        return distance
    data['distance'] = data.apply(lambda row: haversine(row['lat'], row['long'], row['merch_lat'], row['merch_long']), axis=1)
    data.drop(['lat', 'long', 'merch_lat', 'merch_long'], axis=1, inplace=True)
    data['dob'] = pd.to_datetime(data['dob'])
    data['age'] = np.ceil((data['trans_date_trans_time'] - data['dob']).dt.days / 365).astype(int)
    data.drop('trans_date_trans_time', axis=1, inplace=True)
    data.drop('dob', axis=1, inplace=True)
    def age_interval(age):
        if 14 <= age < 30:
            return '14 to 30'
        elif 30 <= age < 45:
            return '30 to 45'
        elif 45 <= age < 60:
            return '45 to 60'
        elif 60 <= age < 75:
            return '60 to 75'
        else:
            return 'Older than 75'
        
    data['age_interval'] = data['age'].apply(age_interval)
    data.drop(['cc_num', 'merchant', 'street', 'city', 'state', 'zip', 'year'], axis=1, inplace=True)
    data['gender'] = data['gender'].map({'Female': 0, 'Male': 1})
    data['is_weekend'] = data['is_weekend'].map({'No': 0, 'Yes': 1})
    skewed = ['amt', 'city_pop']
    data[skewed] = data[skewed].apply(np.log1p)
    # Load the saved encoder
    loaded_encoder = joblib.load('encoder_model.joblib')
    columns_to_encode = ['category', 'job', 'address', 'month', 'day_of_week', 'cc_type', 'full_name', 'age_interval']
    data[columns_to_encode] = loaded_encoder.transform(data[columns_to_encode])

    less_info = ['is_weekend', 'cc_type', 'gender', 'day_of_week']
    data.drop(less_info, axis=1, inplace=True)

    
    # Perform prediction using the model
    predictions = loaded_ensemble_model.predict(data)

    # Pass trans_num_values and corresponding predictions to the template
    return render_template('single_prediction.html', predictions=zip(trans_num_values, predictions))


@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')


@app.route('/results')
def results():
    return render_template('results.html')


@app.route('/results2')
def results2():
    return render_template('results2.html')

#importing required libraries

from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn import metrics 
import warnings
import pickle
warnings.filterwarnings('ignore')
from feature import FeatureExtraction

file = open("model.pkl","rb")
log = pickle.load(file)
file.close()

@app.route("/index", methods=["GET", "POST"])
def index():
    if request.method == "POST":

        url = request.form["url"]
        obj = FeatureExtraction(url)
        x = np.array(obj.getFeaturesList()).reshape(1,30) 

        y_pred =log.predict(x)[0]
        #1 is safe       
        #-1 is unsafe
        y_pro_phishing = log.predict_proba(x)[0,0]
        y_pro_non_phishing = log.predict_proba(x)[0,1]
        # if(y_pred ==1 ):
        pred = "It is {0:.2f} % safe to go ".format(y_pro_phishing*100)
        return render_template('index.html',xx =round(y_pro_non_phishing,2),url=url )
    return render_template("index.html", xx =-1)


if __name__ == '__main__':
    create_table()
    app.run(debug=True)
