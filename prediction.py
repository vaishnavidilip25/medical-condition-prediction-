import streamlit as st
import numpy as np
import pickle
import sqlite3
from passlib.hash import pbkdf2_sha256

# Function to load models
def load_model(file_path):
    try:
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error(f"Model file '{file_path}' not found.")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
    return None

# Function to create user table and handle user authentication
def setup_database():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users 
                 (username TEXT PRIMARY KEY, password TEXT)''')
    conn.commit()
    return conn

def create_user(username, password):
    conn = setup_database()
    c = conn.cursor()
    hashed_password = pbkdf2_sha256.hash(password)  # Hash the password
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_password))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False

def authenticate_user(username, password):
    conn = setup_database()
    c = conn.cursor()
    c.execute("SELECT password FROM users WHERE username = ?", (username,))
    hashed_password = c.fetchone()
    if hashed_password:
        if pbkdf2_sha256.verify(password, hashed_password[0]):
            return True
    return False

# Main Function
def main():

    st.markdown(
        """
        <style>
        .main {
            background-image: url('https://i.pinimg.com/originals/4c/98/4e/4c984ef0291409fef0a0942b391f6287.jpg');
            background-size: cover;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.title('Medical Condition Prediction')

    # Initialize session state
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False

    # Display login/register form if not logged in
    if not st.session_state['logged_in']:
        login_register()

    # Display medical options if logged in
    if st.session_state['logged_in']:
        display_medical_options()

# Login/Register Form
def login_register():
    st.sidebar.header('Login / Registration')
    page = st.sidebar.radio('Select Option', ['Login', 'Register'])

    if page == 'Login':
        login_page()
    elif page == 'Register':
        registration_page()

# Login Page
def login_page():
    st.subheader('Login')
    username_input = st.text_input("Username")
    password_input = st.text_input("Password", type="password")
    login_clicked = st.button("Login")

    if login_clicked:
        if username_input and password_input:
            if authenticate_user(username_input, password_input):
                st.session_state['logged_in'] = True
                st.success(f"Logged in as {username_input}")
            else:
                st.error("Invalid username or password. Please try again.")
        else:
            st.warning("Please enter both username and password.")

# Registration Page
def registration_page():
    st.subheader('Registration')
    username_input = st.text_input("Username")
    password_input = st.text_input("Password", type="password")
    register_clicked = st.button("Register")

    if register_clicked:
        if username_input and password_input:
            if create_user(username_input, password_input):
                st.success("Account created successfully! Please proceed to login.")
            else:
                st.warning("Username already exists. Please choose a different one.")
        else:
            st.warning("Please enter both username and password.")


# Display Medical Options
def display_medical_options():
    st.sidebar.header('Medical Options')
    disease_choice = st.sidebar.selectbox('Select Model', ('Diabetes', 'Heart Disease', 'Parkinsons'))

    if disease_choice == 'Diabetes':
        display_diabetes_prediction_form()

    elif disease_choice == 'Heart Disease':
        display_heart_disease_prediction_form()

    elif disease_choice == 'Parkinsons':
        display_parkinsons_prediction_form()

def display_diabetes_prediction_form():
    st.subheader("Diabetes Prediction")
    pregnancies = st.slider("Number of Pregnancies", 0, 17, 1)
    glucose = st.slider("Glucose Level", 0, 200, 100)
    blood_pressure = st.slider("Blood Pressure", 0, 122, 70)
    skin_thickness = st.slider("Skin Thickness", 0, 99, 20)
    insulin = st.slider("Insulin Level", 0, 846, 79)
    bmi = st.slider("BMI", 0.0, 67.1, 31.4)
    dpf = st.slider("Diabetes Pedigree Function", 0.0, 2.42, 0.4)
    age = st.slider("Age", 21, 81, 29)
    if st.button("Predict"):
        input_data = np.array([pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]).reshape(1, -1)
        model = load_model('diabetes_model.sav')
        if model:
            prediction = model.predict(input_data)
            if prediction[0] == 0:
                st.success("The person is not diabetic")
            else:
                st.error("The person is diabetic")

def display_heart_disease_prediction_form():
    st.subheader("Heart Disease Prediction")
    age = st.slider("Age", 29, 77, 55)
    sex = st.radio("Sex", ('Male', 'Female'))
    cp = st.slider("Chest Pain Type", 0, 3, 1)
    trestbps = st.slider("Resting Blood Pressure", 94, 200, 125)
    chol = st.slider("Cholesterol Level", 126, 564, 245)
    fbs = st.radio("Fasting Blood Sugar > 120 mg/dl", ('Yes', 'No'))
    restecg = st.slider("Resting Electrocardiographic Results", 0, 2, 1)
    thalach = st.slider("Maximum Heart Rate Achieved", 71, 202, 150)
    sex_value = 1 if sex == 'Male' else 0
    fbs_value = 1 if fbs == 'Yes' else 0
    input_data = np.array([age, sex_value, cp, trestbps, chol, fbs_value, restecg, thalach], dtype=float).reshape(1, -1)
    if st.button("Predict"):
        model = load_model('heart_disease_model.sav')
        if model:
            prediction = model.predict(input_data)
            if prediction[0] == 0:
                st.success("The Person does not have Heart Disease")
            else:
                st.error("The Person has Heart Disease")

def display_parkinsons_prediction_form():
    st.subheader("Parkinson's Disease Prediction")
    features = [
        "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)", "MDVP:Jitter(Abs)",
        "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP", "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3",
        "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR", "RPDE", "DFA", "spread1", "spread2", "D2", "PPE"
    ]
    input_data = [st.number_input(feature) for feature in features]
    if st.button("Predict"):
        input_data = np.array(input_data).reshape(1, -1)
        model = load_model('parkinsons_model.sav')
        if model:
            prediction = model.predict(input_data)
            if prediction[0] == 0:
                st.success("The Person does not have Parkinson's Disease")
            else:
                st.error("The Person has Parkinson's Disease")


if __name__ == '__main__':
    main()
