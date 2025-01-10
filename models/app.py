import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu

# Set page configuration
st.set_page_config(
    page_title="Prediction of Disease Outbreak",
    layout="wide",
    page_icon="🧑‍⚕️"
)

# Get the working directory of the script
working_dir = os.path.dirname(os.path.abspath(__file__))

# Load models with error handling
try:
    diabetes_model = pickle.load(open('models\diabetes_model.sav', 'rb'))
    heart_model = pickle.load(open( 'models\heart_disease_model.sav', 'rb'))
    parkinsons_model = pickle.load(open( 'models\parkinsons_model.sav', 'rb'))
except FileNotFoundError as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# Sidebar for navigation
with st.sidebar:
    selected = option_menu(
        'Multiple Disease Prediction System',
        ['Diabetes Prediction', 'Heart Disease Prediction', 'Parkinsons Prediction'],
        menu_icon='hospital-fill',
        icons=['activity', 'heart', 'person'],
        default_index=0
    )

# Diabetes Prediction Page
if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction using ML')

    # Input fields for user data
    col1, col2, col3 = st.columns(3)
    with col1:
        pregnancies = st.text_input('Number of Pregnancies')
    with col2:
        glucose = st.text_input('Glucose Level')
    with col3:
        blood_pressure = st.text_input('Blood Pressure Value')
    with col1:
        skin_thickness = st.text_input('Skin Thickness Value')
    with col2:
        insulin = st.text_input('Insulin Level')
    with col3:
        bmi = st.text_input('BMI Value')
    with col1:
        diabetes_pedigree_function = st.text_input('Diabetes Pedigree Function Value')
    with col2:
        age = st.text_input('Age of the Person')

    # Prediction logic
    if st.button('Diabetes Test Result'):
        try:
            user_input = [float(pregnancies), float(glucose), float(blood_pressure), float(skin_thickness),
                          float(insulin), float(bmi), float(diabetes_pedigree_function), float(age)]
            diab_prediction = diabetes_model.predict([user_input])
            diab_diagnosis = 'The person is diabetic' if diab_prediction[0] == 1 else 'The person is not diabetic'
            st.success(diab_diagnosis)
        except ValueError:
            st.error("Please enter valid numeric values for all fields.")

# Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction using ML')

    # Input fields for user data
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.text_input('Age')
    with col2:
        sex = st.text_input('Sex (0 = Female, 1 = Male)')
    with col3:
        cp = st.text_input('Chest Pain Type')
    with col1:
        trestbps = st.text_input('Resting Blood Pressure')
    with col2:
        chol = st.text_input('Serum Cholesterol in mg/dl')
    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)')
    with col1:
        restecg = st.text_input('Resting Electrocardiographic Results')
    with col2:
        thalach = st.text_input('Maximum Heart Rate Achieved')
    with col3:
        exang = st.text_input('Exercise Induced Angina (1 = Yes, 0 = No)')
    with col1:
        oldpeak = st.text_input('ST Depression Induced by Exercise')
    with col2:
        slope = st.text_input('Slope of the Peak Exercise ST Segment')
    with col3:
        ca = st.text_input('Number of Major Vessels (0–3) Colored by Fluoroscopy')
    with col1:
        thal = st.text_input('Thalassemia (0 = Normal, 1 = Fixed Defect, 2 = Reversible Defect)')

    # Prediction logic
    if st.button('Heart Disease Test Result'):
        try:
            user_input = [float(x) for x in [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]
            heart_prediction = heart_model.predict([user_input])
            heart_diagnosis = 'The person has heart disease' if heart_prediction[0] == 1 else 'The person does not have heart disease'
            st.success(heart_diagnosis)
        except ValueError:
            st.error("Please enter valid numeric values for all fields.")

# Parkinson's Prediction Page
if selected == 'Parkinsons Prediction':
    st.title("Parkinson's Disease Prediction using ML")

    # Input fields for user data
    cols = st.columns(5)
    inputs = [
        'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 
        'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 
        'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR', 
        'HNR', 'RPDE', 'DFA', 'Spread1', 'Spread2', 'D2', 'PPE'
    ]
    user_inputs = []

    for i, field in enumerate(inputs):
        with cols[i % 5]:
            user_inputs.append(st.text_input(field))

    # Prediction logic
    if st.button("Parkinson's Test Result"):
        try:
            user_input = [float(x) for x in user_inputs]
            parkinsons_prediction = parkinsons_model.predict([user_input])
            parkinsons_diagnosis = "The person has Parkinson's disease" if parkinsons_prediction[0] == 1 else "The person does not have Parkinson's disease"
            st.success(parkinsons_diagnosis)
        except ValueError:
            st.error("Please enter valid numeric values for all fields.")














