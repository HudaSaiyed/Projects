import streamlit as st
import numpy as np
import pickle
from PIL import Image
# import sys
# sys.path.append(r"C:\Users\91876\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\site-packages")
# sys.path.insert(1,  "C:/Users/91876/AppData/Local/Packages/PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0/LocalCache/local-packages/Python313/site-packages")
# from streamlit_option_menu import option_menu


# load model
loaded_db_model=pickle.load(open("C:/Users/91876/Desktop/project8/project/diabetes.sav", 'rb'))
loaded_heart_model=pickle.load(open("C:/Users/91876/Desktop/project8/project/heart.sav", 'rb'))
loaded_parkinson_model=pickle.load(open("C:/Users/91876/Desktop/project8/project/parkinson.sav", 'rb'))
loaded_liver_model=pickle.load(open("C:/Users/91876/Desktop/project8/project/liver.sav", 'rb'))

# create a diabetes func
def diabetes_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    prediction = loaded_db_model.predict(input_data_reshaped)
    print(prediction)
    if (prediction[0] == 0):
      return 'The person is not diabetic'
    else:
      return 'The person is diabetic'

# create a heart funct
def heart_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    prediction = loaded_heart_model.predict(input_data_reshaped)
    if (prediction[0] == 0):
      return 'The person is does not have a Heart Disease'
    else:
      return 'The person is have heart disease'

# parkinson
def parkinson_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    prediction = loaded_parkinson_model.predict(input_data_reshaped)
    if (prediction[0] == 0):
      return "The Person does not have Parkinsons Disease"
    else:
      return "The Person has Parkinsons"

# liver
def liver_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    prediction = loaded_liver_model.predict(input_data_reshaped)
    if (prediction[0] == 0):
      return "The Person does not have Liver issues"
    else:
      return "The Person has liver issues"

# bmi
def calculate_bmi(weight, height):
    bmi = weight / (height / 100) ** 2
    return bmi

def interpret_bmi(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif 18.5 <= bmi < 24.9:
        return "Normal Weight"
    elif 25 <= bmi < 29.9:
        return "Overweight"
    else:
        return "Obese"


def main():
        st.markdown("""
        <style>
        div[class*="stRadio"] > label > div[data-testid="stMarkdownContainer"] > p {
        font-size: 25px;
        }
        </style>
        """, unsafe_allow_html=True)

        st.markdown("""
        <style>
    
        /* Radio option styling */
        section[data-testid="stSidebar"] div[data-testid="stRadio"] label {
            font-size: 25px;
             margin-bottom: 25px; 
            
        }
    
        /* Padding between radio buttons */
        section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > div {
           
        }
        </style>
        """, unsafe_allow_html=True)

        
        with st.sidebar:
        # Display styled title
            # st.markdown('<div class="title-text">Disease Diagnosis and Recommendation System</div>', unsafe_allow_html=True)
            
            # Styled radio options
            selected = st.radio(
            "Disease Diagnosis and Recommendation System",
            [
                '🏠 Home',
                '🩸 Diabetes Prediction',
                '🫀 Heart Disease Prediction',
                '🧬 Parkinsons Prediction',
                '🩺 Liver Functioning',
                '⚖️ BMI CALCULATOR'
            ],
            index=0
            )
    
        if selected=="🏠 Home":
            st.title("Medical Diagnosis Support Sytem")
            image = Image.open('C:/Users/91876/Desktop/project8/project/front3.jpg')
            st.image(image, width=650)
            st.subheader("ℹ️ About This Project")
            st.markdown("""
            This project is developed using Python and Streamlit.  
            It leverages pre-trained machine learning models for disease prediction based on publicly available datasets.  
            """)
            st.info('Designed for educational and awareness purposes only.')
                    
        # diabets section
        if selected =="🩸 Diabetes Prediction":
            st.title("Diabetes Prediction")
            st.markdown("""
            #### 🩸 Enter Patient Information for Diabetes Prediction
            """)
            # col 
            col1, col2 = st.columns(2)
            Pregnancies = col1.text_input('Number of Pregnancies')
            Glucose = col2.text_input('Glucose Level (mg/dL)')
            Bp = col1.text_input('Blood Pressure (mm Hg)')
            SkinThickness = col2.text_input('Skin Thickness (mm)')
            Insulin = col1.text_input('Insulin Level (mu U/ml)')
            BMI = col2.text_input('Body Mass Index (BMI)')
            DiaPedi = col1.text_input('Diabetes Pedigree Function')
            Age = col2.text_input('Age (in years)')
            
            diabetes_diagnosis=''
            # creating a button
            if st.button('Diabetes predict'):
                if bool(Pregnancies)==False:
                    st.error("invalid input")
                else:
                    diabetes_diagnosis = diabetes_prediction([float(Pregnancies),float(Glucose), float(Bp), float(SkinThickness),float(Insulin), float(BMI), float(DiaPedi),float(Age)])
                    # out of if with success of var 'diagnosis'
                    st.success(diabetes_diagnosis)

        # heart section
        if selected=="🫀 Heart Disease Prediction":
            st.title('Heart Disease Prediction')
            st.markdown("""
            #### 🫀 Enter Patient Information for Heart Disease Prediction
            """)

            col1, col2=st.columns(2)           
            age = col1.text_input('Age')
            sex = col2.selectbox('Sex',['Male','Female'])
            col1, col2, col3=st.columns(3)
            cp = col1.text_input('Chest Pain type(0,1,2,3)')
            trestbps = col2.text_input('Resting Blood Pressure')
            chol = col3.text_input('Serum Cholestoral in mg/dl')
            fbs = col1.text_input('Fasting Blood Sugar > 120 mg/dl')
            restecg = col2.text_input('Resting Electrocardiographic results')
            thalach = col3.text_input('Maximum Heart Rate achieved')
            exang = col1.text_input('Exercise Induced Angina')
            oldpeak = col2.text_input('ST depression induced by exercise')
            slope = col3.text_input('Slope of the peak exercise ST segment')
            ca = col1.text_input('Major vessels colored by flourosopy')
            thal = col2.text_input('Results of Nuclear Stress Test(0,1,2)')

            heart_diagnosis=''
            # creating a button
            if st.button('Predict'):
                if bool(age)==False:
                    st.error("invalid input")
                else:
                    sex_mapping = {'Male': 1, 'Female': 0}
                    sex_numeric = sex_mapping[sex]
                    heart_diagnosis = heart_prediction([float(age),float(sex_numeric),float(cp),float(trestbps),float(chol),float(fbs), float(restecg),float(thalach),float(exang), float(oldpeak),float(slope),float(ca),float(thal)])
                    st.success(heart_diagnosis)

        if selected=="🧬 Parkinsons Prediction":
            st.title("Parkinsons Prediction")
            st.markdown("""
            #### 🧬 Enter Patient Information of Biomedical Voice Measurement Test
            """)
            col1, col2, col3, col4, col5,col6 = st.columns(6)
            
            fo = col1.text_input('Fo(Hz)')
            fhi = col2.text_input('Fhi(Hz)')
            flo = col3.text_input('Flo(Hz)')
            Jitter_percent = col4.text_input('Jitter(%)')
            Jitter_Abs = col5.text_input('Jitter(Abs)')
            RAP = col1.text_input('RAP')
            PPQ = col2.text_input('PPQ')
            DDP = col3.text_input('DDP')
            Shimmer = col4.text_input('Shimmer')
            Shimmer_dB = col5.text_input('Shimmer(dB)')
            APQ3 = col1.text_input('APQ3')
            APQ5 = col2.text_input('APQ5')
            APQ = col3.text_input('APQ')
            DDA = col4.text_input('DDA')
            NHR = col5.text_input('NHR')
            HNR = col1.text_input('HNR')
            RPDE = col2.text_input('RPDE')
            DFA = col3.text_input('DFA')
            spread1 = col4.text_input('spread1')
            spread2 = col5.text_input('spread2')
            D2 = col1.text_input('D2')
            PPE = col2.text_input('PPE')

            parkinson_diagnosis=''
            if st.button('Predict'):
                if bool(fo)==False:
                    st.error("invalid input")
                else:
                    parkinson_diagnosis = parkinson_prediction([float(fo),float(fhi), float(flo), float(Jitter_percent),float(Jitter_percent), float(RAP), float(PPQ),float(DDP),float(Shimmer), float(Shimmer_dB),float(APQ3),float(APQ5),float(APQ),float(DDA),float(NHR),float(HNR),float(RPDE),float(DFA),float(spread1),float(spread2),float(D2),float(PPE)])
                    # out of if with success of var 'diagnosis'
                    st.success(parkinson_diagnosis)
           
            

        # liver section
        if selected=="🩺 Liver Functioning":
            st.title('Liver Functioning')
            st.markdown("""
            #### 🩺 Enter Clinical Information to Assess Liver Health
            """)
            
            col1, col2=st.columns(2)           
            age = col1.text_input('Age')
            sex = col2.selectbox('Gender',['Male','Female'])
            col1, col2, col3=st.columns(3)
            Total_Bilirubin = col1.text_input('Total_Bilirubin')
            Direct_Bilirubin = col2.text_input('Direct_Bilirubin')
            Alkaline_Phosphotase = col3.text_input('Alkaline_Phosphotase')
            Alamine_Aminotransferase = col1.text_input('Alamine_Aminotransferase')
            Aspartate_Aminotransferase = col2.text_input('Aspartate_Aminotransferase')
            Total_Protiens = col3.text_input('Total_Protiens')
            Albumin = col1.text_input('Albumin')
            Albumin_and_Globulin_Ratio = col2.text_input('Albumin_and_Globulin_Ratio')
            
            liver_diagnosis=''
            # creating a button
            if st.button('Predict'):
                if bool(age)==False:
                    st.error("invalid input")
                else:
                    sex_mapping = {'Male': 1, 'Female': 0}
                    sex_numeric = sex_mapping[sex]
                    
                    liver_diagnosis = liver_prediction([float(age),float(sex_numeric), float(Total_Bilirubin), float(Direct_Bilirubin),float(Alkaline_Phosphotase), float(Alamine_Aminotransferase), float(Aspartate_Aminotransferase),float(Total_Protiens),float(Albumin), float(oldpeak),float(slope),float(ca),float(thal)])
                    # out of if with success of var 'diagnosis'
                    st.success(liver_diagnosis)
            
        # BMI section
        if (selected == '⚖️ BMI CALCULATOR'):
            st.title("BMI CALCULATOR")
    
            st.write("Body Mass Index (BMI) is a measure of body fat based on height and weight.")
            st.write("Use this calculator to find out your BMI category.")
            col1,col2 = st.columns([2,1])
            with col1:
                weight = st.text_input("Enter your weight (in kilograms)")
                height = st.text_input("Enter your height (in centimeters)")
            
                if st.button("Calculate BMI"):
                    if bool(weight)==False:
                        st.error("invalid input")
                    else:
                        weight = float(weight)
                        height = float(height)
                        bmi = calculate_bmi(weight, height)
                        category = interpret_bmi(bmi)
                
                        st.write("### Results")
                        st.write(f"Your BMI: {bmi:.2f}")
                        st.write(f"Category: {category}")
            # with col2:
            # image = Image.open('images/bmi.png')
            # st.image(image,width =350)  

if __name__=='__main__':
    main()
