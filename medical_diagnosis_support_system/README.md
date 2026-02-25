**Medical Diagnosis Support System**
This project is a machine learning–based healthcare prediction system designed to assess the risk of multiple diseases using structured clinical data and real-time user input.

**Overview**
The system predicts the likelihood of:
- Diabetes
- Heart Disease
- Liver Disease
- Parkinson’s Disease
Multiple supervised learning models are trained and integrated into a unified Streamlit web application for real-time disease risk prediction.

**Dataset Information**
The project uses structured healthcare datasets including:
- Diabetes Dataset – 768 records, 8 clinical features
- Heart Disease Dataset – 303 records, 13 medical attributes
- Liver Disease Dataset – 583 records, biochemical indicators
- Parkinson’s Dataset – 195 records, 22 voice signal features
Overall, the system is trained on 1,800+ patient records.

**Machine Learning Approach**
- Performed Exploratory Data Analysis (EDA)
- Handled missing values and categorical encoding
- Implemented binary classification models
- Used Logistic Regression and Support Vector Machine (SVM)
- Achieved up to 88% accuracy across datasets
- Evaluated models using Accuracy and Confusion Matrix

**Tech Stack**
Python
Pandas
NumPy
Scikit-learn
Streamlit
Matplotlib / Seaborn
Pickle for model serialization

**Application Features**
Interactive Streamlit-based user interface
Real-time disease prediction
BMI calculator integration
Serialized models for efficient inference
