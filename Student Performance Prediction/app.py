from flask import Flask, request, jsonify, render_template,url_for
import pickle
import numpy as np
import pandas as pd

app=Flask(__name__)

with open('result_model.pkl', 'rb') as model_file:
    result_model=pickle.load(model_file)

with open('result_scaler.pkl', 'rb') as scaler_file:
    result_scaler=pickle.load(scaler_file)

with open('math_model.pkl', 'rb') as model_file:
    math_model=pickle.load(model_file)

with open('math_scaler.pkl', 'rb') as scaler_file:
    math_scaler=pickle.load(scaler_file)

@app.route('/')
def home():
    return render_template('home.html') #here

@app.route('/result')
def result():
    return render_template('result.html')

@app.route('/submit', methods=['POST'])
def submit():
    if request.method=='POST':
        try:
            # get inputs
            gender=request.form.get('gender')
            ethnicity=request.form.get('ethnicity')
            parental_education=request.form.get('parental_level_of_education')
            lunch=request.form.get('lunch')
            test_prep=request.form.get('test_preparation_course')


            # Validate no empty selection
            if '' in [gender, ethnicity, parental_education, lunch, test_prep]:
                return render_template('result.html', prediction_text="Please select all fields.") #here

            # Encode inputs using mapping dictionaries as used in training
            gender_map = {'female':0, 'male':1}
            ethnicity_map = {'group A':0, 'group B':1, 'group C':2, 'group D':3, 'group E':4}
            parental_map = {
                "associate's degree":0,
                "bachelor's degree":1,
                'high school':2,
                "master's degree":3,
                'some college':4,
                'some high school':5
            }
            lunch_map = {'standard':1, 'free/reduced':0}
            test_prep_map = {'none':1, 'completed':0}

            # Convert to numerical values
            data = [
                gender_map[gender],
                ethnicity_map[ethnicity],
                parental_map[parental_education],
                lunch_map[lunch],
                test_prep_map[test_prep]
            ]

            
            final_features = [np.array(data)]
            print(final_features)

            scaled_dp=result_scaler.transform(final_features)
            print(scaled_dp)

            prediction=result_model.predict(scaled_dp)
            print(prediction)

            if prediction[0]==1:
                output='Pass'
            else:
                output="Fail"

            return render_template('result.html', prediction_text="Prediction is  {}".format(output)) #here
        
        except ValueError:
            return render_template('result.html', prediction_text='Please enter valid numerical values.')#here

@app.route('/math')
def math():
    return render_template('math.html')

@app.route('/mathsubmit', methods=['POST'])
def mathsubmit():
    if request.method=='POST':
        try:
            # get inputs
            gender=request.form.get('gender')
            ethnicity=request.form.get('ethnicity')
            parental_education=request.form.get('parental_level_of_education')
            lunch=request.form.get('lunch')
            test_prep=request.form.get('test_preparation_course')
            reading_score=float(request.form.get('reading_score'))
            writing_score=float(request.form.get('writing_score'))


            # Validate no empty selection
            if '' in [gender, ethnicity, parental_education, lunch, test_prep, reading_score, writing_score]:
                return render_template('math.html', prediction_text="Please select all fields.") #here

            # Encode inputs using mapping dictionaries as used in training
            gender_map = {'female':0, 'male':1}
            ethnicity_map = {'group A':0, 'group B':1, 'group C':2, 'group D':3, 'group E':4}
            parental_map = {
                "associate's degree":0,
                "bachelor's degree":1,
                'high school':2,
                "master's degree":3,
                'some college':4,
                'some high school':5
            }
            lunch_map = {'standard':1, 'free/reduced':0}
            test_prep_map = {'none':1, 'completed':0}

            # Convert to numerical values
            data = [
                gender_map[gender],
                ethnicity_map[ethnicity],
                parental_map[parental_education],
                lunch_map[lunch],
                test_prep_map[test_prep],
                reading_score,
                writing_score
            ]

            
            final_features = [np.array(data)]
            print(final_features)

            scaled_dp=math_scaler.transform(final_features)
            print(scaled_dp)

            prediction=math_model.predict(scaled_dp)
            print(prediction)

            return render_template('math.html', prediction_text="Predicted math score is  {}".format(prediction)) #here
        
        except ValueError:
            return render_template('math.html', prediction_text='Please enter valid numerical values.')#here

if __name__=="__main__":
    app.run(debug=True)