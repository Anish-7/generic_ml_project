from flask import Flask,render_template,request
import pandas as pd 
import numpy as np
from src.pipeline.predict_pipeline import Predict_pipeline , Custom_data

application = Flask(__name__)
app = application


@app.route('/',methods=['GET','POST'])
def project():
    if request.method == 'GET':
        return render_template('anish.html')

    if request.method == 'POST':
        sample_data = Custom_data(
                    gender = request.form.get('gender'),
                    ethnicity = request.form.get('ethnicity'),
                    parental_level_of_education = request.form.get('parental_level_of_education'),
                    lunch = request.form.get('lunch'),
                    test_preparation_course = request.form.get('test_preparation_course'),
                    reading_score = int(request.form.get('reading_score')),
                    writing_score = int(request.form.get('writing_score')),

        )

        data_df = sample_data.get_features()
        results = Predict_pipeline().predict(data_df)


        return render_template('anish.html',results=results[0])

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8000,debug=True)