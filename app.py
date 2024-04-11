# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 10:59:35 2024

@author: becky
"""

from flask import Flask, request, render_template
import joblib
import pandas as pd
import numpy as np
from datetime import datetime


app=Flask(__name__)

#load pipelines
#model=joblib.load("./final_pipeline.pkl")
lr_pipeline=joblib.load("./Logistic Regression_pipeline.pkl")
dt_pipeline=joblib.load("./Decision Tree_pipeline.pkl")
svm_pipeline=joblib.load("./SVM_pipeline.pkl")
rf_pipeline=joblib.load("./Random Forest_pipeline.pkl")
nn_pipeline=joblib.load("./Neural Network_pipeline.pkl")


#route the app
@app.route("/")
#associate route with a function-render templates(html files in the templates folder)
def home():
    return render_template("index.html")

@app.route("/result", methods=["POST"])

def result():
   #get values from form
    date_input=request.form.get("dateInput")
    time_input=request.form.get("timeInput")
    ROAD_CLASS=request.form.get("roadClass")
    LATITUDE=float(request.form.get("latitudeInput"))
    LONGITUDE=float(request.form.get("longitudeInput"))
    classifier=request.form.get("classifier")
    
    #process date/time data
    date_data=date_input.split("-")
    Month=int(date_data[1])
    Day=int(date_data[2])
    #Convert day of week to int
    Weekday=pd.Timestamp(date_input).weekday()
    #Convert time to 24h clock
    time_obj=datetime.strptime(time_input, "%H:%M")
    TIME=time_obj.strftime("%H%M")
    TIME=TIME.zfill(4)
    print(TIME)

    columns=["TIME", "ROAD_CLASS", "LATITUDE", "LONGITUDE", "Weekday", "Day", "Month"]
    features=pd.DataFrame([[TIME,ROAD_CLASS,LATITUDE,LONGITUDE,Weekday,Day,Month]],columns=columns)
    

    prediction=predict(classifier, features)
    
    return render_template("result.html",prediction=prediction)

def predict(classifier,features):
    if classifier == 'lr':
        model=lr_pipeline
    elif classifier =='dt':
        model=dt_pipeline
    elif classifier =='svm':
        model=svm_pipeline
    elif classifier == 'rf':
        model=rf_pipeline
    elif classifier == 'nn':
        model=nn_pipeline
        
    prediction=model.predict(features)
    return prediction[0]

#run the app
if __name__=="__main__":
    app.run(debug=True, port=5000)