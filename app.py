import numpy as np
import pandas as pd
from flask import Flask,request,render_template
import pickle

flask_app=Flask(__name__)

model=pickle.load(open("model.pkl","rb"))

@flask_app.route('/',methods=["GET"])
def home():
    return render_template("index.html")

@flask_app.route("/predict",methods=["POST"])

def predict():
    float_feature=[float(x) for x in request.form.values()]
    features = [np.array(float_feature)]
    prediction = model.predict(features)
    return render_template("index.html",prediction_text="the predicted crop is: {}".format(prediction))

if __name__=="__main__":
    flask_app.run()
