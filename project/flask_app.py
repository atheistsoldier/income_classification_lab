import pickle

from flask import Flask,request, url_for, redirect, render_template, jsonify
import pandas as pd
import pickle


app = Flask(__name__)
filename = 'finalized_model.sav'
model = pickle.load(open(filename,'rb'))
cols = model.feature_names_in_

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict',methods=['POST'])
def predict():
    features = request.form.to_dict()
    print(request.get_json())
    features['fnlwgt']=1
    features['education-num']=13
    data_unseen = pd.DataFrame(columns = cols)
    inp_data={}
    print(features)
    for col in features.keys():
        
        inp_data[col]=[features[col]]
    inp_data=pd.DataFrame(inp_data)
    inp_data=pd.get_dummies(inp_data)
    for col in cols:
        data_unseen[col]=[0]
    for col in inp_data:
        data_unseen[col]=inp_data[col]
        data_unseen['capital-gain']=data_unseen['capital-gain']+5
    data_unseen['capital-loss']=data_unseen['capital-loss']+5

    #data_unseen.drop(columns=['income_ >50K', 'income_ <=50K'],inplace=True)   
    prediction = model.predict(data_unseen)
    prediction = int(prediction[0])
    print(prediction)
    return render_template('home.html',pred='Income >=50 k {}'.format(prediction))



if __name__ == '__main__':
    app.run(debug=True)