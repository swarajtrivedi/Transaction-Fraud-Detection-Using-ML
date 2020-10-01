from flask import Flask,render_template,request
import pickle
import numpy as np
import pandas as pd
app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))
@app.route('/')
def hello_world():
    return render_template('index.html')
@app.route('/predict',methods = ['POST','GET'])
def predict():
    
    col_names = model.get_booster().feature_names
    vals = [[int(x) for x in request.form.values()]]
    vals = pd.DataFrame(vals,columns=col_names)
    
    print(vals.head)
    prediction = model.predict(vals)
    if prediction==1:
        return render_template('fraud.html')
    else:
        return render_template('not_fraud.html')

if __name__ == "__main__":
    app.run()