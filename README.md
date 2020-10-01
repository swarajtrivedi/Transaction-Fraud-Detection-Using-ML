# Transaction-Fraud-Detection-Using-ML
This is a web application to detect the fraud credit card transactions using machine learning, the model used in the code is the XGBoost model. Accuracy, AUC Score is around 0.99.
Anyhow you can try all sorts of models to replace the one in the code.
The Repository consists of html and python files,
1)The app.py file has the code for your flask app.
2)The index.py is code for the ML model ,also this code creates a model.pkl file on your folder which will be later used by the app.py to make predictions.(NOTE: Download the creditcard.csv dataset from https://www.kaggle.com/mlg-ulb/creditcardfraud and place it in the same directory as of the index.py file) .
3)index.html and all the other .html  files need to be placed in a 'templates' folder (under the same directory),you can create a folder named 'templates' and add the html files to it,this is necessary because flask recognizes the html files only under the templates folder.
4)To run the app, first run the index.py file ,your model.pkl will be created in the same directory.
5)After this, run app.py and go to your localhost(127.0.0.1:5000),input the feature values and click on predict.
6)If it is a fraud , fraud.html will open up else not_fraud.html will open .
