#title : 'CREDIT CARD FRAUD DETECTION'
import pickle

import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import seaborn as sns
data = pd.read_csv('creditcard.csv')

#class is the one we want to predict
#before splitting the dataset let us first impute the values 


y = data[['Class']] #targ
X = data

#let us try using SelectKbest with chi^2
'''bestfeatures = SelectKBest(score_func=chi2,k=10)

fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
featurescores = pd.concat([dfcolumns,dfscores],axis =1)
featurescores.columns = ['Specs','Scores']
print(featurescores.nlargest(10,'Scores'))'''
#as this does not work,i tried extratrees classifier for best features
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model = ExtraTreesClassifier()
model.fit(X,y)
print(model.feature_importances_)
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()
#plotting heatmap for the correlation
corrmatrix = data.corr()
top_features = corrmatrix.index
plt.figure()
g= sns.heatmap(data[top_features].corr(),annot=True,cmap = 'RdYlGn')
plt.show()
#quite unclear,hence we use the extratreesclassiefier only
#now knowing that we have the features suitable for model prediction 
#split into train and test data
from sklearn.model_selection import train_test_split

X = data[['V3','V4','V16','V18','V11','V10','V12','V14','V17']]
X_train,X_test,y_train,y_test = train_test_split(X,y)

#using training data for model training
import xgboost as xgb
from xgboost import XGBClassifier

model_first = XGBClassifier()#Imp: classifier is used here, and not regressor
model_first.fit(X_train,y_train) #fitting model to the training data

pickle.dump(model_first,open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))
'''
from sklearn.metrics import accuracy_score
#finding the auc score for this model
auc = accuracy_score(y_test,model_prediction)
print(auc) '''









