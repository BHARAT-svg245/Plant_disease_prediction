import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
#Reading the data file
df=pd.read_csv("Plant_disease.csv")
#Scaling the data
scaler=StandardScaler()
splt=StratifiedShuffleSplit(n_splits=1,test_size=0.1,random_state=42)
for train_idx,test_idx in splt.split(df,df["Target"]):
    train_set=df.loc[train_idx]
    test_set=df.loc[test_idx]
#Training data
train_feature=train_set.drop("Target",axis=1)
train_label=train_set["Target"]
train_feature=scaler.fit_transform(train_feature)
#Testing data
test_feature=test_set.drop("Target",axis=1)
test_feature=scaler.fit_transform(test_feature)
test_label=test_set["Target"]
#Making the model
model=SVC(kernel="rbf")
model=RandomForestClassifier()
model.fit(train_feature,train_label)
#Cross-validation
cvs=cross_val_score(model,train_feature,train_label,cv=10,scoring="accuracy")
#print(cvs)
#making the prediction
test_pred=model.predict(test_feature)
train_pred=model.predict(train_feature)
#Accuracy
test_acc=accuracy_score(test_pred,test_label)
train_acc=accuracy_score(train_label,train_label)
#using the 
l1=[15.201649380601122,36.87071032809986,5.345501147061558,5.890208238089201]
import joblib
# #joblib.dump(model,"Plant_disease_model.joblib")
loaded_model=joblib.load("Plant_disease_model.joblib")
f=np.array([l1])
y_pred=loaded_model.predict(f)
#print(y_pred)
