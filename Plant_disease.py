import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,BaggingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
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
model=DecisionTreeClassifier(ccp_alpha=0.001, criterion='entropy', max_depth= 10, min_samples_leaf=30,min_samples_split=25,min_impurity_decrease=0)
model.fit(train_feature,train_label)
prediction=model.predict(test_feature)
#print(prediction)
#Accuracy
Acc=accuracy_score(prediction,test_label)
print(Acc)
#using the model
import joblib
joblib.dump(model,"ABCD.pkl")
loaded_model = joblib.load("ABCD.pkl")
l1=[30.7,34.34,65.4,2.32]
feature=np.array(l1).reshape(1,-1)
result=loaded_model.predict(feature)
print(result)