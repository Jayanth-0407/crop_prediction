import pandas as pd
#import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns

from sklearn.model_selection import train_test_split
#from sklearn.metrics import classification_report,accuracy_score
from sklearn.ensemble import RandomForestClassifier

import pickle


df=pd.read_csv('Crop_recommendation.csv')
#df.head()

#df.shape

#df.isnull().sum()



x=df.drop('label',axis=1) #features
y=df['label']  #labels

#y.head()

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

model=RandomForestClassifier()

model.fit(x_train,y_train)  

pickle.dump(model,open("model.pkl","wb"))      #dumping the model

#prediction=model.predict(x_test)

#accuracy=accuracy_score(y_test,prediction)

#print(accuracy)

#new_feature=[[36,58,25,28.66,59.31,8.399,36.92]]
#predicted_crop=model.predict(new_feature)

#print(predicted_crop)