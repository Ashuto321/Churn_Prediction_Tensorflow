import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('C:\\Users\\Ashutosh Pandey\\Desktop\\churn.csv')
X= pd.get_dummies(df.drop(['Churn', 'Customer ID'], axis=1))
y= df['Churn'].apply(lambda x:1 if x=='yes' else 0)

X_train,X_text, y_train, y_test= train_test_split(X,y,test_size=.2)

# print(X_train.head())
# print(y_train.head())

# impoting main dependencies: 

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score

# Build and complie Model
model= Sequential()
model.add(Dense(units=32,activation='relu',input_dim=len(X_train.columns)))
model.add(Dense(units=64,activation='relu'))
model.add(Dense(units=1,activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='sgd',metrics= ['accuracy'])

# Fit,Predict and Evalute 
model.fit(X_train,y_train,epochs=50, batch_size=32)
y_hat=model.predict(X_text)
y_hat=[0 if val<0.5 else 1 for val in y_hat]
# print(y_hat)
print(accuracy_score(y_test,y_hat))
 