import pandas as pd 
import sklearn 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
file=pd.read_csv('trafficaccident.csv')
file.head()
#preprocessing
df = file.drop([ 'Location_Easting_OSGR', 'Location_Northing_OSGR','Longitude',
       'Latitude', 'Police_Force','Date','Local_Authority_(District)','Local_Authority_(Highway)',
       '1st_Road_Class', '1st_Road_Number', 'Speed_limit','Junction_Detail', 'Junction_Control', '2nd_Road_Class',
       '2nd_Road_Number', 'Pedestrian_Crossing-Human_Control', 'Pedestrian_Crossing-Physical_Facilities', 
       'Special_Conditions_at_Site', 'Carriageway_Hazards','Did_Police_Officer_Attend_Scene_of_Accident','Accident_Index',
       'LSOA_of_Accident_Location','Urban_or_Rural_Area','Time'], axis=1)

tr=file['Urban_or_Rural_Area']

df['Light_Conditions']=LabelEncoder().fit_transform(df['Light_Conditions'])

df['Road_Type']=LabelEncoder().fit_transform(df['Road_Type'])

df['Weather_Conditions']=LabelEncoder().fit_transform(df['Weather_Conditions'])

df['Road_Surface_Conditions']=LabelEncoder().fit_transform(df['Road_Surface_Conditions'])

X_train,X_test,y_train,y_test=train_test_split(df,tr,train_size=0.8)
X_train

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import time
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,log_loss,mean_squared_error

#do classification of this data with seven metrix 
c1=time.time()
kn=KNeighborsClassifier(n_neighbors=20,).fit(X_train,y_train)
yp=kn.predict(X_test)
c11=time.time()
print(accuracy_score(y_test,yp))
print(f1_score(y_test,yp))
print(precision_score(y_test,yp))
print(recall_score(y_test,yp))
print(log_loss(y_test,yp))
print(mean_squared_error(y_test,yp))
print("computation time for knn",(c11-c1))

c2=time.time()
rf=RandomForestClassifier(max_depth=12,n_jobs=-1).fit(X_train,y_train)
yp2=rf.predict(X_test)
c22=time.time()
print(accuracy_score(y_test,yp2))
print(f1_score(y_test,yp2))
print(precision_score(y_test,yp2))
print(recall_score(y_test,yp2))
print(log_loss(y_test,yp2))
print(mean_squared_error(y_test,yp2))
print("computation time for knn",(c22-c2))

c3=time.time()
ds=DecisionTreeClassifier(max_depth=13).fit(X_train,y_train)
yp3=ds.predict(X_test)
c33=time.time()
print(accuracy_score(y_test,yp3))
print(f1_score(y_test,yp3))
print(precision_score(y_test,yp3))
print(recall_score(y_test,yp3))
print(log_loss(y_test,yp3))
print(mean_squared_error(y_test,yp3))
print("computation time for knn",(c33-c3))

c4=time.time()
ad=AdaBoostClassifier(n_estimators=19).fit(X_train,y_train)
yp4=ad.predict(X_test)
c44=time.time()
print(accuracy_score(y_test,yp4))
print(f1_score(y_test,yp4))
print(precision_score(y_test,yp4))
print(recall_score(y_test,yp4))
print(log_loss(y_test,yp4))
print(mean_squared_error(y_test,yp4))
print("computation time for knn",(c44-c4))


# implement of adam optimizer 
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
c5=time.time()
# Define your neural network architecture
model=Sequential()
model.add(Dense(64,activation='relu', input_dim=9))  # Input layer
model.add(Dense(32, activation='relu'))
model.add(Dense(1,activation='sigmoid'))  
    # [(Dense(64, activation='relu',input_shape=9)),(Dense(32, activation='relu')),(Dense(1, activation='sigmoid'))])
# Define your Adam optimizer with a learning rate of 0.001
adam_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# # Compile your model with the Adam optimizer and a categorical crossentropy loss function
model.compile(optimizer=adam_optimizer,loss='binary_crossentropy',metrics=['accuracy'])

# # Train your model with the Adam optimizer
model.fit(X_train, y_train, epochs=10)
ff=model.predict(X_test)
c55=time.time()
print(accuracy_score(y_test,ff))
print(f1_score(y_test,ff))
print(precision_score(y_test,ff))
print(recall_score(y_test,ff))
print(log_loss(y_test,ff))
print(mean_squared_error(y_test,ff))
print("computation time for knn",(c55-c5))



# algorithm with hyperparameter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.optimizers import Adam

 
# Create a KNN classifier with default k=5
knn = KNeighborsClassifier()
# Define the hyperparameters to tune
hyperparams = {'n_neighbors': [11, 15, 20],
               'weights': ['uniform', 'distance'],
               'metric': ['euclidean', 'manhattan']}

# Create a GridSearchCV object to tune the hyperparameters using 5-fold cross-validation
grid_search = GridSearchCV(knn, hyperparams, cv=5)

# Train the KNN classifier with the Adam optimizer and tuned hyperparameters using Grid Search CV
grid_search.fit(X_train, y_train)

# Get the best hyperparameters and the corresponding model
best_hyperparams = grid_search.best_params_
best_model = grid_search.best_estimator_

# Evaluate the best model on the testing data
score = best_model.score(X_test, y_test)
pp=grid_search.predict(X_test)
print(f"Best hyperparameters: {best_hyperparams}")
print(f"Best KNN score: {grid_search.best_score_:.3f}")
print(f"Test score: {score:.3f}")

ran = RandomForestClassifier()
# Define the hyperparameters to tune
hyperparams = {'n_estimators': [1,2,3,20]}

# Create a GridSearchCV object to tune the hyperparameters using 5-fold cross-validation
grid_search = GridSearchCV(ran, hyperparams, cv=5)

# Train the KNN classifier with the Adam optimizer and tuned hyperparameters using Grid Search CV
grid_search.fit(X_train, y_train)

# Get the best hyperparameters and the corresponding model
best_hyperparams = grid_search.best_params_
best_model = grid_search.best_estimator_

# Evaluate the best model on the testing data
score = best_model.score(X_test, y_test)
pp=grid_search.predict(X_test)
print(f"Best hyperparameters: {best_hyperparams}")
print(f"Best KNN score: {grid_search.best_score_:.3f}")
print(f"Test score: {score:.3f}")



dec = DecisionTreeClassifier()
# Define the hyperparameters to tune
hyperparams = {'random_state': [100,30,20]}

# Create a GridSearchCV object to tune the hyperparameters using 5-fold cross-validation
grid_search = GridSearchCV(dec,hyperparams, cv=5)

# Train the KNN classifier with the Adam optimizer and tuned hyperparameters using Grid Search CV
grid_search.fit(X_train, y_train)

# Get the best hyperparameters and the corresponding model
best_hyperparams = grid_search.best_params_
best_model = grid_search.best_estimator_

# Evaluate the best model on the testing data
score = best_model.score(X_test, y_test)
pp1=grid_search.predict(X_test)

print(f"Best hyperparameters: {best_hyperparams}")
print(f"Best dec score: {grid_search.best_score_:.3f}")
print(f"Test score: {score:.3f}")



c90=time.time()
ada=AdaBoostClassifier()
# Define the hyperparameters to tune
hyperparams = {'n_estimators': [1,2,3,20]}

# Create a GridSearchCV object to tune the hyperparameters using 5-fold cross-validation
grid_search = GridSearchCV(ada, hyperparams, cv=5)

# Train the KNN classifier with the Adam optimizer and tuned hyperparameters using Grid Search CV
grid_search.fit(X_train, y_train)

# Get the best hyperparameters and the corresponding model
best_hyperparams = grid_search.best_params_
best_model = grid_search.best_estimator_

# Evaluate the best model on the testing data
score = best_model.score(X_test, y_test)
pp2=grid_search.predict(X_test)
c09=time.time()

print(pp2)
print(f"Best hyperparameters: {best_hyperparams}")
print(f"Best ada score: {grid_search.best_score_:.3f}")
print(f"Test score: {score:.3f}")

