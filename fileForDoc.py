# necessary Imports
import pandas as pd
#import matplotlib.pyplot as plt
import pickle
#from main import categorical_to_numeric
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from scipy import stats 
from sklearn.metrics import mean_squared_error
import pickle


#test = pd.read_csv('train_bikes.csv', parse_dates=['datetime'])
train= pd.read_csv('new_train.csv') # loading the training data

print(train.head())

Y = train['count']
X = train.drop(columns = ['count'])

X_train, X_test, y_train, y_test = train_test_split(
    X, Y,test_size=0.25,random_state=255
)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

Rf = RandomForestRegressor(n_estimators = 100,criterion='mse',random_state=255,max_depth=50,min_samples_split=10, verbose=3,oob_score=True)
Rf.fit(X_train,y_train)

print(Rf.score(X_test,y_test))

Rf.predict([[1,0,0,1,9.84,14.395,81,0.0,3,13,0]])


# saving the model to the local file system
filename = 'finalized_bike_model.pickle'
pickle.dump(Rf, open(filename, 'wb'))

'''filename = 'finalized_new_bike_model.pickle'
RF1 = pickle.load(open(filename, 'rb'))
pred=RF1.predict([[1,0,1,1,10.66,13.65,35,7.0015,12,55,2]])
print('prediction is', pred)'''

'''df= pd.read_csv('Admission_Prediction.csv') # reading the CSV file

df.head() # cheking the first five rows from the dataset
df.info() # printing the summary of the dataframe

df['GRE Score'].fillna(df['GRE Score'].mode()[0],inplace=True)
#to replace the missing values in the 'GRE Score' column with the mode of the column
# Mode has been used here to replace the scores with the most occuring scores so that data follows the general trend

df['TOEFL Score'].fillna(df['TOEFL Score'].mode()[0],inplace=True)
#to replace the missing values in the 'GRE Score' column with the mode of the column
# Mode has been used here to replace the scores with the most occuring scores so that data follows the general trend

df['University Rating'].fillna(df['University Rating'].mean(),inplace=True)
#to replace the missing values in the 'University Rating' column with the mode of the column
# Mean has been used here to replace the scores with the average score

# dropping the 'Chance of Admit' and 'serial number' as they are not going to be used as features for prediction
x=df.drop(['Chance of Admit','Serial No.'],axis=1)
# 'Chance of Admit' is the target column which shows the probability of admission for a candidate
y=df['Chance of Admit']


plt.scatter(df['GRE Score'],y) # Relationship between GRE Score and Chance of Admission
plt.scatter(df['TOEFL Score'],y) # Relationship between TOEFL Score and Chance of Admission
plt.scatter(df['CGPA'],y) # Relationship between CGPA and Chance of Admission


# splitting the data into training and testing sets
from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.33, random_state=100)

# fitting the date to the Linear regression model
from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit(train_x, train_y)

# calucltaing the accuracy of the model
from sklearn.metrics import r2_score
score= r2_score(reg.predict(test_x),test_y)

# saving the model to the local file system
filename = 'finalized_model.pickle'
pickle.dump(reg, open(filename, 'wb'))

# prediction using the saved model.
loaded_model = pickle.load(open(filename, 'rb'))
prediction=loaded_model.predict(([[320,120,5,5,5,10,1]]))
print(prediction[0])'''