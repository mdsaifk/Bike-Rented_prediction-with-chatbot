
# importing the necessary dependencies
from flask_cors import CORS,cross_origin
from sklearn.ensemble import RandomForestRegressor
#import sklearn.linear_model.LinearRegression
from flask import Flask, request, jsonify,render_template
import os
import numpy as np
from wsgiref import simple_server
from sklearn.linear_model import LinearRegression
import pickle
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from chat import get_response

app = Flask(__name__)
 # initializing a flask app

@app.route('/') # route to display the home page
@cross_origin()
def home():
    return render_template("temp.html")

@app.post("/predict")
def predict():
    text = request.get_json().get("message")
   
    # we now see if the message is valid one or not
    response = get_response(text)
    print(response)
    message = {"answer": response}
    return jsonify(message)

@app.route('/pred',methods=['POST','GET']) # route to show the predictions in a web UI
@cross_origin()
def pred():
    if request.method == 'POST':
        try:#  reading the inputs given by the user
            season=float(request.form['season'])
            holiday = float(request.form['holiday'])
            workingday = float(request.form['workingday'])
            weather = float(request.form['weather'])
            temp = float(request.form['temp'])
            atemp = float(request.form['atemp'])
            humidity = float(request.form['humidity'])
            windspeed = float(request.form['windspeed'])
            casual = float(request.form['casual'])
            registered = float(request.form['registered'])
            hour = float(request.form['hour'])
            #train= pd.read_csv('new_train.csv') # loading the training data
         #X_train, X_test, y_train, y_test = train_test_split(X, Y,test_size=0.25,random_state=255)

            #Rf = RandomForestRegressor(n_estimators = 100,criterion='mse',random_state=255,max_depth=50,min_samples_split=10, verbose=3,oob_score=True)
            #Rf.fit(X_train,y_train)
            filename = 'finalized_bike_model.pickle'
            #pickle.dump(Rf, open(filename, 'wb'))
            
            RF1 = pickle.load(open(filename, 'rb')) # loading the model file from the storage
            # predictions using the loaded model file
            #prediction=loaded_model.predict([[gre_score,toefl_score,university_rating,sop,lor,cgpa,research]])
            prediction=RF1.predict([[season,holiday,workingday,weather,temp,atemp,humidity,windspeed,casual,registered,hour]])
            # showing the prediction results in a UI
            #return render_template('results.html')


            return render_template('temp.html',prediction=np.round(prediction))
        except Exception as e:
            print('The Exception message is: ',e)
            return 'something is wrong'
    # return render_template('results.html')
    else:
        return render_template('temp.html')


if __name__ == "__main__":
    #app.run(port = 4000)
    port = int(os.getenv("PORT"))
    #clApp = ClientApp()
    host = '0.0.0.0'
    httpd = simple_server.make_server(host=host,port=port, app=app)
    httpd.serve_forever()
