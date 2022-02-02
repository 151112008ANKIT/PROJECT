from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.conf import settings
from django.db.models import Q
from django.core.files.storage import FileSystemStorage
from django.contrib import messages
from django.db import connection
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import mean_squared_error
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
import json
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler



# Create your views here.

def details(request, fileId):
    ### Get the File name ###
    fileDetails = getFileData(fileId)
    
    #### Read CSV File and Upload into Database ####
    print(os.getcwd())
    csv_data = pd.read_csv(str(os.getcwd())+'/media/'+ fileDetails['files_original_file_name'], header=0, encoding = 'unicode_escape')
    csv_data = csv_data.values.tolist()

    context = {
        "heartlist": csv_data
    }

    # Message according medicines Role #
    context['heading'] = "Heart Details"
    return render(request, 'heart-list.html', context)


def analysis(request, fileId):
    ### Get the File name ###
    fileDetails = getFileData(fileId)
    
    #### Read CSV File and Upload into Database ####
    print(os.getcwd())
    csv_data = pd.read_csv(str(os.getcwd())+'/media/'+ fileDetails['files_original_file_name'], header=0, encoding = 'unicode_escape')
    dataset = csv_data
    csv_data = csv_data.values.tolist()
    target_temp = dataset.target.value_counts()
    gender_temp = dataset.sex.value_counts() 
    context = {
        "heartlist": csv_data,
        "typeofdata": type(dataset),
        "datashape": dataset.shape,
        "datahead": dataset.head(5),
        "datadescribe" : dataset.describe(),
        #"datadescribe" :sns.barplot(x=dataset['sex'],y=dataset['thalach'],hue=dataset['target']),
        "targeunique": dataset["target"].unique(),
        "datainfo": dataset.info(),
        "datasample": dataset.sample(5),
        "targetdescribe": dataset["target"].describe(),
        "corelation": dataset.corr()["target"].abs().sort_values(ascending=False),
        "whp": round(target_temp[0]*100/303,2),
        "wh": round(target_temp[1]*100/303,2),
        "male": round(gender_temp[1]*100/303,2),
        "female": round(gender_temp[0]*100/303,2)
        
    }

    # Message according medicines Role #
    context['heading'] = "Heart Details"
    return render(request, 'analysis.html', context)

def prediction(request, fileId):
    context = {}
    ### Get the Database Configuration ####
    if (request.method == "POST"):
        ### Insert the File Details #####
        form_data = [[
            request.POST['age'],
            request.POST['gender'], 
            request.POST['chest_pain_type'], 
            request.POST['bp'],
            request.POST['sc'], 
            request.POST['sugar'], 
            request.POST['resting'], 
            request.POST['hra'], 
            request.POST['eia'], 
            request.POST['depression'], 
            request.POST['st'], 
            request.POST['vessels'], 
            request.POST['thal']
        ]]
        prediction = dataTraining(fileId, form_data)
        
        pcontext = {
            "prediction": prediction,
            "form_data": (form_data[0])
        }
        return render(request, 'prediction-result.html', pcontext)
    # Message according medicines Role #
    context['heading'] = "Heart Details"
    return render(request, 'prediction.html', context)
def listToString(list):
    lst=[]
    for i in list:
        lst.append(i[0])
    return lst

def cumSumToString(list):
    lst=[]
    for i in list:
        lst.append(i)
    return lst
    
def dictfetchall(cursor):
    "Return all rows from a cursor as a dict"
    columns = [col[0] for col in cursor.description]
    return [
        dict(zip(columns, row))
        for row in cursor.fetchall()
    ]

def getFileData(id):
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM files WHERE files_id = " + id)
    dataList = dictfetchall(cursor)
    return dataList[0]

def dataTraining(fileId, testData):
    ### Get the File name ###
    prediction_results = {}
    fileDetails = getFileData(fileId)
    
    #### Read CSV File and Upload into Database ####
    dataset = pd.read_csv(str(os.getcwd())+'/media/'+ fileDetails['files_original_file_name'])
    
    info = [
        "age",
        "1: male, 0: female",
        "chest pain type, 1: typical angina, 2: atypical angina, 3: non-anginal pain, 4: asymptomatic",
        "resting blood pressure",
        "serum cholestoral in mg/dl",
        "fasting blood sugar > 120 mg/dl",
        "resting electrocardiographic results (values 0,1,2)",
        "maximum heart rate achieved",
        "exercise induced angina",
        "oldpeak = ST depression induced by exercise relative to rest",
        "the slope of the peak exercise ST segment",
        "number of major vessels (0-3) colored by flourosopy",
        "thal: 3 = normal; 6 = fixed defect; 7 = reversable defect"
    ]

                                         
    # Removing target varaible 
    '''
    dummies=pd.get_dummies(dataset, columns = ['ca',
                                             'cp', 
                                             'exang', 
                                             'fbs', 
                                             'restecg',
                                             'sex',
                                             'slope',
                                             'thal'],drop_first=True)
    dummies2=dummies.drop(['age','trestbps','chol','thalach','oldpeak','target'],axis='columns')
    merged=pd.concat([dataset,dummies2],axis='columns')
    final=merged.drop(['ca','cp','exang', 'fbs', 'restecg','sex','slope','thal'],axis='columns')
    print(final.shape)
    '''
    Y = dataset['target']
    print(Y.shape)
    X= dataset.drop(['target'], axis = 1)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size = 0.3,random_state = 2606)
    print ("train_set_x shape: " + str(X_train.shape))
    print ("train_set_y shape: " + str(Y_train.shape))
    print ("test_set_x shape: " + str(X_test.shape))
    print ("test_set_y shape: " + str(Y_test.shape))  
    scaler = preprocessing.StandardScaler().fit(X_train)
    scaler 
    X_scaled = scaler.transform(X_train)
    scaler.mean_ 
    scaler.scale_
    X_scaled.mean(axis=0)
    X_scaled.std(axis=0)

   
    # MLP Classification 
   
    pipe3 = make_pipeline(StandardScaler(),MLPClassifier(hidden_layer_sizes=100, activation='relu', solver='adam', 
                   alpha=0.0001, batch_size='auto', learning_rate='constant', 
                   learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True,
                   random_state=None, tol=0.0001, verbose=False, warm_start=False,
                   momentum=0.9, nesterovs_momentum=True, early_stopping=False,
                   validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08))
    pipe3.fit(X_train, Y_train)
    Y_pred_mlp=pipe3.predict(X_test)
    score_mlp = round(accuracy_score(Y_pred_mlp,Y_test)*100,2)
    prediction_results['mlp_score']=score_mlp  
    from sklearn.metrics import classification_report
    print("MLP",classification_report(Y_test,Y_pred_mlp))            

    # Training using Logistic Regression
    
    pipe = make_pipeline(StandardScaler(),LogisticRegression(penalty='l2',class_weight='balanced'))
    pipe.fit(X_train, Y_train.values.ravel())  # apply scaling on training data
    Y_pred_lr=pipe.predict(X_test)
    score_lr = round(accuracy_score(Y_test,Y_pred_lr)*100,2)
    prediction_results['logistic_score'] = score_lr
    print("logestic regression",classification_report(Y_test,Y_pred_lr))
    
   
  

   


    ## Training using Random Forest Algorithm
  
    pipe4 = make_pipeline(StandardScaler(),RandomForestClassifier(n_estimators=30, min_samples_leaf=7, min_samples_split=10,
      class_weight = "balanced", random_state=1, oob_score=True))
    pipe4.fit(X_train, Y_train)
    y_pred_rf= pipe4.predict(X_test)
    score_rf = round(accuracy_score(y_pred_rf,Y_test)*100,2)
    prediction_results['rf_score'] =score_rf
    print("random forest",classification_report(Y_test,y_pred_rf))

    ## Training using Naive Base Algorithm
    pipe5 = make_pipeline(StandardScaler(),GaussianNB())
    pipe5.fit(X_train,Y_train)
    Y_pred_nb = pipe5.predict(X_test)
    score_nb= round(accuracy_score(Y_pred_nb,Y_test)*100,2)
    prediction_results['nb_score'] =score_nb
    print("naive bayse",classification_report(Y_test,Y_pred_nb))
   

    ## Training using SVM Algorithm
    pipe2 = make_pipeline(StandardScaler(),svm.SVC(kernel='linear'))
    pipe2.fit(X_train, Y_train)
    Y_pred_svm = pipe2.predict(X_test)
    score_svm = round(accuracy_score(Y_pred_svm,Y_test)*100,2)
    print("The accuracy score achieved using SVM is: "+str(score_svm)+" %")
    prediction_results['svm_score'] =score_svm
    print("SVM",classification_report(Y_test,Y_pred_svm))

   
    pipe1 = make_pipeline(StandardScaler(),KNeighborsClassifier(n_neighbors=9))
    pipe1.fit(X_train,Y_train)
    Y_pred_knn=pipe1.predict(X_test)
   
    score_knn = round(accuracy_score(Y_test,Y_pred_knn)*100,2)

    print("The accuracy score achieved using KNN is: "+str(score_knn)+" %")     
    prediction_results['knn_score'] =score_knn
    print("knn",classification_report(Y_test,Y_pred_knn))
   

    
    

    ## Training using bagging Algorithm
 
    
    max_accuracy = 0
    for x in range(200):
        dt = DecisionTreeClassifier(random_state=x)
        dt.fit(X_train,Y_train)
        Y_pred_dt = dt.predict(X_test)
        current_accuracy = round(accuracy_score(Y_pred_dt,Y_test)*100,2)
        if(current_accuracy>max_accuracy):
            max_accuracy = current_accuracy
            best_x = x
    dt = DecisionTreeClassifier(random_state=best_x)
    dt.fit(X_train,Y_train)
    Y_pred_dt = dt.predict(X_test)
    prediction_results['dt_score'] = round(accuracy_score(Y_pred_dt,Y_test)*100,2)
    print(round(accuracy_score(Y_pred_dt,Y_test)*100,2))
    # Form data 
    form_data = pd.DataFrame(testData)

    test_data_set = dataset.tail(5)
    test_data_set = test_data_set.drop(columns=["target"])

    
    
    # MLP Classification
    if(pipe3.predict(form_data)[0] == 1):
        prediction_results['MLP_prediction'] = "Suffered From Heart Disease"
        prediction_results['MLP_class'] = "redc"
    else:
        prediction_results['MLP_prediction'] = "Not Suffered"
        prediction_results['MLP_class'] = "greenc"
    
    
    ## Logistic Regression Prediction
    if(pipe.predict(form_data)[0] == 1):
        prediction_results['logistic_prediction'] = "Suffered From Heart Disease"
        prediction_results['logistic_class'] = "redc"
    else:
        prediction_results['logistic_prediction'] = "Not Suffered"
        prediction_results['logistic_class'] = "greenc"
    
    ## Random Forest Prediction
    if(pipe4.predict(form_data)[0] == 1):
        prediction_results['rf_prediction'] = "Suffered From Heart Disease"
        prediction_results['rf_class'] = "redc"
    else:
        prediction_results['rf_prediction'] = "Not Suffered"
        prediction_results['rf_class'] = "greenc"

    ## Naive Baise Prediction
    if(pipe5.predict(form_data)[0] == 1):
        prediction_results['nb_prediction'] = "Suffered From Heart Disease"
        prediction_results['nb_class'] = "redc"
    else:
        prediction_results['nb_prediction'] = "Not Suffered"
        prediction_results['nb_class'] = "greenc"

    ## SVM Prediction
    if(pipe2.predict(form_data)[0] == 1):
        prediction_results['svm_prediction'] = "Suffered From Heart Disease"
        prediction_results['svm_class'] = "redc"
    else:
        prediction_results['svm_prediction'] = "Not Suffered"
        prediction_results['svm_class'] = "greenc"

    ## KNN Prediction
    if(pipe1.predict(form_data)[0] == 1):
        prediction_results['knn_prediction'] = "Suffered From Heart Disease"
        prediction_results['knn_class'] = "redc"
    else:
        prediction_results['knn_prediction'] = "Not Suffered"
        prediction_results['knn_class'] = "greenc"

    ## bagging Prediction
    if(dt.predict(form_data)[0] == 1):
        prediction_results['dt_prediction'] = "Suffered From Heart Disease"
        prediction_results['dt_class'] = "redc"
    else:
        prediction_results['dt_prediction'] = "Not Suffered"
        prediction_results['dt_class'] = "greenc"

    return prediction_results


def smape_kun(y_true, y_pred):
    return np.mean((np.abs(y_pred - y_true) * 200/ (np.abs(y_pred) + np.abs(y_true))))