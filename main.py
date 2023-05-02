# imports
import pandas as pd
import numpy as np

from datetime import datetime

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from Logistic_Classifier import logistic_clf_model
from Gradient_Boosting_Classifier import gb_clf_model
from Random_Forest_Classifier import rf_clf_model
# from SVM_Classifier import svm_clf_model

f = open("Summary2.txt", "w")
f.write("Start Time = " + datetime.now().strftime("%H:%M:%S") + "\n")

f.write("\nReading Fraud data file\n")
fraud_data = pd.read_csv("./Fraud.csv")

fraud_data = fraud_data.sample(frac=1, random_state = 42).reset_index(drop=True)
f.close()

for i in range(3, 10):
    f = open("Summary2.txt", "a")
    f.write("\nSplitting data into training (" + str(i*10) + "%) validating(" + str(50 - (i*5)) + "%) and testing(" + str(50 - (i*5)) + "%)\n")
    training_data = fraud_data[:int(len(fraud_data) * i/10)]
    validation_data = fraud_data[int(len(fraud_data) * i/10) : int(len(fraud_data) * (0.5 + 0.05*i))]
    test_data = fraud_data[int(len(fraud_data)*(0.5 + 0.05*i)):]

    f.write("\nSeperating features and target\n")
    X_train = training_data.drop(['isFraud'], axis = 1)
    y_train = training_data['isFraud']

    X_val = validation_data.drop(['isFraud'], axis = 1)
    y_val = validation_data['isFraud']

    X_test = test_data.drop(['isFraud'], axis = 1)
    y_test = test_data['isFraud']

    f.write("\nFeature Engineering:\n")
    def filter_type(type):
        if type == "TRANSFER" or type == "CASH_OUT":
            return type
        else:
            return "OTHER"

    f.write("Adding a column with values as 'TRANSFER', 'CASH_OUT' and 'OTHER'\n")
    X_train['filter_type'] = list(map(filter_type, X_train['type']))
    X_val['filter_type'] = list(map(filter_type, X_val['type']))
    X_test['filter_type'] = list(map(filter_type, X_test['type']))

    f.write("Adding column with M as Merchant and C as Customer using nameDest initials\n")
    X_train['nameDestType'] = list(map(lambda name_dest: name_dest[0], X_train['nameDest']))
    X_val['nameDestType'] = list(map(lambda name_dest: name_dest[0], X_val['nameDest']))
    X_test['nameDestType'] = list(map(lambda name_dest: name_dest[0], X_test['nameDest']))

    f.write("Adding column that gives changeBalance as change in Balance using 'newbalanceOrig' and 'oldbalanceOrg'\n")
    X_train['changeBalance'] = X_train['newbalanceOrig'] - X_train['oldbalanceOrg']
    X_val['changeBalance'] = X_val['newbalanceOrig'] - X_val['oldbalanceOrg']
    X_test['changeBalance'] = X_test['newbalanceOrig'] - X_test['oldbalanceOrg']

    f.write("\nSeperating numerical and category attributes:\n")
    num_attribs = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'isFlaggedFraud', 'changeBalance']
    cat_attribs =['type', 'nameOrig', 'nameDest', 'nameDestType', 'filter_type']

    f.write("\nGenerating preprocessing numerical(Standard Scaling) and categorical(One hot encoding) attributes\n\n")
    num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())

    cat_pipeline = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore"))

    preprocessing = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs),
    ])

    f.close()
    f = open("Summary2.txt", "a")
    logistic_clf_model(f, X_train, y_train, X_test, y_test, preprocessing)
    f.close()
    f = open("Summary2.txt", "a")
    gb_clf_model(f, X_train, y_train, X_test, y_test, preprocessing)
    f.close()
    f = open("Summary2.txt", "a")
    rf_clf_model(f, X_train, y_train, X_test, y_test, preprocessing)
    f.close()
    # svm_clf_model(f, X_train, y_train, X_test, y_test)

f = open("Summary2.txt", "a")
f.write("End Time = " + datetime.now().strftime("%H:%M:%S") + "\n")
f.close()