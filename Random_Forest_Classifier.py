from datetime import datetime

from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

def rf_clf_model(f, X_train, y_train, X_test, y_test, preprocessing):
    time1 = datetime.now()
    f.write("Current Time = " + time1.strftime("%H:%M:%S") + "\n")

    f.write('Training Random Forest Classifier model...\n')
    rf_clf = make_pipeline(preprocessing, RandomForestClassifier(n_estimators=10, random_state=42))
    rf_clf.fit(X_train, y_train)

    time2 = datetime.now()
    f.write("Current Time = " + time2.strftime("%H:%M:%S") + "\n")

    f.write('Predicting Random Forest Classifier model on train data...\n')
    rf_clf_pred = rf_clf.predict(X_train)
    # f.write('Predicting Random Forest Classifier model on test data...\n')
    # rf_clf_pred = rf_clf.predict(X_test)

    time3 = datetime.now()
    f.write("Current Time = " + time3.strftime("%H:%M:%S") + "\n")

    f.write('Analysis:\n')

    # confusionMatrix = confusion_matrix(y_test, rf_clf_pred)
    accuracy = accuracy_score(y_train, rf_clf_pred)
    precision = precision_score(y_train, rf_clf_pred, average = None)
    recall = recall_score(y_train, rf_clf_pred, average = None)
    # accuracy = accuracy_score(y_test, rf_clf_pred)
    # precision = precision_score(y_test, rf_clf_pred, average = None)
    # recall = recall_score(y_test, rf_clf_pred, average = None)

    # f.write('\nClassification Report\n' + classification_report(y_test, rf_clf_pred, target_names=['Fraud', 'Not Fraud']))
    # f.write("\nConfusion Matrix:\n" + confusionMatrix + "\n")
    f.write("Accuracy:" + str(accuracy) + "\n")
    f.write("Precision:" + str(precision) + "\n")
    f.write("Recall:" + str(recall) + "\n\n")
    # f.write("Training Time:" + (time2 - time1).strftime("%H:%M:%S") + "\n")
    # f.write("Predicting Time:" + (time3 - time2).strftime("%H:%M:%S") + "\n")