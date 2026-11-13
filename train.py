import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,f1_score
from data_prep import X_train,X_test,y_train,y_test

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Churn_Prediction_Experiment")

with mlflow.start_run():
    n_trees = 50
    depth = 3

    mlflow.log_param("n_estimators",n_trees)
    mlflow.log_param("max_depth",depth)

    model = RandomForestClassifier(n_estimators=n_trees,max_depth=depth,random_state=42)
    model.fit(X_train,y_train)

    pred = model.predict(X_test)
    acc  = accuracy_score(y_test,pred)
    prec = precision_score(y_test,pred)
    f1   = f1_score(y_test,pred)

    mlflow.log_metrics({
        "accuracy":acc,
        "precision":prec,
        "f1_score":f1
    })

    model_info = mlflow.sklearn.log_model(model,"rf_model",skops_trusted_types=["sklearn.tree._tree.Tree"],registered_model_name="rf3_model")

    print("RF model accuracy is : ", acc)

    client = MlflowClient()

    if f1 > 0.80:
        client.set_registered_model_alias("rf3_model",alias="production",version=model_info.registered_model_version)
        mlflow.set_tag("deployment_stage","Production")
    else:
        mlflow.set_tag("deployment_stage","Staged")