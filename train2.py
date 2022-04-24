import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn

import logging
logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)


def eval_metrics(actual, pred):
    print(actual)
    print(pred)
    rmse = np.sqrt(np.mean(np.square((actual-pred)/actual)))
    mae = np.mean(np.abs((actual-pred)/actual))
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

if __name__=="__main__":
    warnings.filterwarnings(action="ignore")
    np.random.seed(40)

    csv_url = (
        "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    )

    try:
        data=pd.read_csv(csv_url, sep=";")
    except Exception as e:
        logger.exception("Unable to download training & test CSV, \
            check your internet connection. Error: %s", e)
    print(data)
    train, test=train_test_split(data)
    train.reset_index(inplace=True, drop=True)
    test.reset_index(inplace=True, drop=True)

    

    train_x=train.drop(['quality'], axis=1)
    test_x=test.drop(['quality'], axis=1)
    train_y=train['quality'].values
    test_y=test['quality'].values

    alpha=float(sys.argv[1]) if len(sys.argv)>1 else 0.5
    l1_ratio=float(sys.argv[2]) if len(sys.argv)>2 else 0.5

    with mlflow.start_run():
        lr=ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        predicted=lr.predict(test_x)

        (rrmse, rmae, r2)=eval_metrics(test_y, predicted)

        logger.info("Linear regression model: alpha=%f, l1_ratio=%f:" %(alpha, l1_ratio) )
        logger.info("RRMSE: %s" %rrmse)
        logger.info("RMAE: %s" % rmae)
        logger.info("R2: %s" %r2)

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rrmse", rrmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("rmae", rmae)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        if tracking_url_type_store != "file":

            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(lr, "model", registered_model_name="ElasticnetWineModel")
        else:
            mlflow.sklearn.log_model(lr, "model")













