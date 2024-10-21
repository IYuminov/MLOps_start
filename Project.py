import io
import os
import json
import pickle
import mlflow
import logging
import datetime
import pandas as pd

from typing import Any, Dict, Literal
from datetime import timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, median_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing

from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.models import DAG, Variable
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
from mlflow.models import infer_signature

for key in [
    "MLFLOW_TRACKING_URI",
    "AWS_ENDPOINT_URL",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_DEFAULT_REGION",
]:
    os.environ[key] = Variable.get(key)

_LOG = logging.getLogger()
_LOG.addHandler(logging.StreamHandler())

BUCKET = Variable.get("S3_BUCKET")
FEATURES = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]
TARGET = "MedHouseVal"

DEFAULT_ARGS = {
    "owner": "Ivan_Yuminov",
    "email_on_failure" : True,
    "email_on_retry" : False,
    "retry" : 3,
    "retry_delay" : timedelta(minutes=1)
}

model_names = ["random_forest", "linear_regression", "desicion_tree"]

models = dict(
    zip(model_names, [
        RandomForestRegressor(),
        LinearRegression(),
        DecisionTreeRegressor(),
    ]))

    
dag = DAG(
    dag_id = 'Ivan_Yuminov',
    schedule_interval = "0 1 * * *",
    start_date = days_ago(2),
    catchup = False,
    tags = ["mlops"],
    default_args = DEFAULT_ARGS
    )


def init(**kwargs) -> Dict[str, Any]:
    metrics = {}

    init_time = datetime.datetime.now().strftime("%d-%m-%Y %H:%M")

    experiment_id = mlflow.create_experiment(name="Ivan_Yuminov")
    with mlflow.start_run(run_name="@Ivan_Yuminov", experiment_id=experiment_id, description="parent") as parent_run:
        parent_run_id = parent_run.info.run_id

    kwargs['task_instance'].xcom_push(key='parent_run_id', value=parent_run_id)
    kwargs['task_instance'].xcom_push(key='experiment_id', value=experiment_id)
    kwargs['task_instance'].xcom_push(key='start_time', value=init_time)

    metrics["init_time"] = init_time

    _LOG.info(f"Pipeline started at {metrics['init_time']}.")

    return metrics



def get_data_from_sklearn(**kwargs) -> Dict[str, Any]:

    task_instance = kwargs['task_instance']
    metrics = task_instance.xcom_pull(task_ids='init')

    start_get_data = datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")

    # Получим датасет California housing
    housing = fetch_california_housing(as_frame=True)
    # Объединим фичи и таргет в один np.array
    data = pd.concat([housing["data"], pd.DataFrame(housing["target"])], axis=1)
    # Использовать созданный ранее S3 connection
    s3_hook = S3Hook("s3_connection")
    filebuffer = io.BytesIO()
    data.to_pickle(filebuffer)
    filebuffer.seek(0)

    # Сохранить файл в формате pkl на S3
    s3_hook.load_file_obj(
        file_obj=filebuffer,
        key=f"Ivan_Yuminov/datasets/california_housing.pkl",
        bucket_name=BUCKET,
        replace=True,
    )

    finish_get_data = datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    data_shape = data.shape

    kwargs['task_instance'].xcom_push(key='start_get_data', value=start_get_data)
    kwargs['task_instance'].xcom_push(key='finish_get_data', value=finish_get_data)
    kwargs['task_instance'].xcom_push(key='data_shape', value=data_shape)

    metrics['start_get_data'] = start_get_data
    metrics['finish_get_data'] = finish_get_data
    metrics['data_shape'] = data_shape

    _LOG.info("Data loaded.")

    return metrics


def prepare_data(**kwargs) -> Dict[str, Any]:

    task_instance = kwargs["task_instance"]
    metrics = task_instance.xcom_pull(task_ids= "get_data_from_sklearn")

    start_prepare_data = datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")

    # Использовать созданный ранее S3 connection
    s3_hook = S3Hook("s3_connection")
    file = s3_hook.download_file(
        key=f"Ivan_Yuminov/datasets/california_housing.pkl",
        bucket_name=BUCKET
        )
    data = pd.read_pickle(file)

    # Сделать препроцессинг
    # Разделить на фичи и таргет
    X, y = data[FEATURES], data[TARGET]

    # Разделить данные на обучение и тест
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = 0.2, random_state = 42
    )

    # Обучить стандартизатор на train
    scaler = StandardScaler()
    X_train_fitted = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index= X_train.index)
    X_test_fitted = pd.DataFrame(scaler.transform(X_test), columns=X_train.columns, index= X_test.index)

    for name, dt in zip(
        ["X_train", "X_test", "y_train", "y_test"],
        [X_train_fitted, X_test_fitted, y_train, y_test],
    ):
        filebuffer = io.BytesIO()
        pickle.dump(dt, filebuffer)
        filebuffer.seek(0)
        s3_hook.load_file_obj(
            file_obj=filebuffer,
            key=f"Ivan_Yuminov/datasets/{name}.pkl",
            bucket_name=BUCKET,
            replace=True,
        )

    finish_prepare_data = datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    features_data = data[FEATURES].columns.to_list()

    kwargs['task_instance'].xcom_push(key='start_prepare_data', value=start_prepare_data)
    kwargs['task_instance'].xcom_push(key='finish_prepare_data', value=finish_prepare_data)
    kwargs['task_instance'].xcom_push(key='features_data', value=features_data)

    metrics['start_prepare_data'] = start_prepare_data
    metrics['finish_prepare_data'] = finish_prepare_data
    metrics['features_data'] = features_data

    _LOG.info("Preprocessing finished.")

    return metrics


def train_model(**kwargs) -> Dict[str, Any]:

    task_instance = kwargs["task_instance"]
    metrics = task_instance.xcom_pull(task_ids="prepare_data")

    experiment_id = kwargs['task_instance'].xcom_pull(key='experiment_id')
    parent_run_id = kwargs['task_instance'].xcom_pull(key='parent_run_id')

    start_train = datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")

    model_name = kwargs["model_name"]
    # Использовать созданный ранее S3 connection
    s3_hook = S3Hook("s3_connection")
    # Загрузить готовые данные с S3
    data = {}
    for name in ["X_train", "X_test", "y_train", "y_test"]:
        file = s3_hook.download_file(
            key=f"Ivan_Yuminov/datasets/{name}.pkl",
            bucket_name=BUCKET,
        )
        data[name] = pd.read_pickle(file)

    X_val, X_test, y_val, y_test = train_test_split(data['X_test'], data['y_test'], test_size=0.5)

    # Обучить модель
    model = models[model_name]

    with mlflow.start_run(parent_run_id=parent_run_id, run_name= model_name, experiment_id= experiment_id, nested= True) as child_run:
        model.fit(data["X_train"], data["y_train"])
        finish_train = datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    
        prediction = model.predict(X_val)
        
        eval_df = X_val.copy()
        eval_df["target"] = y_val

        signature = infer_signature(X_test, prediction)
        model_info = mlflow.sklearn.log_model(model, model_name, signature=signature, registered_model_name=f"sklearn-{model_name}-model")
        mlflow.evaluate(
            model=model_info.model_uri,
            data=eval_df,
            targets="target",
            model_type="regressor",
            evaluators=["default"],
        )

        kwargs['task_instance'].xcom_push(key=f'start_train_{model_name}', value=start_train)
        kwargs['task_instance'].xcom_push(key=f'finish_train_{model_name}', value=finish_train)
    
        metrics[f'start_train_{model_name}'] = start_train
        metrics[f'finish_train_{model_name}'] = finish_train

    _LOG.info(f"{model_name} trained.")

    print('Это metrics:', metrics)
    return metrics


def save_results(**kwargs) -> None:

    # print('Это kwargs', kwargs)
    metrics = {}

    metrics['train_random_forest'] = kwargs["task_instance"].xcom_pull(task_ids="train_random_forest")
    metrics['train_linear_regression'] = kwargs["task_instance"].xcom_pull(task_ids="train_linear_regression")
    metrics['train_desicion_tree'] = kwargs["task_instance"].xcom_pull(task_ids="train_desicion_tree")

    s3_hook = S3Hook("s3_connection")
    buffer = io.BytesIO()
    buffer.write(json.dumps(metrics).encode())
    buffer.seek(0)
    s3_hook.load_file_obj(
        file_obj=buffer,
        key=f"Ivan_Yuminov/results/metrics.json",
        bucket_name=BUCKET,
        replace=True
        )
    
    _LOG.info("Saving results successfully. Pipeline finished.")


task_init = PythonOperator(task_id = "init", python_callable = init, dag = dag)
task_get_data = PythonOperator(task_id = "get_data_from_sklearn", python_callable = get_data_from_sklearn, dag = dag)
task_prepare_data = PythonOperator(task_id = "prepare_data", python_callable = prepare_data, dag = dag)
task_train_model = [
    PythonOperator(task_id = "train_random_forest", python_callable = train_model, dag = dag, provide_context = True, op_kwargs = {"model_name": "random_forest"}),
    PythonOperator(task_id = "train_linear_regression", python_callable = train_model, dag = dag, provide_context = True, op_kwargs = {"model_name": "linear_regression"}),
    PythonOperator(task_id = "train_desicion_tree", python_callable = train_model, dag = dag, provide_context = True, op_kwargs = {"model_name": "desicion_tree"})
]
task_save_results = PythonOperator(task_id = "save_results", python_callable = save_results, dag = dag)

task_init >> task_get_data >> task_prepare_data >> task_train_model >> task_save_results
