import io
import json
import time
import logging
import pickle
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
    "email" : "marley@example.com",
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

    
def create_dag(dag_id: str, m_name: Literal["random_forest", "linear_regression", "desicion_tree"]):
    
    dag = DAG(dag_id = dag_id,
              schedule_interval = "0 1 * * *",
              start_date = days_ago(2),
              catchup = False,
              tags = ["mlops"],
              default_args = DEFAULT_ARGS)


    def init(m_name: Literal["random_forest", "linear_regression", "desicion_tree"]) -> Dict[str, Any]:
        metrics = {}
        metrics["model_name"] = m_name

        _LOG.info(f"Train pipeline started for {metrics['model_name']}")
        
        return metrics



    def get_data_from_sklearn(**kwargs) -> Dict[str, Any]:

        task_instance = kwargs['task_instance']
        metrics = task_instance.xcom_pull(task_ids='init')

        model_name = metrics["model_name"]
        _LOG.info(f"Name model: {model_name}")

        start_time = time.time()

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
            key=f"Ivan_Yuminov/{model_name}/datasets/california_housing.pkl",
            bucket_name=BUCKET,
            replace=True,
        )

        end_time = time.time()
        metrics["loading_time"] = end_time - start_time
        _LOG.info(f"Data loaded. Loading time : {metrics['loading_time']}")

        return metrics


    def prepare_data(**kwargs) -> Dict[str, Any]:

        task_instance = kwargs["task_instance"]
        metrics = task_instance.xcom_pull(task_ids= "get_data_from_sklearn")

        _LOG.info(f"METRICS prepare_data: {metrics}")

        start_time = time.time()
        model_name = kwargs["model_name"]

        # Использовать созданный ранее S3 connection
        s3_hook = S3Hook("s3_connection")
        file = s3_hook.download_file(key=f"Ivan_Yuminov/{model_name}/datasets/california_housing.pkl", bucket_name=BUCKET)
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
        X_train_fitted = scaler.fit_transform(X_train)
        X_test_fitted = scaler.transform(X_test)

        # Обучить стандартизатор на train
        scaler = StandardScaler()
        X_train_fitted = scaler.fit_transform(X_train)
        X_test_fitted = scaler.transform(X_test)

        for name, data in zip(
            ["X_train", "X_test", "y_train", "y_test"],
            [X_train_fitted, X_test_fitted, y_train, y_test],
        ):
            filebuffer = io.BytesIO()
            pickle.dump(data, filebuffer)
            filebuffer.seek(0)
            s3_hook.load_file_obj(
                file_obj=filebuffer,
                key=f"Ivan_Yuminov/{model_name}/datasets/{name}.pkl",
                bucket_name=BUCKET,
                replace=True,
            )

        end_time = time.time()
        metrics["preproces_time"] = end_time - start_time
        _LOG.info(f"Preprocessing time: {metrics['preproces_time']}")
        return metrics


    def train_model(**kwargs) -> Dict[str, Any]:

        task_instance = kwargs["task_instance"]
        metrics = task_instance.xcom_pull(task_ids="prepare_data")

        _LOG.info(f"METRICS train_model: {metrics}")

    
        start_time = time.time()
        model_name = kwargs["model_name"]
        # Использовать созданный ранее S3 connection
        s3_hook = S3Hook("s3_connection")
        # Загрузить готовые данные с S3
        data = {}
        for name in ["X_train", "X_test", "y_train", "y_test"]:
            file = s3_hook.download_file(
                key=f"Ivan_Yuminov/{model_name}/datasets/{name}.pkl",
                bucket_name=BUCKET,
            )
            data[name] = pd.read_pickle(file)

        # Обучить модель
        model = models[model_name]
        model.fit(data["X_train"], data["y_train"])
        prediction = model.predict(data["X_test"])

        # Посчитать метрики
        result = {}
        result["r2_score"] = r2_score(data["y_test"], prediction)
        result["rmse"] = mean_squared_error(data["y_test"], prediction) ** 0.5
        result["mae"] = median_absolute_error(data["y_test"], prediction)

        metrics["model_metrics"] = result

        end_time = time.time()

        metrics["training_time"] = end_time - start_time
        
        _LOG.info(f"Model trained. Metrics: {metrics['model_metrics']}")
        _LOG.info(f"Training time: {metrics['training_time']}")

        return metrics

    def save_results(**kwargs) -> None:

        ti = kwargs["ti"]
        metrics = ti.xcom_pull(task_ids="train_model")

        _LOG.info(f"METRICS save_results: {metrics}")

        # model_metrics = metrics["model_metrics"]

        model_name = metrics["model_name"]


        s3_hook = S3Hook("s3_connection")
        buffer = io.BytesIO()
        buffer.write(json.dumps(metrics).encode())
        buffer.seek(0)
        s3_hook.load_file_obj(
            file_obj=buffer,
            key=f"Ivan_Yuminov/{model_name}/results/metrics.json",
            bucket_name=BUCKET,
            replace=True
            )
        
        _LOG.info("Success.")


    with dag:
        task_init = PythonOperator(task_id = "init", python_callable = init, dag = dag, op_kwargs= {"m_name": m_name})
        task_get_data = PythonOperator(task_id = "get_data_from_sklearn", python_callable = get_data_from_sklearn, dag = dag, provide_context = True, op_kwargs= {"model_name": model_name})
        task_prepare_data = PythonOperator(task_id = "prepare_data", python_callable = prepare_data, dag = dag, provide_context = True, op_kwargs= {"model_name": model_name})
        task_train_model = PythonOperator(task_id = "train_model", python_callable = train_model, dag = dag, provide_context = True, op_kwargs = {"model_name": model_name})
        task_save_results = PythonOperator(task_id = "save_results", python_callable = save_results, dag = dag)
    
    task_init >> task_get_data >> task_prepare_data >> task_train_model >> task_save_results


for model_name in models.keys():
    create_dag(f"IvanYuminov_{model_name}", model_name)