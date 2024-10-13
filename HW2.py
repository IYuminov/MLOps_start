import os
import warnings
import mlflow
import pandas as pd

from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository

from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

experiment_id = mlflow.create_experiment(name="Ivan_Yuminov")
search_experiment = mlflow.search_experiments(filter_string="name = 'Ivan_Yuminov'")
print(f'Created experiment: {search_experiment}')

# Получим датасет California housing
housing = fetch_california_housing(as_frame=True)

# Разделить данные на обучение, валидацию и тест
X_train, X_test, y_train, y_test = train_test_split(housing['data'], housing['target'])
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5)

# Обучить стандартизатор на train
scaler = StandardScaler()
X_train_scalled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index= X_train.index)
X_val_scalled = pd.DataFrame(scaler.transform(X_val), columns=X_train.columns, index= X_val.index)
X_test_scalled = pd.DataFrame(scaler.transform(X_test), columns=X_train.columns, index= X_test.index)

models = {
    "RandomForestRegressor": RandomForestRegressor(),
    "LinearRegression": LinearRegression(),
    "DecisionTreeRegressor": DecisionTreeRegressor()
}

with mlflow.start_run(run_name="@Ivan_Yuminov", experiment_id = experiment_id, description = "parent") as parent_run:
    for model_name in models.keys():
        with mlflow.start_run(run_name= model_name, experiment_id= experiment_id, nested=True) as child_run:
            model = models[model_name]

            # Обучим модель.
            model.fit(X_train_scalled, y_train)

            # Сделаем предсказание.
            prediction = model.predict(X_val_scalled)

            # Создадим валидационный датасет.
            eval_df = X_val_scalled.copy()
            eval_df["target"] = y_val

            # Сохраним результаты обучения с помощью MLFlow.
            signature = infer_signature(X_test_scalled, prediction)
            model_info = mlflow.sklearn.log_model(model, 'linreg', signature=signature, registered_model_name=f"sklearn-{model_name}-model")
            mlflow.evaluate(
                model=model_info.model_uri,
                data=eval_df,
                targets="target",
                model_type="regressor",
                evaluators=["default"],
            )