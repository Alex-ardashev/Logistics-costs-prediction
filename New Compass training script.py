# Databricks notebook source
# MAGIC %md # XGBoost for Cost Prediction

# COMMAND ----------

import mlflow
from mlflow.deployments import get_deploy_client
mlflow.set_registry_uri("databricks-uc")
mlflow.autolog(disable=True)

# COMMAND ----------

# MAGIC %pip install --upgrade hyperopt xgboost optuna
# MAGIC %pip uninstall -y scikit-learn
# MAGIC %pip install scikit-learn==1.5.2

# COMMAND ----------

import numpy as np
import pandas as pd
import optuna
from hyperopt import fmin, tpe, hp, SparkTrials, STATUS_OK
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split, KFold
from xgboost import XGBRegressor
import hashlib

# COMMAND ----------

# MAGIC %md ## Setup
# MAGIC - Configure MLflow with Unity Catalog
# MAGIC - Define catalog and schema
# MAGIC - Prepare and preprocess data

# COMMAND ----------

CATALOG_NAME = "generic_catalog"
SCHEMA_NAME = "ml_schema"

# COMMAND ----------

# MAGIC %md ### Sample Data Query
# MAGIC This is a generalized version of the data preparation query

# COMMAND ----------

query = """
WITH transactions AS (
    SELECT  
        t.id AS transaction_id,
        CAST(t.created_date AS DATE) AS created_date,
        CAST(t.delivery_date AS DATE) AS delivery_date,
        t.total_amount AS transaction_total,
        t.estimated_cost AS estimated_cost,
        t.margin AS transaction_margin,
        t.destination_country,
        t.destination_city
    FROM source.transactions t
    WHERE t.created_date >= '2023-01-01'
      AND t.status = 'Completed'
),

suppliers AS (
    SELECT 
        s.id AS supplier_id,
        s.supplier_name,
        s.city AS supplier_city,
        s.country AS supplier_country
    FROM source.suppliers s
    WHERE s.active = true
),

items AS (
    SELECT 
        i.transaction_id,
        i.quantity,
        i.unit_price,
        i.product_id,
        s.supplier_id,
        s.supplier_city AS city,
        s.supplier_country AS country,
        p.units_per_container,
        p.containers_per_shipment,
        CASE 
            WHEN p.containers_per_shipment IS NOT NULL THEN i.quantity / p.containers_per_shipment
            ELSE NULL
        END AS shipment_share
    FROM source.transaction_items i
    JOIN source.supplier_products sp ON i.product_id = sp.product_id
    JOIN suppliers s ON sp.supplier_id = s.supplier_id
    JOIN source.products p ON i.product_id = p.id
),

agg_items AS (
    SELECT 
        transaction_id,
        SUM(shipment_share) AS total_shipment_share,
        AVG(shipment_share) AS avg_shipment_share,
        AVG(units_per_container) AS avg_units_per_container,
        SUM(units_per_container) AS total_units_per_container,
        AVG(quantity) AS avg_quantity,
        SUM(quantity) AS total_quantity,
        AVG(unit_price) AS avg_unit_price,
        COUNT(DISTINCT supplier_id) AS num_suppliers,
        COUNT(DISTINCT city) AS num_cities,
        COUNT(DISTINCT country) AS num_countries,
        COUNT(DISTINCT product_id) AS num_products
    FROM items
    GROUP BY transaction_id
),

actual_costs AS (
    SELECT 
        e.transaction_id,
        SUM(e.actual_cost) AS actual_cost
    FROM source.events e
    WHERE e.status = 'Completed'
    GROUP BY e.transaction_id
    HAVING SUM(e.actual_cost) > 0
)

SELECT
    t.*,
    ai.total_shipment_share,
    ai.avg_shipment_share,
    ai.avg_units_per_container,
    ai.total_units_per_container,
    ai.avg_quantity,
    ai.total_quantity,
    ai.avg_unit_price,
    ai.num_suppliers,
    ai.num_cities,
    ai.num_countries,
    ai.num_products,
    ac.actual_cost
FROM transactions t
JOIN agg_items ai ON t.transaction_id = ai.transaction_id
JOIN actual_costs ac ON t.transaction_id = ac.transaction_id
ORDER BY t.created_date
"""

data_df = spark.sql(query)
spark.sql(f"DROP TABLE IF EXISTS {CATALOG_NAME}.{SCHEMA_NAME}.training_data")
data_df.write.saveAsTable(f"{CATALOG_NAME}.{SCHEMA_NAME}.training_data")

# COMMAND ----------

# MAGIC %md ### Preprocess Data

# COMMAND ----------

data = spark.read.table(f"{CATALOG_NAME}.{SCHEMA_NAME}.training_data").toPandas()

def hash_encoding(val, num_bins=1000):
    return int(hashlib.sha256(str(val).encode()).hexdigest(), 16) % num_bins

def preprocess_data(df, train_stats=None):
    df = df.copy()
    is_train = train_stats is None
    if is_train:
        train_stats = {}
    
    # Date features
    for col in ['created_date', 'delivery_date']:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    df['delivery_days'] = (df['delivery_date'] - df['created_date']).dt.days
    
    # Numeric columns
    num_cols = ['transaction_total', 'transaction_margin', 'total_quantity',
                'num_suppliers', 'num_countries', 'total_shipment_share']
    for col in num_cols:
        if is_train:
            train_stats[f'{col}_median'] = df[col].median()
        df[col].fillna(train_stats[f'{col}_median'], inplace=True)
    
    # Hash encoding
    for col in ['destination_country', 'destination_city']:
        df[f'{col}_hash'] = df[col].apply(lambda x: hash_encoding(x))
    
    # Target and features
    y = df['actual_cost']
    X = df.drop(columns=['transaction_id', 'created_date', 'delivery_date', 
                        'estimated_cost', 'actual_cost',
                        'destination_country', 'destination_city'])
    
    return (X, y, train_stats) if is_train else (X, y)

# Split and preprocess
train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)
X_train, y_train, train_stats = preprocess_data(train_df)
X_test, y_test = preprocess_data(test_df, train_stats)

# COMMAND ----------

# Hyperparameter Tuning
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
    }
    
    model = XGBRegressor(**params, random_state=42)
    scores = []
    
    for train_idx, val_idx in KFold(n_splits=5).split(X_train):
        model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
        preds = model.predict(X_train.iloc[val_idx])
        scores.append(np.sqrt(mean_squared_error(y_train.iloc[val_idx], preds)))
        
    return np.mean(scores)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

# Final Model Training
best_params = study.best_params

# COMMAND ----------

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

mlflow.autolog(disable=False)
with mlflow.start_run(run_name='xgb_best_model'):
    best_model = XGBRegressor(**best_params)
    best_model.fit(X_train, y_train)
    
    y_pred_test = best_model.predict(X_test)
    
    rmse_test = rmse(y_test, y_pred_test)
    mape_test = mean_absolute_percentage_error(y_test, y_pred_test)
    r2_test = r2_score(y_test, y_pred_test)
    
    mlflow.log_metrics({
        "rmse_test": rmse_test,
        "mape_test": mape_test,
        "r2_test": r2_test
    })
    
    print(f"\n[TEST] RMSE: {rmse_test:.2f}")
    print(f"[TEST] MAPE: {mape_test:.2f}")
    print(f"[TEST] R-squared: {r2_test:.2f}")

# COMMAND ----------

runs = mlflow.search_runs()
best_run = runs.sort_values('metrics.rmse_test', ascending=True).iloc[0]
best_run_id = best_run.run_id

print(f"Best Run ID: {best_run_id}")
print(f"Best RMSE: {best_run['metrics.rmse_test']}")

# COMMAND ----------

model_uri = f'runs:/{best_run_id}/model'
mlflow.register_model(model_uri, f"{CATALOG_NAME}.{SCHEMA_NAME}.xgb_best_model")

client = mlflow.tracking.MlflowClient()
latest_version = client.search_model_versions(
    f"name='{CATALOG_NAME}.{SCHEMA_NAME}.xgb_best_model'"
)[0].version

client.set_registered_model_alias(
    name=f"{CATALOG_NAME}.{SCHEMA_NAME}.xgb_best_model",
    alias="Production",
    version=latest_version
)

deploy_client = get_deploy_client("databricks")
endpoint = deploy_client.update_endpoint(
    endpoint="cost_prediction_endpoint",
    config={
        "served_entities": [
            {
                "entity_name": f"{CATALOG_NAME}.{SCHEMA_NAME}.xgb_best_model",
                "entity_version": latest_version,
                "workload_size": "Small",
                "scale_to_zero_enabled": True
            }
        ],
    }
)

print('Deployment completed')
