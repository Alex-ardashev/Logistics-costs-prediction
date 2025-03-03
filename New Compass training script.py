# Databricks notebook source
# MAGIC %md # XGBoost for logistics total costs prediction

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

from hyperopt import fmin, tpe, hp, SparkTrials, Trials, STATUS_OK
from hyperopt.pyll import scope
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score, make_scorer
from sklearn.model_selection import train_test_split, KFold
from xgboost import XGBRegressor
import hashlib


import xgboost as xgb
import matplotlib.pyplot as plt

# COMMAND ----------

# MAGIC %md ## Setup
# MAGIC
# MAGIC - Configure the MLflow client to use Unity Catalog as the model registry.  
# MAGIC - Set the catalog and schema where the model will be registered.
# MAGIC - Read in the data and save it to tables in Unity Catalog.
# MAGIC - Preprocess the data.

# COMMAND ----------

# Specify the catalog and schema to use. You must have USE_CATALOG privilege on the catalog and USE_SCHEMA, CREATE_TABLE, and CREATE_MODEL privileges on the schema.
# Change the catalog and schema here if necessary.
CATALOG_NAME = "sandbox"
SCHEMA_NAME = "ml_schema"

# COMMAND ----------

# MAGIC %md ### Read in data and save it to tables in Unity Catalog
# MAGIC The dataset is available in `databricks-datasets`. In the following cell, you read the data in from `.csv` files into Spark DataFrames. You then write the DataFrames to tables in Unity Catalog. This both persists the data and lets you control how to share it with others.

# COMMAND ----------

query = """
WITH quot AS (
    SELECT  
        q.Id AS quote_id,
        CAST(q.createdDate AS DATE) AS created_date,
        CAST(Delivery_date_min__c AS DATE) AS delivery_date,
        Total_Sales_price__c AS quote_total_sales_price,
        COALESCE(Estimated_Logistics_costs__c, estimated_logistics_costs_automated__c) AS quote_estimated_log_costs,
        Estimated_Logistics_cost_ML__c AS quote_estimated_log_costs_ml,
        Gross_Margin__c AS quote_gross_margin,
        ShippingCountry AS quote_delivery_country,
        ShippingCity AS quote_delivery_city
    FROM main.salesforce.silver_quote q
    JOIN main.salesforce.silver_opportunity op
      ON q.OpportunityId = op.Id
    WHERE 1=1 
      AND lower(BillingName) NOT LIKE '%ummy%'
      AND lower(BillingName) NOT LIKE '%ndercore%'
      AND COALESCE(Estimated_Logistics_costs__c, estimated_logistics_costs_automated__c) IS NOT NULL
      AND q.createdDate >= '2024-06-01'
      AND Status = 'Accepted'
      AND op.StageName = 'Closed Won'
),

acc AS (
    SELECT 
        Id AS acc_id,
        name,
        ShippingCity AS acc_shipping_city,
        ShippingCountry AS acc_shipping_country
    FROM main.salesforce.silver_account
    WHERE RecordTypeId IN ('01209000000qLEbAAM', '01209000000qLEWAA2', '01209000000qLEgAAM')
      AND Id <> '001Jy000009ys3nIAA' -- andercore placeholder
      AND lower(name) NOT LIKE '%ummy%'
      AND lower(name) NOT LIKE '%ndercore%'
),

qli AS (
    SELECT 
        qli.QuoteId AS quote_id,
        qli.delivery_date__c as delivery_date,
        qli.Quantity AS qli_quantity,
        qli.unitPrice AS qli_unit_price,
        qli.Product2Id AS qli_product_id,
        pq.Supplier_Account__c AS supplier_id,
        supplier.acc_shipping_city AS city,
        supplier.acc_shipping_country AS country,
        p.Units_per_pallet__c AS product_per_pallet, 
        p.Pallets_per_40ft_truck__c AS product_per_truck,
        CASE 
            WHEN p.Pallets_per_40ft_truck__c IS NOT NULL THEN qli.Quantity / p.Pallets_per_40ft_truck__c
            ELSE NULL
        END AS truck_share
    FROM main.salesforce.silver_quote_line_item qli
    JOIN main.salesforce.silver_procurement_quotation pq
      ON qli.Product2Id = pq.Product__c
    JOIN acc AS supplier 
      ON pq.Supplier_Account__c = supplier.acc_id
    JOIN main.salesforce.silver_product p
      ON qli.Product2Id = p.Id
),

agg_qli AS (
    SELECT 
        quote_id,
        SUM(DISTINCT truck_share) AS total_truck_share,
        AVG(truck_share) AS avg_truck_share,
        count(distinct delivery_date) AS total_delivery_dates_qli,

        AVG(product_per_pallet) AS avg_product_per_pallet, 
        SUM(DISTINCT product_per_pallet) AS sum_product_per_pallet, 
        percentile_cont(0.5) WITHIN GROUP (ORDER BY product_per_truck) AS median_product_per_truck,
        AVG(qli_quantity) AS avg_product_quantity,
        SUM(DISTINCT qli_quantity) AS total_product_quantity,    
        AVG(qli_unit_price) AS avg_product_unit_price,
        SUM(DISTINCT qli_unit_price) AS sum_product_unit_price,
        COUNT(DISTINCT supplier_id) AS num_suppliers,
        COUNT(DISTINCT city) AS num_unique_cities,
        COUNT(DISTINCT country) AS num_unique_countries,
        COUNT(DISTINCT qli_product_id) AS num_unique_product_ids
    FROM qli
    GROUP BY quote_id
),

final AS (
    SELECT 
        q.quote_id,
        q.created_date,
        q.delivery_date,
        q.quote_delivery_country,
        q.quote_delivery_city,
        q.quote_total_sales_price,
        q.quote_gross_margin,
        q.quote_estimated_log_costs,
        -- Fields from agg_qli
        agg_qli.total_truck_share,
        agg_qli.avg_truck_share,
        -- agg_qli.total_delivery_dates_qli,

        agg_qli.avg_product_per_pallet,
        agg_qli.sum_product_per_pallet,
        agg_qli.median_product_per_truck,
        agg_qli.avg_product_quantity,
        agg_qli.total_product_quantity,
        agg_qli.avg_product_unit_price,
        agg_qli.sum_product_unit_price,
        agg_qli.num_suppliers,
        agg_qli.num_unique_cities,
        agg_qli.num_unique_countries,
        agg_qli.num_unique_product_ids
    FROM quot q
    JOIN agg_qli ON q.quote_id = agg_qli.quote_id
),

fact_log_costs AS (
    SELECT 
        so.Quote_ID__c AS quote_id,   
        SUM(Total_net_price__c) AS total_fe_revenue,
        SUM(fe.Total_Logistics_costs_FE__c) AS fact_log_costs_fe
    FROM main.salesforce.silver_fulfillment_event fe
    LEFT JOIN main.salesforce.silver_order so 
      ON fe.Order__c = so.Id
    WHERE so.Quote_ID__c IS NOT NULL
      AND fe.Status__c = 'Completed'
    GROUP BY so.Quote_ID__c
    HAVING SUM(fe.Total_Logistics_costs_FE__c) > 0
)

SELECT
    final.*,
    fl.fact_log_costs_fe
FROM final
JOIN fact_log_costs fl ON final.quote_id = fl.quote_id
where 
1=1
ORDER BY ALL
--AND ABS(total_fe_revenue - quote_total_sales_price) < (0.1 * LEAST(total_fe_revenue, quote_total_sales_price))
"""

quotes_all = spark.sql(query)

# Write to table in Unity Catalog
spark.sql(f"DROP TABLE IF EXISTS {CATALOG_NAME}.{SCHEMA_NAME}.quotes_all_training")
quotes_all.write.saveAsTable(f"{CATALOG_NAME}.{SCHEMA_NAME}.quotes_all_training")




# COMMAND ----------

# MAGIC %md ### Preprocess data

# COMMAND ----------

data = spark.read.table(f"{CATALOG_NAME}.{SCHEMA_NAME}.quotes_all_training").toPandas()

def hash_encoding(val, num_bins=1000):
    return int(hashlib.sha256(str(val).encode()).hexdigest(), 16) % num_bins

def preprocess_data(df, train_stats=None):
    """Simplified preprocessing with leakage prevention"""
    df = df.copy()
    
    # Initialize training stats
    is_train = train_stats is None
    if is_train:
        train_stats = {}
        
    # Date features
    for col in ['created_date', 'delivery_date']:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    df['delivery_days'] = (df['delivery_date'] - df['created_date']).dt.days
    
    # Numeric columns
    num_cols = ['quote_total_sales_price', 'quote_gross_margin', 'total_product_quantity',
                'num_suppliers', 'num_unique_countries', 'total_truck_share' #, 'total_delivery_dates_qli'
                ]
    for col in num_cols:
        if is_train:
            train_stats[f'{col}_median'] = df[col].median()
        df[col].fillna(train_stats[f'{col}_median'], inplace=True)
    
    # Hash encoding
    for col in ['quote_delivery_country', 'quote_delivery_city']:
        df[f'{col}_hash'] = df[col].apply(lambda x: hash_encoding(x))
    
    # Target and features
    y = df['fact_log_costs_fe']
    X = df.drop(columns=['quote_id', 'created_date', 'delivery_date', 
                        'quote_estimated_log_costs', 'fact_log_costs_fe',
                        'quote_delivery_country', 'quote_delivery_city'])
    
    return (X, y, train_stats) if is_train else (X, y)

# Split and preprocess data
train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)
X_train, y_train, train_stats = preprocess_data(train_df)
X_test, y_test = preprocess_data(test_df, train_stats)

# COMMAND ----------

def clean_and_preprocess_data(data, train_stats=None):
    """Process raw quote data using training parameters to prevent leakage"""
    data = data.copy()
    
    # Initialize training statistics storage
    if train_stats is None:
        train_stats = {'is_train': True}
    else:
        train_stats['is_train'] = False

    # 1) Date Conversion
    for col in ['created_date', 'delivery_date']:
        if col in data.columns:
            data[col] = pd.to_datetime(data[col], errors='coerce')

    # 2) Numeric Type Conversion
    numeric_cols = [
        'quote_total_sales_price', 'quote_gross_margin', 'total_product_quantity',
        'avg_product_quantity', 'avg_product_unit_price', 'num_suppliers',
        'num_unique_countries', 'num_unique_cities', 'num_unique_product_ids'
    ]
    
    for col in numeric_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')

    # 3) Product-per Features Handling (with correct column names)
    if {'avg_product_per_pallet', 'median_product_per_truck'}.issubset(data.columns):
        valid_mask = data['avg_product_per_pallet'].notna() & \
                    data['median_product_per_truck'].notna() & \
                    (data['avg_product_per_pallet'] != 0)
        
        if train_stats['is_train']:
            median_ratio = (data.loc[valid_mask, 'median_product_per_truck'] / 
                           data.loc[valid_mask, 'avg_product_per_pallet']).median()
            train_stats['product_ratio'] = median_ratio if not np.isnan(median_ratio) else 1.0
            train_stats['pallet_median'] = data['avg_product_per_pallet'].median()
            train_stats['truck_median'] = data['median_product_per_truck'].median()
        
        # Cross-impute missing values using trained ratio
        mask1 = data['avg_product_per_pallet'].notna() & data['median_product_per_truck'].isna()
        data.loc[mask1, 'median_product_per_truck'] = data.loc[mask1, 'avg_product_per_pallet'] * train_stats['product_ratio']

        mask2 = data['median_product_per_truck'].notna() & data['avg_product_per_pallet'].isna()
        data.loc[mask2, 'avg_product_per_pallet'] = data.loc[mask2, 'median_product_per_truck'] / train_stats['product_ratio']

        # Final imputation using trained medians
        data['avg_product_per_pallet'].fillna(train_stats['pallet_median'], inplace=True)
        data['median_product_per_truck'].fillna(train_stats['truck_median'], inplace=True)

    # 4) Missing Value Imputation with trained medians
    for col in numeric_cols:
        if col in data.columns:
            if train_stats['is_train']:
                train_stats[f'{col}_median'] = data[col].median()
            data[col].fillna(train_stats[f'{col}_median'], inplace=True)

    # 5) Truck Share Features
    truck_share_cols = ['total_truck_share', 'avg_truck_share']
    for col in truck_share_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
            if train_stats['is_train']:
                train_stats[f'{col}_median'] = data[col].median()
            data[col].fillna(train_stats[f'{col}_median'], inplace=True)

    # 6) Feature Engineering
    if {'created_date', 'delivery_date'}.issubset(data.columns):
        data['delivery_days'] = (data['delivery_date'] - data['created_date']).dt.days
    
    if 'quote_total_sales_price' in data.columns:
        # More accurate price per unit calculation
        data['log_price_per_unit'] = np.log(data['quote_total_sales_price'] / 
                                          (data['total_product_quantity'] + 1e-6))
        data['margin_ratio'] = data['quote_gross_margin'] / (data['quote_total_sales_price'] + 1e-6)

    # 7) Safe Categorical Encoding
    cat_cols = ['quote_delivery_country', 'quote_delivery_city']
    for col in cat_cols:
        if col in data.columns:
            data[f'{col}_hash'] = data[col].apply(lambda x: hash_encoding(x))
    data.drop(columns=cat_cols, errors='ignore', inplace=True)

    # 8) Final Sanitization
    numeric_features = data.select_dtypes(include=np.number).columns
    for col in numeric_features:
        data[col] = data[col].replace([np.inf, -np.inf], np.nan)
        if train_stats['is_train'] and f'{col}_final_median' not in train_stats:
            train_stats[f'{col}_final_median'] = data[col].median()
        data[col].fillna(train_stats[f'{col}_final_median'], inplace=True)

    # 9) Prepare Features
    excluded = ['quote_id', 'quote_estimated_log_costs', 'created_date', 
                'delivery_date', 'fact_log_costs_fe', 'margin_ratio','avg_product_per_pallet','sum_product_per_pallet', 'avg_truck_share',]
    y = data['fact_log_costs_fe']
    X = data.drop(columns=[c for c in excluded if c in data], errors='ignore')
    
    return (X, y, train_stats) if train_stats['is_train'] else (X, y)



def hash_encoding(val, num_bins=1000):
    """Consistent hashing without leakage"""
    return int(hashlib.sha256(str(val).encode()).hexdigest(), 16) % num_bins

# Proper usage with leakage prevention
# Split first to prevent leakage
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Process training data and get parameters
X_train, y_train, train_stats = clean_and_preprocess_data(train_data)

# Process test data with training parameters
del train_stats['is_train']  # Remove training flag
X_test, y_test = clean_and_preprocess_data(test_data, train_stats=train_stats)

# COMMAND ----------

display(y_train)

# COMMAND ----------

display(X_train)

# COMMAND ----------

mlflow.autolog(disable=True)
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

# Define the RMSE function
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Enable MLflow autologging
mlflow.autolog(disable=False)

# Start MLflow run for the best model training
with mlflow.start_run(run_name='xgb_best_model'):
    
    # Initialize the best model with parameters from best_params
    best_model = XGBRegressor(**best_params)
    # best_model = XGBRegressor()
    # Fit the model with early stopping on the test set
    best_model.fit(
        X_train, 
        y_train
    )

    # Predictions on the test set
    y_pred_test = best_model.predict(X_test)

    # Compute metrics for XGBoost predictions
    rmse_xgb_test = rmse(y_test, y_pred_test)  # Calculate RMSE using our custom function
    mape_xgb_test = mean_absolute_percentage_error(y_test, y_pred_test)
    r2_xgb_test = r2_score(y_test, y_pred_test)
    
    # Log metrics to MLflow
    mlflow.log_metrics({
        "rmse_xgb_test": rmse_xgb_test,
        "mape_xgb_test": mape_xgb_test,
        "r2_xgb_test": r2_xgb_test
    })
    
    # Print test metrics
    print(f"\n[TEST] RMSE (XGBoost): {rmse_xgb_test:.2f}")
    print(f"[TEST] MAPE (XGBoost): {mape_xgb_test:.2f}")
    print(f"[TEST] R-squared (XGBoost): {r2_xgb_test:.2f}")

    # ----------------------------------
    # External Estimator Evaluation
    # ----------------------------------
    if 'quote_estimated_log_costs' in data.columns:
        # Assuming the external predictions are stored in the data DataFrame aligned with the test set indices
        external_preds_test = data.loc[X_test.index, 'quote_estimated_log_costs'].copy()
        
        rmse_external = rmse(y_test, external_preds_test)
        mape_external = mean_absolute_percentage_error(y_test, external_preds_test)
        r2_external = r2_score(y_test, external_preds_test)
        
        mlflow.log_metrics({
            "rmse_external": rmse_external,
            "mape_external": mape_external,
            "r2_external": r2_external
        })
        
        print(f"\n[TEST] RMSE (External Estimates): {rmse_external:.2f}")
        print(f"[TEST] MAPE (External Estimates): {mape_external:.2f}")
        print(f"[TEST] R-squared (External Estimates): {r2_external:.2f}")

# COMMAND ----------

# Search all runs in the current experiment
runs = mlflow.search_runs()
best_run = runs.sort_values('metrics.rmse_xgb_test', ascending=True).iloc[0]
best_run_id = best_run.run_id

print(f"Best Run ID: {best_run_id}")
print(f"Best RMSE: {best_run['metrics.rmse_xgb_test']}")

display(best_run)
# Display metrics
print("Best Run Metrics:")
print(f"RMSE: {best_run['metrics.rmse_xgb_test']}")
print(f"MAPE: {best_run['metrics.mape_xgb_test']}")
print(f"R-squared: {best_run['metrics.r2_xgb_test']}")

# COMMAND ----------

model_uri = 'runs:/{run_id}/model'.format(
    run_id=best_run.run_id
)

mlflow.register_model(model_uri, f"{CATALOG_NAME}.{SCHEMA_NAME}.xgb_best_model")

# Set alias for the new version
client = mlflow.tracking.MlflowClient()

latest_version = client.search_model_versions(
    f"name='{CATALOG_NAME}.{SCHEMA_NAME}.xgb_best_model'"
)[0].version

client.set_registered_model_alias(
    name=f"{CATALOG_NAME}.{SCHEMA_NAME}.xgb_best_model",
    alias="Production",
    version=latest_version
)

# Get the deployment client
deploy_client = get_deploy_client("databricks")

# Update the endpoint
endpoint = deploy_client.update_endpoint(
    endpoint="xgb_log_costs_test",
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

print('finished, test predictions in 10 minutes')

# COMMAND ----------

mlflow.autolog(disable=True)
