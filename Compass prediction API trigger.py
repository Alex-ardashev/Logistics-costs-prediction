# Databricks notebook source
# MAGIC %md
# MAGIC # Triggering predictions

# COMMAND ----------

# MAGIC %md
# MAGIC ## Optimized query

# COMMAND ----------

# MAGIC %sql
# MAGIC WITH quot AS (
# MAGIC     SELECT  
# MAGIC         q.Id AS quote_id,
# MAGIC         CAST(q.createdDate AS DATE) AS created_date,
# MAGIC         CAST(Delivery_date_min__c AS DATE) AS delivery_date,
# MAGIC         Total_Sales_price__c AS quote_total_sales_price,
# MAGIC         COALESCE(Estimated_Logistics_costs__c, estimated_logistics_costs_automated__c) AS quote_estimated_log_costs,
# MAGIC         Estimated_Logistics_cost_ML__c AS quote_estimated_log_costs_ml,
# MAGIC         Gross_Margin__c AS quote_gross_margin,
# MAGIC         ShippingCountry AS quote_delivery_country,
# MAGIC         ShippingCity AS quote_delivery_city
# MAGIC     FROM main.salesforce.silver_quote q
# MAGIC     JOIN main.salesforce.silver_opportunity op
# MAGIC       ON q.OpportunityId = op.Id
# MAGIC     WHERE 1=1 
# MAGIC       AND lower(BillingName) NOT LIKE '%ummy%'
# MAGIC       AND lower(BillingName) NOT LIKE '%ndercore%'
# MAGIC       AND q.Id = '0Q0Jy0000047RijKAE'
# MAGIC ),
# MAGIC
# MAGIC acc AS (
# MAGIC     SELECT 
# MAGIC         Id AS acc_id,
# MAGIC         name,
# MAGIC         ShippingCity AS acc_shipping_city,
# MAGIC         ShippingCountry AS acc_shipping_country
# MAGIC     FROM main.salesforce.silver_account
# MAGIC     WHERE RecordTypeId IN ('01209000000qLEbAAM', '01209000000qLEWAA2', '01209000000qLEgAAM')
# MAGIC       AND Id <> '001Jy000009ys3nIAA' -- andercore placeholder
# MAGIC       AND lower(name) NOT LIKE '%ummy%'
# MAGIC       AND lower(name) NOT LIKE '%ndercore%'
# MAGIC ),
# MAGIC
# MAGIC qli AS (
# MAGIC     SELECT 
# MAGIC         qli.QuoteId AS quote_id,
# MAGIC         qli.Quantity AS qli_quantity,
# MAGIC         qli.unitPrice AS qli_unit_price,
# MAGIC         qli.Product2Id AS qli_product_id,
# MAGIC         pq.Supplier_Account__c AS supplier_id,
# MAGIC         supplier.acc_shipping_city AS city,
# MAGIC         supplier.acc_shipping_country AS country,
# MAGIC         p.Units_per_pallet__c AS product_per_pallet, 
# MAGIC         p.Pallets_per_40ft_truck__c AS product_per_truck,
# MAGIC         CASE 
# MAGIC             WHEN p.Pallets_per_40ft_truck__c IS NOT NULL THEN qli.Quantity / p.Pallets_per_40ft_truck__c
# MAGIC             ELSE NULL
# MAGIC         END AS truck_share
# MAGIC     FROM main.salesforce.silver_quote_line_item qli
# MAGIC     JOIN main.salesforce.silver_procurement_quotation pq
# MAGIC       ON qli.Product2Id = pq.Product__c
# MAGIC     JOIN acc AS supplier 
# MAGIC       ON pq.Supplier_Account__c = supplier.acc_id
# MAGIC     JOIN main.salesforce.silver_product p
# MAGIC       ON qli.Product2Id = p.Id
# MAGIC ),
# MAGIC
# MAGIC agg_qli AS (
# MAGIC     SELECT 
# MAGIC         quote_id,
# MAGIC         SUM(DISTINCT truck_share) AS total_truck_share,
# MAGIC         AVG(truck_share) AS avg_truck_share,
# MAGIC         AVG(product_per_pallet) AS avg_product_per_pallet, 
# MAGIC         SUM(DISTINCT product_per_pallet) AS sum_product_per_pallet, 
# MAGIC         percentile_cont(0.5) WITHIN GROUP (ORDER BY product_per_truck) AS median_product_per_truck,
# MAGIC         AVG(qli_quantity) AS avg_product_quantity,
# MAGIC         SUM(DISTINCT qli_quantity) AS total_product_quantity,    
# MAGIC         AVG(qli_unit_price) AS avg_product_unit_price,
# MAGIC         SUM(DISTINCT qli_unit_price) AS sum_product_unit_price,
# MAGIC         COUNT(DISTINCT supplier_id) AS num_suppliers,
# MAGIC         COUNT(DISTINCT city) AS num_unique_cities,
# MAGIC         COUNT(DISTINCT country) AS num_unique_countries,
# MAGIC         COUNT(DISTINCT qli_product_id) AS num_unique_product_ids
# MAGIC     FROM qli
# MAGIC     GROUP BY quote_id
# MAGIC ),
# MAGIC
# MAGIC final AS (
# MAGIC     SELECT 
# MAGIC         q.quote_id,
# MAGIC         q.created_date,
# MAGIC         q.delivery_date,
# MAGIC         q.quote_delivery_country,
# MAGIC         q.quote_delivery_city,
# MAGIC         q.quote_total_sales_price,
# MAGIC         q.quote_gross_margin,
# MAGIC         q.quote_estimated_log_costs,
# MAGIC         -- Fields from agg_qli
# MAGIC         agg_qli.total_truck_share,
# MAGIC         agg_qli.avg_truck_share,
# MAGIC         agg_qli.avg_product_per_pallet,
# MAGIC         agg_qli.sum_product_per_pallet,
# MAGIC         agg_qli.median_product_per_truck,
# MAGIC         agg_qli.avg_product_quantity,
# MAGIC         agg_qli.total_product_quantity,
# MAGIC         agg_qli.avg_product_unit_price,
# MAGIC         agg_qli.sum_product_unit_price,
# MAGIC         agg_qli.num_suppliers,
# MAGIC         agg_qli.num_unique_cities,
# MAGIC         agg_qli.num_unique_countries,
# MAGIC         agg_qli.num_unique_product_ids
# MAGIC     FROM quot q
# MAGIC     JOIN agg_qli ON q.quote_id = agg_qli.quote_id
# MAGIC ),
# MAGIC
# MAGIC fact_log_costs AS (
# MAGIC     SELECT 
# MAGIC         so.Quote_ID__c AS quote_id,   
# MAGIC         SUM(Total_net_price__c) AS total_fe_revenue,
# MAGIC         SUM(fe.Total_Logistics_costs_FE__c) AS fact_log_costs_fe
# MAGIC     FROM main.salesforce.silver_fulfillment_event fe
# MAGIC     LEFT JOIN main.salesforce.silver_order so 
# MAGIC       ON fe.Order__c = so.Id
# MAGIC     WHERE so.Quote_ID__c IS NOT NULL
# MAGIC       AND fe.Status__c = 'Completed'
# MAGIC     GROUP BY so.Quote_ID__c
# MAGIC     HAVING SUM(fe.Total_Logistics_costs_FE__c) > 0
# MAGIC )
# MAGIC
# MAGIC SELECT
# MAGIC     final.*,
# MAGIC     fl.fact_log_costs_fe
# MAGIC FROM final
# MAGIC left JOIN fact_log_costs fl ON final.quote_id = fl.quote_id
# MAGIC

# COMMAND ----------

# ------------------------------------------
# 1. Fetch enriched data using Spark + SQL
# ------------------------------------------
quote_id = "0Q0Jy0000047RijKAE"
query = f"""
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
      AND q.Id = '{quote_id}'
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
left JOIN fact_log_costs fl ON final.quote_id = fl.quote_id
"""

spark_df = spark.sql(query)
data = spark_df.toPandas()
display(data)

# COMMAND ----------

import numpy as np
import pandas as pd
import hashlib

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
                'num_suppliers', 'num_unique_countries', 'total_truck_share']
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
    
    return X

# COMMAND ----------

# input schema
{
  "quote_total_sales_price": "double",
  "quote_gross_margin": "double",
  "total_truck_share": "double",
  "avg_truck_share": "double",
  "avg_product_per_pallet": "double",
  "sum_product_per_pallet": "double",
  "median_product_per_truck": "double",
  "avg_product_quantity": "double",
  "total_product_quantity": "long",
  "avg_product_unit_price": "double",
  "sum_product_unit_price": "double",
  "num_suppliers": "long",
  "num_unique_cities": "long",
  "num_unique_countries": "long",
  "num_unique_product_ids": "long",
  "delivery_days": "double",
  "quote_delivery_country_hash": "long",
  "quote_delivery_city_hash": "long"
}


final_data = preprocess_data(data)
display(final_data)


# COMMAND ----------

import requests

session = requests.Session() 
auth_token = "xxx"
endpoint_url = "xxx"

headers = {
       "Authorization": f"Bearer {auth_token}",
       "Content-Type": "application/json"
   }

payload = {"dataframe_records": final_data.to_dict(orient="records")}

response = session.post(endpoint_url, json=payload, headers=headers)

print(payload)
print(response.text)