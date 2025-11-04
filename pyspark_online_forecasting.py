# Online Forecasting with PySpark and scikit-learn

# This script demonstrates an online forecasting pipeline using PySpark's Structured Streaming
# for data processing and scikit-learn for model training.

# Import necessary libraries
import os
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.window import Window
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
from datetime import datetime

def main():
    # Create output directory
    os.makedirs('outputs', exist_ok=True)
    
    # Redirect stdout to capture all prints
    class Logger:
        def __init__(self):
            self.terminal = sys.stdout
            self.log = open('outputs/console_output.txt', 'w')
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()
        def flush(self):
            pass
    
    sys.stdout = Logger()
    
    # Initialize Spark Session
    spark = SparkSession.builder.appName("OnlineForecasting").getOrCreate()
    print("Spark Session Created")
    print(f"Run started at: {datetime.now()}")

    # --- 1) Data Loading and Feature Engineering ---

    # Define schema for the CSV file
    schema = StructType([
        StructField("Date", StringType(), True),
        StructField("Time", StringType(), True),
        StructField("CO(GT)", DoubleType(), True),
        StructField("PT08.S1(CO)", DoubleType(), True),
        StructField("NMHC(GT)", DoubleType(), True),
        StructField("C6H6(GT)", DoubleType(), True),
        StructField("PT08.S2(NMHC)", DoubleType(), True),
        StructField("NOx(GT)", DoubleType(), True),
        StructField("PT08.S3(NOx)", DoubleType(), True),
        StructField("NO2(GT)", DoubleType(), True),
        StructField("PT08.S4(NO2)", DoubleType(), True),
        StructField("PT08.S5(O3)", DoubleType(), True),
        StructField("T", DoubleType(), True),
        StructField("RH", DoubleType(), True),
        StructField("AH", DoubleType(), True),
        StructField("_c15", StringType(), True),
        StructField("_c16", StringType(), True)
    ])

    # Read the data as batch
    df = spark.read.format("csv").option("header", "true").option("sep", ";").schema(schema).load("Data/AirQualityUCI.csv")

    # Data Cleaning and Preparation
    df = df.withColumn("datetime_str", concat_ws(" ", col("Date"), col("Time")))
    df = df.withColumn("datetime", to_timestamp("datetime_str", "dd/MM/yyyy HH.mm.ss"))
    df = df.na.replace(-200, None)
    df = df.drop("Date", "Time", "datetime_str", "_c15", "_c16")

    # Sanitize column names
    new_columns = [c.replace('.', '_').replace('(', '').replace(')', '') for c in df.columns]
    df = df.toDF(*new_columns)

    # Feature Engineering
    window = Window.orderBy("datetime")

    # Lag features
    feature_cols = [c for c in df.columns if c != "datetime"]
    for c in feature_cols:
        for l in [1, 2, 3, 6, 12, 24]:
            df = df.withColumn(f"{c}_lag{l}", lag(c, l).over(window))

    # Rolling features
    for c in feature_cols:
        for w in [3, 6, 12]:
            df = df.withColumn(f"{c}_rollmean{w}", avg(c).over(window.rowsBetween(-w, -1)))
            df = df.withColumn(f"{c}_rollstd{w}", stddev(c).over(window.rowsBetween(-w, -1)))

    # Calendar features
    df = df.withColumn("hour", hour(col("datetime")))
    df = df.withColumn("dow", dayofweek(col("datetime")))
    df = df.withColumn("month", month(col("datetime")))
    df = df.withColumn("is_weekend", when(col("dow").isin([1, 7]), 1).otherwise(0))

    # Target variable
    TARGET = "NO2_GT"
    HORIZON = 1
    df = df.withColumn("target", lead(col(TARGET), HORIZON).over(window))

    df = df.dropna()

    # --- 2) Model Training and Prediction ---
    
    # Convert to pandas for processing
    pandas_df = df.toPandas()
    print(f"Dataset shape: {pandas_df.shape}")
    
    # Prepare features and target
    feature_cols = [c for c in pandas_df.columns if c not in ["datetime", "target"]]
    X = pandas_df[feature_cols]
    y = pandas_df["target"]
    
    # Split data for online simulation
    split_idx = int(len(pandas_df) * 0.7)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Initialize and train model
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", SGDRegressor(random_state=42))
    ])
    
    print("Training initial model...")
    model.fit(X_train, y_train)
    
    # Online prediction simulation
    predictions = []
    actuals = []
    
    print("Simulating online predictions...")
    for i in range(len(X_test)):
        # Predict
        pred = model.predict(X_test.iloc[i:i+1])[0]
        actual = y_test.iloc[i]
        
        predictions.append(pred)
        actuals.append(actual)
        
        # Update model with new data point
        model.named_steps["model"].partial_fit(
            model.named_steps["scaler"].transform(
                model.named_steps["imputer"].transform(X_test.iloc[i:i+1])
            ), 
            [actual]
        )
    
    # Create results dataframe
    all_results = pd.DataFrame({
        'datetime': pandas_df['datetime'][split_idx:].reset_index(drop=True),
        'actual': actuals,
        'prediction': predictions
    })
    
    # --- 3) Evaluation and Saving Results ---
    all_results['error'] = (all_results['actual'] - all_results['prediction']).abs()
    
    mae = mean_absolute_error(all_results['actual'], all_results['prediction'])
    rmse = root_mean_squared_error(all_results['actual'], all_results['prediction'])

    print(f"Online MAE: {mae:.4f}")
    print(f"Online RMSE: {rmse:.4f}")

    # Save results to CSV in outputs folder
    all_results.to_csv('outputs/forecasting_results.csv', index=False)
    print("Results saved to outputs/forecasting_results.csv")
    
    # Save summary report
    summary_report = f"""Air Quality Forecasting Results Summary
{'='*50}
Run Date: {datetime.now()}
Dataset Shape: {pandas_df.shape}
Target Variable: {TARGET}
Forecast Horizon: {HORIZON} hour(s)
Training Split: 70% train, 30% test

Model Performance:
- Mean Absolute Error (MAE): {mae:.4f}
- Root Mean Squared Error (RMSE): {rmse:.4f}

Feature Engineering:
- Lag features: 1, 2, 3, 6, 12, 24 hours
- Rolling statistics: 3, 6, 12 hour windows
- Calendar features: hour, day of week, month, weekend indicator
- Total features: {len(feature_cols)}

Output Files:
- forecasting_results.csv: Detailed predictions and actuals
- pyspark_online_forecasting.png: Visualization plots
- console_output.txt: Complete console log
- summary_report.txt: This summary
"""
    
    with open('outputs/summary_report.txt', 'w') as f:
        f.write(summary_report)
    print("Summary report saved to outputs/summary_report.txt")

    # --- 4) Visualization ---
    plt.figure(figsize=(12, 6))

    # Plot predictions vs actuals
    plt.subplot(2, 1, 1)
    plt.plot(all_results['datetime'], all_results['actual'], label="Actual")
    plt.plot(all_results['datetime'], all_results['prediction'], label="Online Forecast (PySpark + SGD)", alpha=0.7)
    plt.title(f"{TARGET} - Online Forecast vs Actual (H+{HORIZON})")
    plt.xlabel("Time")
    plt.ylabel(TARGET)
    plt.legend()

    # Plot rolling MAE
    plt.subplot(2, 1, 2)
    rolling_mae = all_results['error'].rolling(window=100).mean()
    plt.plot(all_results['datetime'], rolling_mae.values, label="Rolling MAE (100 samples)")
    plt.title("Online Model Performance (Rolling MAE)")
    plt.xlabel("Time")
    plt.ylabel("Mean Absolute Error")
    plt.legend()

    plt.tight_layout()
    plt.savefig("outputs/pyspark_online_forecasting.png", dpi=150, bbox_inches='tight')
    print("Visualization saved to outputs/pyspark_online_forecasting.png")
    plt.close()
    
    # Save feature importance (top 20)
    feature_names = feature_cols
    if hasattr(model.named_steps['model'], 'coef_'):
        importance = abs(model.named_steps['model'].coef_)
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False).head(20)
        feature_importance.to_csv('outputs/feature_importance.csv', index=False)
        print("Feature importance saved to outputs/feature_importance.csv")
    
    print(f"\nAll outputs saved to 'outputs/' directory")
    print(f"Run completed at: {datetime.now()}")
    
    # Stop Spark session
    spark.stop()
    
    # Restore stdout
    sys.stdout.log.close()
    sys.stdout = sys.stdout.terminal

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        with open('outputs/error_log.txt', 'w') as f:
            f.write(f"Error occurred at {datetime.now()}\n")
            f.write(f"Error: {str(e)}\n")
        raise
