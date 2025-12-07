#!/usr/bin/env python3
import os
import time

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    IntegerType, LongType, ShortType,
    FloatType, DoubleType, DecimalType
)

from pyspark.ml.pipeline import PipelineModel
from pyspark.ml.feature import VectorAssembler



# CONFIG


BASE_DIR = "/storage/work/ksc5629/Course_Project/"

DATA_PATH = os.path.join(BASE_DIR, "data/parquet/train.pqt")
MODEL_PATH = os.path.join(BASE_DIR, "notebooks/models/best_model")
RESULT_DIR = os.path.join(BASE_DIR, "notebooks/cluster_results_2")

os.makedirs(RESULT_DIR, exist_ok=True)

LOG_PATH = os.path.join(RESULT_DIR, "cluster_log.txt")
SUMMARY_PATH = os.path.join(RESULT_DIR, "summary_stats.txt")
PRED_DIST_PATH = os.path.join(RESULT_DIR, "prediction_distribution.txt")
RUNTIME_PATH = os.path.join(RESULT_DIR, "runtime_report.txt")
PARTITION_PATH = os.path.join(RESULT_DIR, "partition_report.txt")



# SIMPLE LOGGER

LOG_LINES = []


def log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    LOG_LINES.append(line)


def flush_log():
    with open(LOG_PATH, "w") as f:
        f.write("\n".join(LOG_LINES) + "\n")



# MAIN


if __name__ == "__main__":
    start_time = time.time()
    log("===== CLUSTER EVALUATION STARTED =====")
    log(f"BASE_DIR: {BASE_DIR}")
    log(f"DATA_PATH: {DATA_PATH}")
    log(f"MODEL_PATH: {MODEL_PATH}")
    log(f"RESULT_DIR: {RESULT_DIR}")

    
    # SparkSession
    
    spark = (
        SparkSession.builder
        .appName("EEG_Cluster_Evaluation")
        .getOrCreate()
    )

    sc = spark.sparkContext
    log(f"Spark master: {sc.master}")
    log(f"Spark app name: {sc.appName}")
    log(f"Spark version: {sc.version}")

    
    # Load dataset
    
    log("Loading dataset...")
    try:
        df = spark.read.parquet(DATA_PATH)
    except Exception as e:
        log(f"ERROR: failed to read parquet at {DATA_PATH}: {e}")
        flush_log()
        spark.stop()
        raise

    log("Dataset loaded.")
    row_count = df.count()
    num_cols = len(df.columns)
    num_partitions = df.rdd.getNumPartitions()

    log(f"Row count: {row_count}")
    log(f"Num columns: {num_cols}")
    log(f"Num partitions: {num_partitions}")
    log(f"First 10 columns: {df.columns[:10]}")

    # Cache for subsequent operations
    df = df.repartition(num_partitions).cache()
    df.count()  # materialize cache

    
    # Partition report
    
    partition_sizes = (
        df.rdd
        .mapPartitions(lambda it: [sum(1 for _ in it)])
        .collect()
    )

    with open(PARTITION_PATH, "w") as f:
        f.write(f"Total partitions: {num_partitions}\n")
        for i, size in enumerate(partition_sizes):
            f.write(f"Partition {i}: {size} rows\n")

    log(f"Wrote partition report to {PARTITION_PATH}")

    
    # Distributed summary stats (on a subset of numeric columns)

    
    log("Computing distributed summary statistics on numeric columns...")

    numeric_types = (
        IntegerType, LongType, ShortType,
        FloatType, DoubleType, DecimalType
    )

    numeric_cols = [
        field.name
        for field in df.schema.fields
        if isinstance(field.dataType, numeric_types)
    ]

    log(f"Detected {len(numeric_cols)} numeric columns.")
    if len(numeric_cols) > 0:
        subset_cols = numeric_cols[:20]  # limit for safety
        log(f"Using first {len(subset_cols)} numeric columns for summary stats.")

        desc_df = df.select(*subset_cols).describe()
        # Collect summary to driver and write to text file
        desc_pd = desc_df.toPandas()
        with open(SUMMARY_PATH, "w") as f:
            f.write("Summary statistics (describe) on subset of numeric columns:\n\n")
            f.write(desc_pd.to_string(index=False))
            f.write("\n")

        log(f"Wrote summary stats to {SUMMARY_PATH}")
    else:
        log("No numeric columns found. Skipping summary stats.")

    
    # Group-based aggregation (shows distributed grouping)
    
    if "patient_id" in df.columns:
        log("Computing group-based aggregation by patient_id...")
        # this stays lightweight but clearly uses distributed aggregation
        grp_df = (
            df.groupBy("patient_id")
            .agg(
                F.count("*").alias("n_rows"),
                F.mean("seizure_vote").alias("mean_seizure_vote")
                if "seizure_vote" in df.columns else F.lit(None).alias("mean_seizure_vote")
            )
        )
        # Do not collect entire DF, just show count and a few rows
        grp_count = grp_df.count()
        log(f"GroupBy patient_id produced {grp_count} groups.")
    else:
        log("Column patient_id not found. Skipping group-based aggregation.")

    
    # Load saved PCA + RF model
    
    log("Loading saved PipelineModel (PCA + RF)...")
    try:
        best_model = PipelineModel.load(MODEL_PATH)
    except Exception as e:
        log(f"ERROR: Failed to load model from {MODEL_PATH}: {e}")
        flush_log()
        spark.stop()
        raise

    log("Model loaded successfully.")

    # Try to find a VectorAssembler stage to infer required feature columns
    assembler_stage = None
    for stage in best_model.stages:
        if isinstance(stage, VectorAssembler):
            assembler_stage = stage
            break

    if assembler_stage is None:
        log("WARNING: No VectorAssembler stage found in PipelineModel. "
            "Will attempt transform() directly and see if it works.")
        required_input_cols = set()
    else:
        required_input_cols = set(assembler_stage.getInputCols())
        log(f"Assembler input columns (truncated): {list(required_input_cols)[:10]} ...")

    
    # Attempt distributed inference
    
    inference_start = time.time()
    can_run_inference = True

    if required_input_cols:
        missing = required_input_cols.difference(df.columns)
        if missing:
            log("WARNING: Cannot run model.transform(df) because some required "
                f"feature columns are missing. Missing (truncated): {list(missing)[:10]} ...")
            can_run_inference = False

    if can_run_inference:
        log("Running distributed inference with saved PCA+RF model...")
        try:
            pred_df = best_model.transform(df)
        except Exception as e:
            log(f"ERROR during model.transform(df): {e}")
            pred_df = None
            can_run_inference = False

    inference_end = time.time()

    if can_run_inference and pred_df is not None:
        # Count predictions
        pred_count = pred_df.count()
        log(f"Inference completed. Rows with predictions: {pred_count}")

        # Check prediction distribution
        if "prediction" in pred_df.columns:
            pred_dist = (
                pred_df.groupBy("prediction")
                .agg(F.count("*").alias("count"))
                .orderBy("prediction")
            )

            dist_rows = pred_dist.collect()
            with open(PRED_DIST_PATH, "w") as f:
                f.write("Prediction distribution:\n")
                for row in dist_rows:
                    f.write(f"prediction={row['prediction']}, count={row['count']}\n")

            log(f"Wrote prediction distribution to {PRED_DIST_PATH}")

        # Save a subset of predictions to parquet for inspection
        pred_out_path = os.path.join(RESULT_DIR, "predictions")
        cols_to_save = []
        for c in ["eeg_id", "spec_id", "patient_id", "prediction"]:
            if c in pred_df.columns:
                cols_to_save.append(c)
        if not cols_to_save:
            cols_to_save = ["prediction"]

        (pred_df
         .select(*cols_to_save)
         .write.mode("overwrite")
         .parquet(pred_out_path))

        log(f"Saved prediction subset to {pred_out_path}")
    else:
        log("Skipped inference or inference failed; no predictions saved.")

    
    # Runtime report
    
    total_time = time.time() - start_time
    inference_time = (
        inference_end - inference_start
        if 'inference_end' in locals() and 'inference_start' in locals()
        else None
    )

    with open(RUNTIME_PATH, "w") as f:
        f.write(f"Total runtime (seconds): {total_time:.2f}\n")
        if inference_time is not None and can_run_inference:
            f.write(f"Inference runtime (seconds): {inference_time:.2f}\n")
        else:
            f.write("Inference runtime: N/A (inference skipped or failed)\n")

        f.write(f"Row count: {row_count}\n")
        f.write(f"Num partitions: {num_partitions}\n")

    log(f"Wrote runtime report to {RUNTIME_PATH}")
    log(f"Total runtime: {total_time:.2f} seconds")

    log("===== CLUSTER EVALUATION FINISHED =====")
    flush_log()
    spark.stop()
