# cluster_eval.py
# ---------------------------------------------
# Distributed evaluation of saved RF+PCA model
# ---------------------------------------------

from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import col
import time

# ---------------------------------------------
# 1. Start Spark cluster session
# ---------------------------------------------
spark = (
    SparkSession.builder
        .appName("EEG_Cluster_Evaluation")
        .config("spark.executor.memory", "8g")
        .config("spark.executor.cores", "1")
        .config("spark.driver.memory", "8g")
        .config("spark.sql.shuffle.partitions", "200")
        .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

print("\n===== CLUSTER EVALUATION STARTED =====\n")

# ---------------------------------------------
# 2. Load dataset (Parquet)
# ---------------------------------------------
print("Loading dataset...")

df = spark.read.parquet("data/parquet/train.pqt")

print("Dataset loaded.")
print("Row count:", df.count())
print("Columns:", df.columns[:10], "...")

# Ensure binary_label exists:
if "binary_label" not in df.columns:
    print("ERROR: binary_label column missing.")
    spark.stop()
    exit()

# ---------------------------------------------
# 3. Load saved RF+PCA model
# ---------------------------------------------
print("\nLoading saved model from disk...")

model_path = "models/best_rf_pca_model"
model = PipelineModel.load(model_path)

print("Model loaded successfully.\n")

# ---------------------------------------------
# 4. Distributed Test 1:
#    Full-dataset inference timing
# ---------------------------------------------
print("Running full-dataset distributed inference...")

start = time.time()
pred_df = model.transform(df)
pred_df.count()  # force computation
end = time.time()

print("\n[TEST 1] Full-dataset inference time:", round(end - start, 2), "seconds")

# ---------------------------------------------
# 5. Distributed Test 2:
#    Shuffle-heavy evaluation (trigger large shuffles)
# ---------------------------------------------
print("\nRunning shuffle stress test...")

shuffle_start = time.time()
class_distribution = (
    pred_df.groupBy("binary_label", "prediction")
           .count()
           .orderBy("binary_label", "prediction")
)
shuffle_results = class_distribution.collect()
shuffle_end = time.time()

print("[TEST 2] Shuffle stress test completed in:", round(shuffle_end - shuffle_start, 2), "seconds")
print("Distribution:", shuffle_results)

# ---------------------------------------------
# 6. Distributed Test 3:
#    Accuracy, F1, Precision, Recall
# ---------------------------------------------
print("\nComputing evaluation metrics...")

evaluator_acc = MulticlassClassificationEvaluator(
    labelCol="binary_label", predictionCol="prediction", metricName="accuracy"
)
evaluator_f1 = MulticlassClassificationEvaluator(
    labelCol="binary_label", predictionCol="prediction", metricName="f1"
)
evaluator_prec = MulticlassClassificationEvaluator(
    labelCol="binary_label", predictionCol="prediction", metricName="weightedPrecision"
)
evaluator_rec = MulticlassClassificationEvaluator(
    labelCol="binary_label", predictionCol="prediction", metricName="weightedRecall"
)

metrics = {
    "accuracy": evaluator_acc.evaluate(pred_df),
    "f1": evaluator_f1.evaluate(pred_df),
    "precision": evaluator_prec.evaluate(pred_df),
    "recall": evaluator_rec.evaluate(pred_df)
}

print("[TEST 3] Distributed model performance:")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")

# ---------------------------------------------
# 7. Per-class accuracy (distributed)
# ---------------------------------------------
print("\nComputing per-class accuracy...")

total_per_class = pred_df.groupBy("binary_label").count()
correct_per_class = pred_df.where(col("binary_label") == col("prediction")).groupBy("binary_label").count()

per_class_acc = (
    correct_per_class.alias("c")
    .join(total_per_class.alias("t"), "binary_label")
    .select(
        col("binary_label"),
        (col("c.count") / col("t.count")).alias("accuracy")
    )
)

print(per_class_acc.collect())

# ---------------------------------------------
# 8. Model broadcast size test (cluster specific)
# ---------------------------------------------
print("\nRunning model broadcast test...")

model_bytes = model.toString().encode("utf-8")
broadcast_start = time.time()
bc = spark.sparkContext.broadcast(model_bytes)
_ = bc.value
broadcast_end = time.time()

print("[TEST 4] Model broadcast time:", round(broadcast_end - broadcast_start, 4), "seconds")
print("Broadcasted model size:", len(model_bytes), "bytes")

# ---------------------------------------------
# END
# ---------------------------------------------
print("\n===== CLUSTER EVALUATION COMPLETE =====\n")

spark.stop()

