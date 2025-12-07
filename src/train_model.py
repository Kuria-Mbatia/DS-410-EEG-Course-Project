from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline

spark = SparkSession.builder.appName("EEG Spectrogram Classification").getOrCreate()

# 1. Load the dataset
df = spark.read.parquet("data/parquet/train.pqt")

# Drop empty/null columns
feature_cols = [f"f{i}" for i in range(400)]
df = df.na.drop(subset=feature_cols)

# 2. Index the string labels
label_indexer = StringIndexer(inputCol="target", outputCol="label")

# 3. Assemble the feature vector
feature_cols = [f"f{i}" for i in range(400)]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

# 4. Choose a classifier
rf = RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=50)

# 5. Build a pipeline
pipeline = Pipeline(stages=[label_indexer, assembler, rf])

# 6. Train/test split
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

# 7. Train
model = pipeline.fit(train_df)

# 8. Evaluate
preds = model.transform(test_df)
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
acc = evaluator.evaluate(preds)

print("Test Accuracy:", acc)

spark.stop()
