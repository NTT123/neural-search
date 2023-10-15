from pathlib import Path

import tensorflow as tf
from datasets import load_dataset
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from transformers import T5Tokenizer

Path("./data").mkdir(exist_ok=True)

# Download the ms_marco dataset v2
dataset = load_dataset("ms_marco", "v2.1")
dataset["train"].to_parquet("data/train.parquet")
dataset["validation"].to_parquet("data/validation.parquet")
dataset["test"].to_parquet("data/test.parquet")


def collect_all_passages():
    spark = SparkSession.builder.master("local[*]").getOrCreate()
    df1 = spark.read.parquet(f"data/train.parquet").select(
        F.explode(F.col("passages").passage_text).alias("passage")
    )
    df2 = spark.read.parquet(f"data/validation.parquet").select(
        F.explode(F.col("passages").passage_text).alias("passage")
    )
    df3 = spark.read.parquet(f"data/test.parquet").select(
        F.explode(F.col("passages").passage_text).alias("passage")
    )
    df = df1.unionByName(df2).unionByName(df3)
    df = df.dropDuplicates()
    df = df.repartition(100)
    df = df.withColumn("passage_id", F.monotonically_increasing_id())
    df.write.mode("overwrite").parquet("data/passage.parquet")
    spark.stop()


collect_all_passages()


def prepare_passage_tfdata():
    def write_tfdata(index, rows):
        tokenizer = T5Tokenizer.from_pretrained("t5-small")
        writer = tf.io.TFRecordWriter(f"data/passage/part_{index:05d}.tfrecord")

        for row in rows:
            passage = "search result: " + row["passage"]
            passage_id = row["passage_id"]
            passage_tokens = tokenizer.encode(
                passage,
                return_tensors="np",
                add_special_tokens=False,
                max_length=256,
                truncation=True,
            )
            passage_tensor = tf.convert_to_tensor(
                passage_tokens.flatten(), dtype=tf.int32
            )

            feature = {
                "passage": tf.train.Feature(
                    bytes_list=tf.train.BytesList(
                        value=[tf.io.serialize_tensor(passage_tensor).numpy()]
                    )
                ),
                "passage_id": tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[passage_id])
                ),
            }

            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

        writer.close()
        return []

    spark = SparkSession.builder.master("local[*]").getOrCreate()
    df = spark.read.parquet("data/passage.parquet")
    df.rdd.mapPartitionsWithIndex(write_tfdata).collect()
    spark.stop()


def prepare_tfdata(split: str, num_parts: int):
    spark = SparkSession.builder.master("local[*]").getOrCreate()
    Path(f"data/{split}").mkdir(parents=True, exist_ok=True)
    df = spark.read.parquet(f"data/{split}.parquet")
    df = df.withColumn(
        "passages",
        F.arrays_zip(
            F.col("passages").is_selected.alias("is_selected"),
            F.col("passages").passage_text.alias("passage_text"),
        ),
    )
    df = df.withColumn("passages", F.filter("passages", lambda x: x.is_selected == 1))
    df = df.withColumn("passages", F.transform("passages", lambda x: x.passage_text))
    df = df.select("query", F.explode("passages").alias("passage"))
    passage_df = spark.read.parquet("data/passage.parquet")
    df = df.join(passage_df, how="inner", on="passage")

    # Load T5 small tokenizer
    tokenizer = T5Tokenizer.from_pretrained("t5-small")

    def write_tfrecord(index, rows):
        writer = tf.io.TFRecordWriter(f"data/{split}/part_{index:05d}.tfrecord")

        for row in rows:
            query = "user query: " + row["query"]
            passage = "search result: " + row["passage"]
            passage_id = row["passage_id"]
            query_tokens = tokenizer.encode(
                query,
                return_tensors="np",
                add_special_tokens=False,
                max_length=32,
                truncation=True,
            )
            passage_tokens = tokenizer.encode(
                passage,
                return_tensors="np",
                add_special_tokens=False,
                max_length=256,
                truncation=True,
            )

            query_tensor = tf.convert_to_tensor(query_tokens.flatten(), dtype=tf.int32)
            passage_tensor = tf.convert_to_tensor(
                passage_tokens.flatten(), dtype=tf.int32
            )

            feature = {
                "query": tf.train.Feature(
                    bytes_list=tf.train.BytesList(
                        value=[tf.io.serialize_tensor(query_tensor).numpy()]
                    )
                ),
                "passage": tf.train.Feature(
                    bytes_list=tf.train.BytesList(
                        value=[tf.io.serialize_tensor(passage_tensor).numpy()]
                    )
                ),
                "passage_id": tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[passage_id])
                ),
            }

            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

        writer.close()
        return []

    df.repartition(num_parts).rdd.mapPartitionsWithIndex(write_tfrecord).collect()
    spark.stop()


Path("data/passage").mkdir(exist_ok=True, parents=True)
prepare_passage_tfdata()
prepare_tfdata("train", 100)
prepare_tfdata("validation", 10)
