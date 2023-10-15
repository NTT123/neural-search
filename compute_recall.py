import faiss
import numpy as np
import tensorflow as tf
from tqdm.auto import tqdm

tf.config.set_visible_devices([], "GPU")


# Create a tf.data.Dataset from the TFRecord files
files = tf.data.Dataset.list_files(f"passage_embed_*", shuffle=False)
raw_dataset = tf.data.TFRecordDataset(files)

# Create a dictionary describing the features
feature_description = {
    "passage_embed": tf.io.FixedLenFeature([], tf.string),
    "passage_id": tf.io.FixedLenFeature([], tf.string),
}


# Define a function to parse the input tf.Example proto
def _parse_function(example_proto):
    # Parse the input tf.Example proto using the dictionary above
    parsed_record = tf.io.parse_single_example(example_proto, feature_description)
    passage_embed = tf.io.parse_tensor(
        parsed_record["passage_embed"], out_type=tf.float32
    )
    passage_id = tf.io.parse_tensor(parsed_record["passage_id"], out_type=tf.int64)
    return passage_embed, passage_id


# Use map to apply this function to each item in the dataset
parsed_dataset = raw_dataset.map(_parse_function)


# Decode the passage_embed and passage_id
a, b = [], []
for passage_embed, passage_id in tqdm(parsed_dataset.as_numpy_iterator()):
    a.append(passage_embed)
    b.append(passage_id)
passage_embed = np.concatenate(a, axis=0)
passage_id = np.concatenate(b, axis=0)


d = passage_embed.shape[-1]  # dimension
index = faiss.IndexFlatL2(d)  # build the index
res = faiss.StandardGpuResources()
cfg = faiss.GpuIndexFlatConfig()
cfg.useFloat16 = False
cfg.device = 0
index = faiss.IndexIDMap(faiss.GpuIndexFlatL2(res, d, cfg))  # use float16 index


index.add_with_ids(passage_embed.astype(np.float32), passage_id)


# load tfrecord
files = tf.data.Dataset.list_files("query_embed.tfrecord", shuffle=False)
feature_description = {
    "query_embed": tf.io.FixedLenFeature([], tf.string),
    "passage_id": tf.io.FixedLenFeature([], tf.string),
}


def _parse_function(example_proto):
    parsed_example = tf.io.parse_single_example(example_proto, feature_description)
    query_embed = tf.io.parse_tensor(parsed_example["query_embed"], out_type=tf.float32)
    passage_id = tf.io.parse_tensor(parsed_example["passage_id"], out_type=tf.int64)
    return query_embed, passage_id


query_loader = tf.data.TFRecordDataset("query_embed.tfrecord").map(_parse_function)


count = 0
total = 0
N = 50
for query_embed, passage_id in tqdm(query_loader.as_numpy_iterator()):
    query_embed = query_embed.astype(np.float32)
    total = total + query_embed.shape[0]
    D, I = index.search(query_embed, N)
    count = count + (I == passage_id[:, None]).sum()

print(f"Recall@{N}: {count / total:.3%}")
