import tensorflow as tf

tf.config.set_visible_devices([], "GPU")


def get_ds(
    batch_size: int,
    split="train",
    repeat=True,
    rank=0,
    world_size=1,
    seed=0,
    drop_remainder=True,
):
    files = tf.data.Dataset.list_files(f"data/{split}/*.tfrecord", shuffle=False)
    L = len(files)
    if repeat:
        files = files.repeat().shuffle(L, seed=seed)
    else:
        files = files.shuffle(L, seed=seed)
    files = files.shard(world_size, rank)
    dataset = tf.data.TFRecordDataset(files, num_parallel_reads=4)
    feature_description = {
        "query": tf.io.FixedLenFeature([], tf.string),
        "passage": tf.io.FixedLenFeature([], tf.string),
        "passage_id": tf.io.FixedLenFeature([], tf.int64),
    }

    def _parse_function(example_proto):
        parsed_example = tf.io.parse_single_example(example_proto, feature_description)
        parsed_example["query"] = tf.reshape(
            tf.io.parse_tensor(parsed_example["query"], out_type=tf.int32), [-1]
        )
        parsed_example["passage"] = tf.reshape(
            tf.io.parse_tensor(parsed_example["passage"], out_type=tf.int32), [-1]
        )
        # del parsed_example["passage_id"]
        return parsed_example

    dataset = (
        dataset.map(_parse_function, num_parallel_calls=4, deterministic=True)
        .shuffle(100 * batch_size)
        .padded_batch(
            batch_size,
            padded_shapes={"query": [32], "passage": [256], "passage_id": []},
            drop_remainder=drop_remainder,
        )
        .prefetch(1)
    )
    return dataset


def get_passage_loader(batch_size: int, rank=0, world_size=1):
    import tensorflow as tf

    files = tf.data.Dataset.list_files(f"data/passage/*.tfrecord", shuffle=False)
    files = files.shard(world_size, rank)
    dataset = tf.data.TFRecordDataset(files, num_parallel_reads=4)
    feature_description = {
        "passage": tf.io.FixedLenFeature([], tf.string),
        "passage_id": tf.io.FixedLenFeature([], tf.int64),
    }

    def _parse_function(example_proto):
        parsed_example = tf.io.parse_single_example(example_proto, feature_description)
        parsed_example["passage"] = tf.reshape(
            tf.io.parse_tensor(parsed_example["passage"], out_type=tf.int32), [-1]
        )
        return parsed_example

    dataset = (
        dataset.map(_parse_function, num_parallel_calls=4)
        .padded_batch(
            batch_size,
            drop_remainder=False,
            padded_shapes={"passage": [256], "passage_id": []},
        )
        .prefetch(1)
    )
    return dataset
