import torch  # isort: skip
from argparse import ArgumentParser

import tensorflow as tf
from tqdm.auto import tqdm

from dataloader import get_ds
from model import MsNet

parser = ArgumentParser()
parser.add_argument("--batch-size", type=int, default=256)
parser.add_argument("--base-model", type=str, default="google/t5-v1_1-small")
parser.add_argument("--ckpt", type=str, required=True)
parser.add_argument("--device", type=str, default="cuda")
FLAGS = parser.parse_args()
device = FLAGS.device
net = MsNet(FLAGS.base_model)
cnet = torch.nn.parallel.DataParallel(net, device_ids=[0], output_device=0)
cnet = torch.compile(cnet)
cnet.load_state_dict(torch.load(FLAGS.ckpt, map_location="cpu")["model_state_dict"])
net = net.to(device)

tf.config.set_visible_devices([], "GPU")
loader = get_ds(FLAGS.batch_size, "validation", repeat=False, drop_remainder=False, keep_passage_id=True)

with tf.io.TFRecordWriter("query_embed.tfrecord") as writer:
    for batch in tqdm(loader.as_numpy_iterator(), desc="compute embed"):
        with torch.no_grad():
            query = torch.from_numpy(batch["query"]).to(device)
            query_embed = net.get_embed(query)
            query_embed = query_embed.cpu().numpy()
        query_embed_tensor = tf.convert_to_tensor(query_embed, dtype=tf.float32)
        passage_id_tensor = tf.convert_to_tensor(batch["passage_id"], dtype=tf.int64)

        features = {
            "query_embed": tf.train.Feature(
                bytes_list=tf.train.BytesList(
                    value=[tf.io.serialize_tensor(query_embed_tensor).numpy()]
                )
            ),
            "passage_id": tf.train.Feature(
                bytes_list=tf.train.BytesList(
                    value=[tf.io.serialize_tensor(passage_id_tensor).numpy()]
                )
            ),
        }

        # Create a Features message using tf.train.Example.
        example_proto = tf.train.Example(features=tf.train.Features(feature=features))
        # Write the serialized example.
        writer.write(example_proto.SerializeToString())
