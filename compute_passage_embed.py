import torch  # isort: skip
import os
from argparse import ArgumentParser
from contextlib import nullcontext

import tensorflow as tf
from tqdm.auto import tqdm

from dataloader import get_passage_loader
from model import MsNet

#### INIT DISTRIBUTED TRAINING ####
if "RANK" in os.environ:
    torch.distributed.init_process_group(backend="nccl")
    RANK = int(os.environ.get("RANK", "0"))
    WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    torch.cuda.set_device(RANK)
else:
    RANK = 0
    WORLD_SIZE = 1

parser = ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--embed-dim", type=int, default=256)
parser.add_argument("--batch-size", type=int, default=256)
parser.add_argument("--base-model", type=str, default="google/t5-v1_1-small")
parser.add_argument("--ckpt", type=str, required=True)
parser.add_argument("--device", type=str, default="cuda")
FLAGS = parser.parse_args()

tf.config.set_visible_devices([], "GPU")
torch.backends.cudnn.benchmark = True
torch.manual_seed(FLAGS.seed)
torch.cuda.manual_seed(FLAGS.seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device = "cuda"
dtype = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else "float16"
)
compile = True
device_type = "cuda" if "cuda" in device else "cpu"
ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[dtype]
ctx = (
    nullcontext()
    if device_type == "cpu"
    else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
)
print(dtype, ptdtype, ctx)


device = FLAGS.device
net = MsNet(FLAGS.base_model, FLAGS.embed_dim).to(device)
net.eval()
net_bak = net

if WORLD_SIZE > 1:
    net = torch.nn.parallel.DistributedDataParallel(
        net, device_ids=[RANK], output_device=RANK
    )

net = torch.compile(net)

net.load_state_dict(torch.load(FLAGS.ckpt, map_location="cpu")["model_state_dict"])


loader = get_passage_loader(FLAGS.batch_size, rank=RANK, world_size=WORLD_SIZE)

with ctx:
    with torch.no_grad():
        with tf.io.TFRecordWriter(f"passage_embed_{RANK}.tfrecord") as writer:
            if RANK == 0:
                r = tqdm(loader.as_numpy_iterator(), desc="compute embed")
            else:
                r = loader.as_numpy_iterator()
            for batch in r:
                passage = torch.from_numpy(batch["passage"]).to(device)
                if passage.shape[0] == FLAGS.batch_size:
                    _, passage_embed = net(None, passage)
                else:
                    _, passage_embed = net_bak(None, passage)
                passage_embed = passage_embed.cpu().numpy()
                passage_embed_tensor = tf.convert_to_tensor(
                    passage_embed, dtype=tf.float32
                )
                passage_id_tensor = tf.convert_to_tensor(
                    batch["passage_id"], dtype=tf.int64
                )

                features = {
                    "passage_embed": tf.train.Feature(
                        bytes_list=tf.train.BytesList(
                            value=[tf.io.serialize_tensor(passage_embed_tensor).numpy()]
                        )
                    ),
                    "passage_id": tf.train.Feature(
                        bytes_list=tf.train.BytesList(
                            value=[tf.io.serialize_tensor(passage_id_tensor).numpy()]
                        )
                    ),
                }

                # Create a Features message using tf.train.Example.
                example_proto = tf.train.Example(
                    features=tf.train.Features(feature=features)
                )
                # Write the serialized example.
                writer.write(example_proto.SerializeToString())

import time

time.sleep(10)
