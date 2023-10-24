import torch  # isort: skip
import glob
import os
from argparse import ArgumentParser
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import tensorflow as tf
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from dataloader import get_ds
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
parser.add_argument("--batch-size", type=int, default=320)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--base-model", type=str, default="google/t5-v1_1-small")
parser.add_argument("--weight-decay", type=float, default=1e-5)
parser.add_argument("--learning-rate", type=float, default=1e-5)

FLAGS = parser.parse_args()
current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = os.path.join("logs", current_time)
if RANK == 0:
    Path("ckpts").mkdir(exist_ok=True, parents=True)
    Path("logs").mkdir(exist_ok=True, parents=True)
    writer = SummaryWriter(log_dir)

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
if RANK == 0:
    print(dtype, ptdtype, ctx)


net = MsNet(FLAGS.base_model).to("cuda")
net.train()

if WORLD_SIZE > 1:
    net = torch.nn.parallel.DistributedDataParallel(
        net, device_ids=[RANK], output_device=RANK
    )

net = torch.compile(net)
optimizer = torch.optim.Adam(
    net.parameters(),
    weight_decay=FLAGS.weight_decay,
    lr=FLAGS.learning_rate,
)
batch_size = FLAGS.batch_size
step = 0
ckpt_files = sorted(glob.glob("ckpts/step_*.pt"))
if ckpt_files:
    latest_ckpt = ckpt_files[-1]
    checkpoint = torch.load(latest_ckpt, map_location="cpu")
    net.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    step = checkpoint["step"]
    print(f"Resuming from step {checkpoint['step']}")


class MyAllGather(torch.autograd.Function):
    """
    Gather tensors from all workers with support for backward propagation:
    This implementation does not cut the gradients as torch.distributed.all_gather does.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(WORLD_SIZE)]
        torch.distributed.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        torch.distributed.all_reduce(all_gradients)
        return all_gradients[RANK]


def loss_fn(net, query, passage):
    query, passage = net(query, passage)

    if WORLD_SIZE > 1:
        all_passages = [torch.empty_like(passage) for _ in range(WORLD_SIZE)]
        all_passages = MyAllGather.apply(passage)
        all_passages = torch.concat(all_passages, dim=0)
    else:
        all_passages = passage

    logits = torch.einsum("AD,BD->AB", query, all_passages) * 100
    target = (
        torch.arange(0, logits.shape[0], device=logits.device) + RANK * query.shape[0]
    )
    loss = torch.nn.functional.cross_entropy(logits, target)
    return loss


eval_dataset = get_ds(
    batch_size,
    "validation",
    rank=RANK,
    world_size=WORLD_SIZE,
    seed=0,
    repeat=True,
    drop_remainder=True,
)

eval_iter = eval_dataset.as_numpy_iterator()

dataset = get_ds(
    batch_size,
    "train",
    rank=RANK,
    world_size=WORLD_SIZE,
    seed=step,
    repeat=True,
    drop_remainder=True,
)
train_iter = dataset.as_numpy_iterator()
if RANK == 0:
    train_iter = tqdm(train_iter)
net.train()

for batch in train_iter:
    step = step + 1
    query, passage = (
        torch.from_numpy(batch["query"]).cuda(),
        torch.from_numpy(batch["passage"]).cuda(),
    )
    with ctx:
        loss = loss_fn(net, query, passage)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    if RANK == 0:
        writer.add_scalar("Loss/train", loss.item(), step)

    if step % 10 == 0:
        net.eval()
        with ctx:
            with torch.no_grad():
                batch = next(eval_iter)
                query, passage = (
                    torch.from_numpy(batch["query"]).cuda(),
                    torch.from_numpy(batch["passage"]).cuda(),
                )
                loss = loss_fn(net, query, passage)
                if RANK == 0:
                    writer.add_scalar("Loss/validation", loss.item(), step)
        net.train()

    if step % 1000 == 0:
        if RANK == 0:
            torch.save(
                {
                    "model_state_dict": net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "step": step,
                },
                f"ckpts/step_{step:08d}.pt",
            )
print("DONE!")
