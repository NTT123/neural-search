# first get rank and world size
import torch  # isort: skip
import os

#### INIT DISTRIBUTED TRAINING ####
if "RANK" in os.environ:
    torch.distributed.init_process_group(backend="nccl")
    RANK = int(os.environ.get("RANK", "0"))
    WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    torch.cuda.set_device(RANK)
else:
    RANK = 0
    WORLD_SIZE = 1

device = "cuda"

# define input for each rank
x = 10 + torch.ones((1,)) * RANK
x = x.data.to(device)


# comput loss
class MyNeT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # define parameters
        p = torch.zeros(1)
        p.requires_grad_(True)
        self.p = torch.nn.Parameter(p, requires_grad=True)

    def forward(self, x):
        y = (x * self.p + x + self.p) ** 0.5
        y = (y + x) ** 0.1
        return y


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


net = MyNeT().to(device)
net = torch.nn.parallel.DistributedDataParallel(
    net, device_ids=[RANK], output_device=RANK
)
y = net(x)

if WORLD_SIZE > 1:
    all_y = [torch.empty_like(y) for _ in range(WORLD_SIZE)]
    all_y = MyAllGather.apply(y)
    all_y = torch.concat(all_y, dim=0)
else:
    all_y = y


# print(all_y.shape)

loss = torch.mean(torch.square(all_y - RANK))
# print(loss)
loss.backward()
import time

time.sleep(RANK)
print("\n", "Grad", RANK, net.module.p.grad, end="\n", flush=True)
