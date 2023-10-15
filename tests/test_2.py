# first get rank and world size
import torch  # isort: skip


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


device = "cuda"
net = MyNeT().to(device)
all_y = []
for RANK in [0, 1]:
    # define input for each rank
    x = 10 + torch.ones((1,)) * RANK
    x = x.data.to(device)

    y = net(x)
    all_y.append(y)

all_y = torch.concat(all_y, dim=0)
losses = []
for RANK in [0, 1]:
    # print(all_y.shape)
    loss = torch.mean(torch.square(all_y - RANK))
    losses.append(loss)
loss = (losses[0] + losses[1]) / 2
loss.backward()
print(net.p.grad, end="\n", flush=True)
