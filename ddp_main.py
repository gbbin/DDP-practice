import os
import time
import argparse
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.cuda.amp import GradScaler


class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Linear(7 * 7 * 32, num_classes)

    def forward(self, x):
        with torch.cuda.amp.autocast():  # utilize mixed precision training to accelerate training and inference
            out = self.layer1(x)
            out = self.layer2(out)
            out = out.reshape(out.size(0), -1)
            out = self.fc(out)
        return out


def prepare():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default="0,1")
    parser.add_argument(
        "-e",
        "--epochs",
        default=3,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        default=32,
        type=int,
        metavar="N",
        help="number of batchsize",
    )
    args = parser.parse_args()

    # The following environment variables are set to enable DDP
    os.environ["MASTER_ADDR"] = "localhost"  # IP address of the master machine
    os.environ["MASTER_PORT"] = "19198"  # port number of the master machine
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu  # specify the GPUs to use
    world_size = torch.cuda.device_count()
    os.environ["WORLD_SIZE"] = str(world_size)
    return args


def init_ddp(local_rank):
    # after this setup, tensors can be moved to GPU via `a = a.cuda()` rather than `a = a.to(local_rank)`
    torch.cuda.set_device(local_rank)
    os.environ["RANK"] = str(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")


def get_ddp_generator(seed=3407):
    local_rank = dist.get_rank()
    g = torch.Generator()
    g.manual_seed(seed + local_rank)
    return g


def train(model, train_dloader, criterion, optimizer, scaler):
    model.train()
    for images, labels in train_dloader:
        images = images.cuda()
        labels = labels.cuda()
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        scaler.scale(loss).backward()  ###
        scaler.step(optimizer)  ###
        scaler.update()  ###


def test(model, test_dloader):
    local_rank = dist.get_rank()
    model.eval()
    size = torch.tensor(0.0).cuda()
    correct = torch.tensor(0.0).cuda()
    for images, labels in test_dloader:
        images = images.cuda()
        labels = labels.cuda()
        with torch.no_grad():
            outputs = model(images)
            size += images.size(0)
        correct += (outputs.argmax(1) == labels).type(torch.float).sum()
    dist.reduce(size, 0, op=dist.ReduceOp.SUM)  ###
    dist.reduce(correct, 0, op=dist.ReduceOp.SUM)  ###
    if local_rank == 0:
        acc = correct / size
        print(f"Accuracy is {acc:.2%}")


def main(local_rank, args):
    init_ddp(local_rank)  ### init DDP
    model = (
        ConvNet().cuda()
    )  ### Note: the `forward` method of the model has been modified
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)  ### Convert BatchNorm layers
    model = nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank]
    )  ### Wrap with DDP
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), 1e-4)
    scaler = GradScaler()  ### Used for mixed precision training
    train_dataset = torchvision.datasets.MNIST(
        root="./data", train=True, transform=transforms.ToTensor(), download=True
    )
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset
    )  ### Sampler specifically for DDP
    g = get_ddp_generator()  ###
    train_dloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=False,  ### shuffle is mutually exclusive with sampler
        num_workers=4,
        pin_memory=True,
        sampler=train_sampler,
        generator=g,
    )  ### generator is used for random seed
    test_dataset = torchvision.datasets.MNIST(
        root="./data", train=False, transform=transforms.ToTensor(), download=True
    )
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_dataset
    )  ### Sampler specifically for DDP
    test_dloader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        sampler=test_sampler,
    )
    for epoch in range(args.epochs):
        if local_rank == 0:  ### avoid redundant printing for each process
            print(f"begin training of epoch {epoch + 1}/{args.epochs}")
        train_dloader.sampler.set_epoch(epoch)  ### set epoch for sampler
        train(model, train_dloader, criterion, optimizer, scaler)
    if local_rank == 0:
        print(f"begin testing")
    test(model, test_dloader)
    if local_rank == 0:  ### avoid redundant saving for each process
        torch.save(
            {"model": model.state_dict(), "scaler": scaler.state_dict()},
            "ddp_checkpoint.pt",
        )
    dist.destroy_process_group() ### destroy the process group, in accordance with init_process_group.


if __name__ == "__main__":
    args = prepare()
    time_start = time.time()
    mp.spawn(main, args=(args,), nprocs=torch.cuda.device_count())
    time_elapsed = time.time() - time_start
    print(f"\ntime elapsed: {time_elapsed:.2f} seconds")
