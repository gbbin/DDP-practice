# A Demo of training with DDP

\[ English | [中文](README_zh.md) \]

This blog presents a demonstration of training utilizing DistributedDataParallel (DDP) and Automatic Mixed Precision (AMP) in PyTorch.

Our emphasis is on the implementation aspects rather than the fundamental mechanisms.

## TOC
- [A Demo of training with DDP](#a-demo-of-training-with-ddp)
  - [TOC](#toc)
  - [Baseline](#baseline)
    - [Entry](#entry)
    - [Initilization](#initilization)
    - [main](#main)
    - [Model](#model)
    - [Train](#train)
    - [Test](#test)
  - [DDP](#ddp)
    - [Entry](#entry-1)
    - [Initialization](#initialization)
    - [main](#main-1)
      - [DDP init](#ddp-init)
      - [model](#model-1)
      - [scaler](#scaler)
    - [Train](#train-1)
    - [Test](#test-1)
  - [Initiate with torchrun](#initiate-with-torchrun)
  - [Checklist](#checklist)
  - [PS](#ps)

The DistributedDataParallel implementation referenced in this blog consults various blogs and documentation, drawing significant insights from [this particular tutorial](https://github.com/KaiiZhang/DDP-Tutorial/blob/main/DDP-Tutorial.md). Building upon this foundation, we endeavor to provide a demonstration that is both accurate and aligned with the conventions of deep learning research literature.

The codebase is tailored specifically for training scenarios involving a single node with multiple GPUs. We employ the NCCL backend for communication and initialize the environment accordingly. Code segments marked with `###` will be the primary focus of our discussion.

## Baseline

The baseline code for training without DistributedDataParallel (DDP) and Automatic Mixed Precision (AMP) is available [here](origin_main.py).

### Entry

Consider the entry point of the code: we execute the `main` function and measure the duration of the process:

```python
if __name__ == '__main__':
    args = prepare()  ###
    time_start = time.time()
    main(args)
    time_elapsed = time.time() - time_start
    print(f'\ntime elapsed: {time_elapsed:.2f} seconds')
```

### Initilization

The `preparer` function is utilized to retrieve command-line arguments:

```python
def prepare():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default="0")
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
    return args
```

### main
Within the `main` function, we commence by acquiring training-related arguments, followed by configuring the model, loss function, optimizer, and dataset. Subsequently, we advance to the phases of training, evaluation, and persisting the model's `state_dict`.

```python
def main(args):
    model = ConvNet().cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), 1e-4)
    train_dataset = torchvision.datasets.MNIST(
        root="./data", train=True, transform=transforms.ToTensor(), download=True
    )
    train_dloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    test_dataset = torchvision.datasets.MNIST(
        root="./data", train=False, transform=transforms.ToTensor(), download=True
    )
    test_dloader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    for epoch in range(args.epochs):
        print(f"begin training of epoch {epoch + 1}/{args.epochs}")
        train(model, train_dloader, criterion, optimizer)
    print(f"begin testing")
    test(model, test_dloader)
    torch.save({"model": model.state_dict()}, "origin_checkpoint.pt")
```

### Model

The model used here is a simple CNN:

```python
import torch.nn as nn

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
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out
```

### Train

The `train` function is:

```python
def train(model, train_dloader, criterion, optimizer):
    model.train()
    for images, labels in train_dloader:
        images = images.cuda()
        labels = labels.cuda()
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### Test

The `test` function is:

```python
def test(model, test_dloader):
    model.eval()
    size = torch.tensor(0.).cuda()
    correct = torch.tensor(0.).cuda()
    for images, labels in test_dloader:
        images = images.cuda()
        labels = labels.cuda()
        with torch.no_grad():
            outputs = model(images)
            size += images.size(0)
        correct += (outputs.argmax(1) == labels).type(torch.float).sum()
    acc = correct / size
    print(f'Accuracy is {acc:.2%}')
```

Finally, we execute the Python script as follows:

```bash
python origin_main.py --gpu 0
```

Outputs

```bash
begin training of epoch 1/3
begin training of epoch 2/3
begin training of epoch 3/3
begin testing
Accuracy is 91.55%

time elapsed: 22.72 seconds
```

## DDP

Following the presentation of the baseline, we adapt the code to incorporate DDP. The modified code can be accessed [here](ddp_main.py).

### Entry

We setup DDP within the scope of `if __name__ == '__main__'`:

```python
import torch.multiprocessing as mp

if __name__ == '__main__':
    args = prepare()  ###
    time_start = time.time()
    mp.spawn(main, args=(args, ), nprocs=torch.cuda.device_count())
    time_elapsed = time.time() - time_start
    print(f'\ntime elapsed: {time_elapsed:.2f} seconds')
```

The arguments for the `spawn` function are detailed as follows:

1. `fn`: This is the `main` function referenced earlier. Each spawned process will execute this function once.
2. `args`: These are the arguments for `fn`. They must be provided in the form of a tuple, even when there is only a single element.
3. `nprocs`: This specifies the number of processes to initiate. It should be set to the same value as `world_size`, with the default being 1. Should this number differ from `world_size`, the processes will halt and await synchronization.

### Initialization

Within the `prepare` function, several specifications pertinent to DDP are established:

```python
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
```

### main
In the `main` function, an argument `local_rank` is introduced.

```python
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
```

#### DDP init
We begin by initializing the model using the `init_ddp` function, employing the `nccl` backend and `env` method:

```python
def init_ddp(local_rank):
    # after this setup, tensors can be moved to GPU via `a = a.cuda()` rather than `a = a.to(local_rank)`
    torch.cuda.set_device(local_rank)
    os.environ["RANK"] = str(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
```
Following initialization, we can readily acquire `local_rank` and `world_size` without the need to pass them as additional arguments through each function.

```python
import torch.distributed as dist
local_rank = dist.get_rank()
world_size = dist.get_world_size()
```

For instance, operations such as `print`, `log`, or `save_state_dict`, can be executed on a single process since all processes maintain an identical version. This can be exemplified as follows:

```python
if local_rank == 0:
    print(f'begin testing')
if local_rank == 0:
    torch.save({'model': model.state_dict(), 'scaler': scaler.state_dict()}, 'ddp_checkpoint.pt')
```

#### model
To acclerate inference, we integrate `torch.cuda.amp.autocast()` within the model's `forward` method as:

```python
def forward(self, x):
    with torch.cuda.amp.autocast():  # utilize mixed precision training to accelerate training and inference
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
    return out
```

While `autocast` may be utilized outside the `forward` function, employing it within this method is the most convenient and universally applicable approach.

After that, we need to convert the model using `convert_sync_batchnorm` and `DistributedDataParallel`.

#### scaler
We instantiate a `GradScaler` to dynamically scale the loss during training:

```python
from torch.cuda.amp import GradScaler
scaler = GradScaler()
```

### Train
When employing DDP, it is necessary to use `torch.utils.data.distributed.DistributedSampler` and provide a `generator` when `num_workers > 1`. Failing to do so will result in identical augmentations across all processes for each worker, thereby reducing randomness. A detailed analysis is available [here](https://zhuanlan.zhihu.com/p/618639620).


```python
def get_ddp_generator(seed=3407):
    local_rank = dist.get_rank()
    g = torch.Generator()
    g.manual_seed(seed + local_rank)
    return g

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
```

In the case of multiple epochs, it is necessary to configure the data loader's sampler for each epoch using `train_dloader.sampler.set_epoch(epoch)`.

Next, let's take a look at the `train` function:

```python
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
```

The final three lines of the preceding code segment have been modified. In contrast to the conventional `loss.backward` and `optimizer.step()`, we employ a `scaler` to scale the loss, mitigating the potential for underflow during Automatic Mixed Precision (AMP) training, and we update the `scaler`'s state accordingly. If multiple losses are computed, they should utilize a shared `scaler`. Additionally, when saving the `state_dict` of the model for subsequent training phases, which is a typical practice in the pretrain-finetune paradigm, it is advisable to also preserve the state of the `scaler`. This ensures continuity when loading the model parameters for finetuning.

### Test
During testing, it is necessary to `reduce` data from all processes to a single process. It is important to note that the `test` function should be executed within the scope of `if local_rank == 0` to avoid synchronization issues that could result in a deadlock among the processes.

```python
def test(model, test_dloader):
    local_rank = dist.get_rank()
    model.eval()
    size = torch.tensor(0.).cuda()
    correct = torch.tensor(0.).cuda()
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
        print(f'Accuracy is {acc:.2%}')
```

The two lines concluding with `###` signify the required `reduce` operations.

These additions constitute the entirety of the modifications made to the baseline code.

The method for executing the Python script remains similar:

```bash
python ddp_main.py --gpu 0,1
```

Results:

```bash
begin training of epoch 1/3
begin training of epoch 2/3
begin training of epoch 3/3
begin testing
Accuracy is 89.21%

time elapsed: 30.82 seconds
```

## Initiate with torchrun
In the demonstration provided, we initiate DistributedDataParallel (DDP) using `mp.spawn`. The `mp` module is a wrapper for the `multiprocessing` module and is not specifically optimized for DDP. An alternative approach is to use `torchrun`, which is the recommended method according to the official documentation. The corresponding code is accessible [here](ddp_main_torchrun.py).

Contrasting with the initiation via `mp.spawn`, `torchrun` simplifies the process by automatically managing certain environment variables. The only requirement is to set `os.environ['CUDA_VISIBLE_DEVICES']` to the desired GPUs (by default, it includes all available GPUs). Manual configuration such as `os.environ['MASTER_ADDR']` is no longer necessary. Moreover, the `local_rank` argument is omitted from the `main` function. The entry point of the program is as follows:

```python
if __name__ == '__main__':
    args = prepare()
    time_start = time.time()
    main(args)
    time_elapsed = time.time() - time_start
    local_rank = int(os.environ['LOCAL_RANK'])
    if local_rank == 0:
        print(f'\ntime elapsed: {time_elapsed:.2f} seconds')
```

The command to execute the Python script transitions from using `python` to `torchrun`, exemplified as follows:

```bash
torchrun --standalone --nproc_per_node=2 ddp_main_torchrun.py --gpu 0,1
```

The `nproc_per_node` parameter specifies the number of processes to be created. It should be set to match the number of GPUs utilized.

## Checklist
After completing the implementation of DistributedDataParallel (DDP), it is prudent to conduct a thorough inspection for potential bugs.

Here is a general checklist to guide the review process:

1. Verify the location and completeness of DDP initialization, which includes code within `if __name__ == '__main__'` and the `main` function. Ensure the process group is destroyed when exiting.
2. Confirm the model encapsulation, which should cover `autocast`, `convert_sync_batchnorm`, and the integration of DDP.
3. Check the configuration of `sampler`, `generator`, `shuffle`, and `sampler.set_epoch`, all of which are tailored for DDP usage.
4. Review the scaling of the loss during training to ensure it is managed correctly.
5. Ascertain that operations such as `print`, `log`, and `save` are performed by only one process to prevent redundancy.
6. Ensure proper execution of the `reduce` operation during testing.

## PS
Running multiple processes is akin to multiplying the `batch_size` by the number of processes. Consequently, the `batch_size` and `learning_rate` may require adjustment. In our implementation, these hyperparameters were not modified, resulting in minor discrepancies in the accuracy observed before and after the integration of DDP.

For smaller models, the difference in training speed is relatively marginal. However, as the model size increases, the adoption of DDP and AMP results in a significant acceleration of training speed and a reduction in GPU memory usage.