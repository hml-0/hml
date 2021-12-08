from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import parser
import time
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import os

# 0. set up distributed device
rank = int(os.environ["RANK"])
local_rank = int(os.environ["LOCAL_RANK"])
world_size = 2
torch.cuda.set_device(rank % torch.cuda.device_count())
dist.init_process_group(backend="nccl",rank=rank,world_size=world_size)
device = torch.device("cuda", local_rank)

print(f"[init] == local rank: {local_rank}, global rank: {rank} ==")

batchsize = 16




# 定义数据初始化
torch.cuda.synchronize()
start = time.time()
print("Start time: ",start)
import torchvision.transforms as transforms
image_size=(224,224) #
data_transforms=transforms.Compose([
    transforms.RandomHorizontalFlip(), #依概率（默认0.5）水平翻转
    transforms.Resize(image_size ),    #图像大小变换，统一处理成224*224
    transforms.ToTensor(),             #将图像转化成Tensor类型
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), #图像正则化

])

# 导入数据集

import torchvision.datasets as datasets
path = 'fruits-360-original-size/fruits-360-original-size/Training'
train_data=datasets.ImageFolder(root=path,transform=data_transforms)    # ImageForder函数只能导入文件夹，不能导入文件
# print(train_data.classes)  # 输出train_data里面的文件夹名称

#每个进程一个sampler
train_sampler = torch.utils.data.distributed.DistributedSampler(train_data,rank = rank)


test_data =  datasets.ImageFolder(root='fruits-360-original-size/fruits-360-original-size/Test',transform=data_transforms)
test_sampler = torch.utils.data.distributed.DistributedSampler(test_data,rank=rank)
# 用DataLoader函数处理数据集
from torch.utils.data import DataLoader

train_loader = DataLoader(train_data,batch_size=batchsize,shuffle=False,num_workers=1,sampler=train_sampler,pin_memory=True)
test_loader = DataLoader(test_data,batch_size=batchsize,shuffle=False,num_workers=1,sampler=test_sampler,pin_memory=True)
# 导入模型
import torchvision.models as models
AlexNet = models.AlexNet()
model = AlexNet

# 权重初始化
import torch.nn.init as init
for name,module in model._modules.items():
    if(name=='fc'):
        init.kaiming_uniform_(module.weight,a=0,mode='fan_in')

# 定义优化器
import torch
optimizer=torch.optim.SGD(model.parameters(),lr=0.01 )
StepLR=torch.optim.lr_scheduler.StepLR (optimizer ,step_size= 3,gamma=1 )  #每隔三次调整一下LR  等间隔的调整学习率，调整倍数为gamma倍，调整间隔为step_size

# 调用GPU     用GPU训练的模型，测试时只能用GPU测试

import torch.distributed

model.to(device)  # 调用GPU训练
model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[local_rank],output_device=local_rank)

# 训练模型

import torch.nn.functional as F


def get_num_correct(outs, label):
    pass

# model.to(device)  # 调用GPU训练

for epoch in range(30):
    total_loss = 0
    print("epoch", epoch, ":***************************")

    for batch in train_loader:
        images, labels = batch

        images = images.cuda(non_blocking=True) # 将图片传入GPU
        labels = labels.cuda(non_blocking=True)  # 将标签传入GPU

        outs = model(images)
        loss = F.cross_entropy(outs, labels)  # 正向传播
        optimizer.zero_grad() # 用过的梯度一定要清零，不然会占内存
        loss.backward()  # 反向传播
        optimizer.step()  # 参数优化
        total_loss += loss.item()
    print('loss', total_loss)

# # 保存模型
# torch.save(model,'FruitModelGPU.pth')

# test
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images,labels = data
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)

        outs = model(images)
        _,predicted = torch.max(outs.data,dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    print('Accuracy on test set: %d %%'%(100 * correct /total))

torch.cuda.synchronize()
end = time.time()
print("End time:",end)
print("Total time:",end-start)