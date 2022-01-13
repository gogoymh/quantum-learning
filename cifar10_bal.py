import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math

from intra_BN import resnet56 as net
from intra_BN import set_class, get_loss
from sampler import BalancedBatchSampler

train_dataset = datasets.CIFAR10(
                        "../data/CIFAR10",
                        train=True,
                        download=True,
                        transform=transforms.Compose([
                                transforms.RandomCrop(32, padding=4),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(
                                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
                                ),
                        )

train_loader = DataLoader(train_dataset,
                          sampler=BalancedBatchSampler(train_dataset),
                          batch_size=100)#, pin_memory=True)


test_loader = DataLoader(
                datasets.CIFAR10(
                        '../data/CIFAR10',
                        train=False,
                        download=True,
                        transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(
                                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
                                ),
                        ),
                batch_size=1, shuffle=False)#, pin_memory=True)


device = torch.device("cuda:0")
model = net().to(device)

optimizer = optim.Adam(model.parameters(), lr=0.01)
#optimizer = optim.SGD(model.parameters(), lr=0.1)
criterion = nn.CrossEntropyLoss()

best_acc = 0
for epoch in range(300):
    runnning_loss = 0
    model.train()
    for X, Y in train_loader:
        optimizer.zero_grad()
               
        loss = 0
        
        for i in range(10):
            x = X[Y==i]
            y = Y[Y==i]
            
            set_class(model, i)
            output = model(x.float().to(device))
            #print((torch.isnan(output)).sum())
            loss += criterion(output, y.long().to(device)) + get_loss(model)*0.0001
        
        loss /= 10
        loss.backward()
        optimizer.step()
        runnning_loss += loss.item()
        #print(loss.item())
        
    runnning_loss /= len(train_loader)
    print("[Epoch:%d] [Loss:%f]" % ((epoch+1), runnning_loss), end=" ")
    
    
    accuracy = 0
    with torch.no_grad():
        model.eval()
        #print("[Scale:%f]" % model.scale, end=" ")
        correct = 0
        for x, y in test_loader:
            output = model(x.float().to(device))
            pred = output.argmax(1, keepdim=True)
            correct += pred.eq(y.long().to(device).view_as(pred)).sum().item()
                
        accuracy = correct / len(test_loader.dataset)

        if accuracy >= best_acc:
            print("[Accuracy:%f] **Best**" % accuracy)#, end=" ")
            best_acc = accuracy
        else:
            print("[Accuracy:%f]" % accuracy)#, end=" ")
        
        #print(model.integer.item())
        
    #scheduler.step()
    
    #torch.save({'model_state_dict': model.state_dict()}, "/data/ymh/gpu_test/resnet56_preact.pth")