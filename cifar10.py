import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math

#from quantum_net import net3 as net
#from preact_resnet import resnet56 as net
from org_resnet import resnet56 as net
#from org_resnet import W_loss
#from glom_resnet import ResNet56 as net
#from quantile_resnet import resnet56 as net
#from attentive_norm import resnet56 as net
#from norm_act import resnet56 as net
#from conv_chat import resnet56 as net
#from preact_resnet4 import resnet56 as net

#from res_trans import resnet56 as net

train_loader = DataLoader(
                datasets.CIFAR10(
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
                        ),
                batch_size=128, shuffle=True)#, pin_memory=True)


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
                batch_size=64, shuffle=False)#, pin_memory=True)


device = torch.device("cuda:0")
model = net().to(device)

optimizer = optim.Adam(model.parameters(), lr=0.01)
#optimizer = optim.SGD(model.parameters(), lr=0.1)
criterion = nn.CrossEntropyLoss()

best_acc = 0
for epoch in range(300):
    runnning_loss = 0
    model.train()
    for x, y in train_loader:        
        optimizer.zero_grad()
               
        output = model(x.float().to(device))
        #print((torch.isnan(output)).sum())
        loss = criterion(output, y.long().to(device))
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