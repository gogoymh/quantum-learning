import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch.nn as nn
import torch.optim as optim
import os

#from network11_3_conv import net
from res_trans import resnet56 as net2
#from trans_res import resnet56 as net
from utils import WarmupLinearSchedule, SAM
from quantile_loss import pseudo_ce_loss #q_ce_loss
from pseudo_cifar10 import pseudo_cifar10

from org_resnet import resnet56 as net

train_loader = DataLoader(
                datasets.CIFAR10(
                        '../data/CIFAR10',
                        train=True,
                        download=True,
                        transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(
                                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
                                ),
                        ),
                batch_size=128, shuffle=False)#, pin_memory=True)

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
                batch_size=128, shuffle=False)#, pin_memory=True)

testfortrain_loader = DataLoader(
                datasets.CIFAR10(
                        '../data/CIFAR10',
                        train=False,
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

new_trainset = pseudo_cifar10(transform=transforms.Compose([
                                transforms.RandomCrop(32, padding=4),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(
                                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
                                ))

device = torch.device("cuda:0")
model = net().to(device)

#params = list(model.parameters()) + list(init.parameters())
optimizer = optim.Adam(model.parameters(), lr=0.01)
#optimizer = optim.Adam(model.parameters(), lr=1e-3)#, weight_decay=0.1)
#optimizer = optim.SGD(model.parameters(), lr=3e-2, momentum=0.9, weight_decay=0.1)
criterion = nn.CrossEntropyLoss()
#criterion = q_ce_loss
#criterion = pseudo_ce_loss


#base_optimizer = torch.optim.SGD
#optimizer = SAM(model.parameters(), base_optimizer, lr=0.1, momentum=0.9)   
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 5, gamma = 0.85)

save_path = "/home/DATA/ymh/onepixel/"
'''
best_acc = 0
epoch = -1
while best_acc <= 0.99:
    epoch += 1
    runnning_loss = 0
    model.train()
    for x, y in testfortrain_loader:
        optimizer.zero_grad()
              
        output = model(x.float().to(device))
        #print((torch.isnan(output)).sum())
        loss = criterion(output, y.long().to(device))
        loss.backward()
        #optimizer.first_step(zero_grad=True)
        
        #criterion(model(x.float().to(device)), y.long().to(device)).backward()
        #optimizer.second_step(zero_grad=True)
        
        optimizer.step()
        runnning_loss += loss.item()
        #print(loss.item())
    #scheduler.step()
    
    runnning_loss /= len(testfortrain_loader)
    print("[Epoch:%d] [Loss:%f]" % ((epoch+1), runnning_loss), end=" ")
    
    
    accuracy = 0
    with torch.no_grad():
        model.eval()
        correct = 0
        for x, y in test_loader:
            output = model(x.float().to(device))
            pred = output.argmax(1, keepdim=True)
            correct += pred.eq(y.long().to(device).view_as(pred)).sum().item()  
            
        accuracy = correct / len(test_loader.dataset)
        
        if accuracy >= best_acc:
            print("[Accuracy:%f] **Best**" % (accuracy))
            best_acc = accuracy
        else:
            print("[Accuracy:%f]" % (accuracy))

torch.save({'model_state_dict': model.state_dict()}, os.path.join(save_path, "test_trained.pth"))
'''
'''
model_name = os.path.join(save_path, "test_trained.pth")
checkpoint = torch.load(model_name)
model.load_state_dict(checkpoint["model_state_dict"])

cnt = 0
model.eval()
for x, _ in train_loader:
    output = model(x.float().to(device))
    pred = output.argmax(1, keepdim=False)
    
    for i in range(x.shape[0]):
        index = cnt + i
        new_trainset.pseudo_label[index] = pred[i].item()
        
    cnt += x.shape[0]
    
    print("[%d/50000]" % cnt)

del model

torch.save(new_trainset.pseudo_label, os.path.join(save_path,'tensor.pt'))
'''
new_trainset.pseudo_label = torch.load(os.path.join(save_path,'tensor.pt'))

new_trainloader = DataLoader(new_trainset, batch_size=128, shuffle=True)
new_model = net().to(device)
new_optimizer = optim.Adam(new_model.parameters(), lr=0.01)

#base_optimizer = torch.optim.SGD
#new_optimizer = SAM(new_model.parameters(), base_optimizer, lr=0.1, momentum=0.9)   
#new_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 5, gamma = 0.85)

best_acc = 0
for epoch in range(300):
    runnning_loss = 0
    new_model.train()
    for x, _, y in new_trainloader:
        new_optimizer.zero_grad()
              
        output = new_model(x.float().to(device))
        #print((torch.isnan(output)).sum())
        loss = criterion(output, y.long().to(device))
        loss.backward()
        #new_optimizer.first_step(zero_grad=True)
        
        #criterion(new_model(x.float().to(device)), y.long().to(device)).backward()
        #new_optimizer.second_step(zero_grad=True)
        
        new_optimizer.step()
        runnning_loss += loss.item()
        #print(loss.item())
    #new_scheduler.step()
    
    runnning_loss /= len(new_trainloader)
    print("[Epoch:%d] [Loss:%f]" % ((epoch+1), runnning_loss), end=" ")
    
    
    accuracy = 0
    with torch.no_grad():
        new_model.eval()
        correct = 0
        for x, y in test_loader:
            output = new_model(x.float().to(device))
            pred = output.argmax(1, keepdim=True)
            correct += pred.eq(y.long().to(device).view_as(pred)).sum().item()  
            
        accuracy = correct / len(test_loader.dataset)
        
        if accuracy >= best_acc:
            print("[Accuracy:%f] **Best**" % (accuracy))
            best_acc = accuracy
        else:
            print("[Accuracy:%f]" % (accuracy))
