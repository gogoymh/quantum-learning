import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch.nn as nn
import torch.optim as optim
import math

#from network11_3_conv import net
from res_trans import resnet56 as net
#from trans_res import resnet56 as net
from utils import WarmupLinearSchedule, SAM
#from quantile_loss import pseudo_ce_loss #q_ce_loss

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
                                        (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
                                ),
                        ),
                batch_size=128, shuffle=True, drop_last=True)#, pin_memory=True)


test_loader = DataLoader(
                datasets.CIFAR10(
                        '../data/CIFAR10',
                        train=False,
                        download=True,
                        transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(
                                        (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
                                ),
                        ),
                batch_size=64, shuffle=False)#, pin_memory=True)


device = torch.device("cuda:0")
model1 = net().to(device)
model2 = net().to(device)

#params = list(model.parameters()) + list(init.parameters())
#optimizer = optim.Adam(model.parameters(), lr=0.01)
#optimizer = optim.Adam(model.parameters(), lr=1e-3)#, weight_decay=0.1)
#optimizer = optim.SGD(model.parameters(), lr=3e-2, momentum=0.9, weight_decay=0.1)
criterion = nn.CrossEntropyLoss(reduction="sum")
#criterion = q_ce_loss
#criterion = pseudo_ce_loss

class softXEnt(nn.Module):
    def __init__(self):
        super(softXEnt, self).__init__()
        
        self.log_softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, input, target):               
        logprobs = self.log_softmax(input)
        return -(target * logprobs).sum(dim=1).sum(dim=0)
    
soft_ce = softXEnt()

#scheduler = WarmupLinearSchedule(optimizer, warmup_steps=15, t_total=300)

base_optimizer = torch.optim.SGD
#optimizer = SAM(model.parameters(), base_optimizer, lr=3e-2, momentum=0.9)
params = list(model1.parameters()) + list(model2.parameters())
optimizer = SAM(params, base_optimizer, lr=0.1, momentum=0.9)   
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 5, gamma = 0.85)

#alpha = torch.linspace(1,0,300)
#alpha = (torch.exp(torch.linspace(1,0,300)) - 1)/(math.e - 1)
#alpha = torch.cat((torch.linspace(1,0,10), torch.zeros(290)),dim=0)
alpha = []
for i in range(300//4):
    alpha.append(torch.linspace(1,0.25,4))
alpha = torch.stack(alpha, dim=0).reshape(-1)

best_acc = 0
for epoch in range(300):
    runnning_loss = 0
    model1.train()
    model2.train()
    for x, y in train_loader:
        optimizer.zero_grad()
        
        B = x.shape[0]
        L = int(B*alpha[epoch])
        U = B - L
        
        loss = 0
        if L > 0:
            xL = x[:L].float().to(device)
            yL = y[:L].long().to(device)
            
            output1L = model1(xL)
            output2L = model2(xL)
            
            loss += (criterion(output1L, yL) + criterion(output2L, yL))/4
            loss += (soft_ce(output1L, output2L.softmax(-1).detach()) + soft_ce(output2L, output1L.softmax(-1).detach()))/4
            
        if U > 0:
            xU = x[L:].float().to(device)
        
            output1U = model1(xU)
            output2U = model2(xU)
        
            loss += (soft_ce(output1U, output2U.softmax(-1).detach()) + soft_ce(output2U, output1U.softmax(-1).detach()))/2
        
        loss /= B
        loss.backward()
        optimizer.first_step(zero_grad=True)
        
        loss2 = 0
        if L > 0:
            output1L = model1(xL)
            output2L = model2(xL)
            
            loss2 += (criterion(output1L, yL) + criterion(output2L, yL))/2
            
        if U > 0:
            output1U = model1(xU)
            output2U = model2(xU)
        
            loss2 += (soft_ce(output1U, output2U.softmax(-1).detach()) + soft_ce(output2U, output1U.softmax(-1).detach()))/2
        
        loss2 /= B
        loss2.backward()
        optimizer.second_step(zero_grad=True)
        
        #optimizer.step()
        runnning_loss += loss.item()
        #print(loss.item(), halt.mean().item())
    scheduler.step()
    
    runnning_loss /= len(train_loader)
    print("[Epoch:%d] [Loss:%f]" % ((epoch+1), runnning_loss), end=" ")
    
    
    with torch.no_grad():
        model1.eval()
        correct1 = 0
        for x, y in test_loader:
            output = model1(x.float().to(device))
            pred = output.argmax(1, keepdim=True)
            correct1 += pred.eq(y.long().to(device).view_as(pred)).sum().item()
            
        model2.eval()
        correct2 = 0
        for x, y in test_loader:
            output = model2(x.float().to(device))
            pred = output.argmax(1, keepdim=True)
            correct2 += pred.eq(y.long().to(device).view_as(pred)).sum().item()
            
        accuracy1 = correct1 / len(test_loader.dataset)
        accuracy2 = correct2 / len(test_loader.dataset)
        
        accuracy = max(accuracy1, accuracy2)
        
        if accuracy >= best_acc:
            print("[Accuracy1:%f] [Accuracy2:%f] [%d:%d] **Best**" % (accuracy1, accuracy2, L, U))
            best_acc = accuracy
        else:
            print("[Accuracy1:%f] [Accuracy2:%f] [%d:%d]" % (accuracy1, accuracy2, L, U))

        
        
    #scheduler.step()
    
    #torch.save({'model_state_dict': model.state_dict()}, "/data/ymh/gpu_test/resnet56_preact.pth")