import torch.nn as nn
import torch.optim as optim
import copy

class softXEnt(nn.Module):
        def __init__(self):
            super(softXEnt, self).__init__()
        
            self.log_softmax = nn.LogSoftmax(dim=1)
        
        def forward(self, input, target):               
            logprobs = self.log_softmax(input)
            return -(target * logprobs).sum(dim=1).sum(dim=0)
    
soft_ce = softXEnt()
criterion = nn.CrossEntropyLoss(reduction="sum")

def Update(idx, model1, model2, device, label_loader, unlabeled_loader):#, weights, losses):
    model1 = model1.to(device)
    model2 = model2.to(device)
    
    param = list(model1.parameters()) + list(model2.parameters())
    optimizer = optim.Adam(param, lr=0.0005)
    
    part_loss = []
    runnning_loss = 0
    model1.train()
    model2.train()
    for epoch in range(5):
        for batch in range(10):
            loss = 0
            optimizer.zero_grad()
        
            x, y = label_loader.__iter__().next()
            xL = x.float().to(device)
            yL = y.long().to(device)
            L = xL.shape[0]
            
            output1L = model1(xL)
            output2L = model2(xL)
                
            loss += (criterion(output1L, yL) + criterion(output2L, yL))/4
            loss += (soft_ce(output1L, output2L.softmax(-1).detach()) + soft_ce(output2L, output1L.softmax(-1).detach()))/4
            
            x, _ = unlabeled_loader.__iter__().next()
            xU = x.float().to(device)
            U = xU.shape[0]
        
            output1U = model1(xU)
            output2U = model2(xU)
        
            loss += (soft_ce(output1U, output2U.softmax(-1).detach()) + soft_ce(output2U, output1U.softmax(-1).detach()))/2
        
            loss /= (L+U)
            loss.backward()
            optimizer.step()

            runnning_loss += loss.item()
    
        runnning_loss /= 10
        part_loss.append(runnning_loss)
        
    #model1.to("cpu")
    #weights.append()
    #losses.append()
    print("%d is Done. Loss is %f." % (idx, sum(part_loss) / len(part_loss)))
        
    return copy.deepcopy(model1.state_dict()), copy.deepcopy(sum(part_loss) / len(part_loss))
