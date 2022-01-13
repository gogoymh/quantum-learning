import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn

'''
q = torch.tensor([0.1, 0.75])

def q_ce_loss(logits, labels):
    ce_loss = F.cross_entropy(logits, labels, reduction='none')
    
    value = torch.quantile(ce_loss, q.to(ce_loss.device), dim=0, keepdim=False)
    
    chosen = (ce_loss >= value[0]).long() + (ce_loss < value[1]).long() - 1

    q_ce_loss = ce_loss[chosen.bool()]
    
    return q_ce_loss.mean()
'''

def onehot(labels: torch.Tensor, label_num):
    return torch.zeros(labels.shape[0], label_num, device=labels.device).scatter_(1, labels.view(-1, 1), 1)

class softXEnt(nn.Module):
    def __init__(self):
        super(softXEnt, self).__init__()
        
        self.log_softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, input, target):               
        logprobs = self.log_softmax(input)
        return -(target * logprobs).sum(dim=1)

q = torch.Tensor([0.8])
num = 4
soft_ce = softXEnt()
def pseudo_ce_loss(logits, labels):
    B = logits.shape[0]
    ce_loss = F.cross_entropy(logits, labels, reduction='none')
    
    value = torch.quantile(ce_loss, q.to(ce_loss.device), dim=0, keepdim=False)
    
    chosen = (ce_loss < value[0]).long()
    
    q_ce_loss = ce_loss[chosen.bool()].sum()
    
    re_B = labels[(1-chosen).bool()].shape[0]
    re_labels = torch.zeros(re_B, 10).to(ce_loss.device).scatter_(1, labels[(1-chosen).bool()].unsqueeze(1), 1/num)
    #re_labels += torch.zeros(re_B, 10).to(ce_loss.device).scatter_(1, logits[(1-chosen).bool()].argmax(1).unsqueeze(1), 1/num)
    re_labels += torch.zeros(re_B, 10).to(ce_loss.device).scatter_(1, logits[(1-chosen).bool()].topk(3, dim=1)[1], 1/num)
	
    q_ce_loss += soft_ce(logits[(1-chosen).bool()], re_labels).sum()
    
    return q_ce_loss/B

if __name__ == "__main__":
    B = 100
    a = torch.randn((B,10))
    b = torch.from_numpy(np.random.choice(10,B,replace=True)).long()
    
    c = pseudo_ce_loss(a,b)
    print(c)
    