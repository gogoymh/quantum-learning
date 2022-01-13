# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 15:35:51 2021

@author: Minhyeong
"""
import torch
import matplotlib.pyplot as plt
import math

#a1 = torch.linspace(math.pi,1.5*math.pi,20)
#a2 = torch.ones(10)
#b = torch.cos(a1) + 1
#b = torch.cat((b,a2),dim=0)
#b = [0.1]
#for i in range(29):
#    b.append(b[i]*0.975)

b = (torch.exp(torch.linspace(1,0,300)) - 1)/(math.e - 1)* 128

#c = torch.linspace(0,3.75,300)
c = torch.linspace(1,0,300) * 128

d = torch.log(torch.linspace(math.e,1,300)) * 128



plt.plot([i for i in range(300)], c[:300].tolist(), color='blue')
plt.plot([i for i in range(300)], b.tolist(), color='red')
plt.plot([i for i in range(300)], d[:300].tolist(), color='green')
plt.show()
plt.close()