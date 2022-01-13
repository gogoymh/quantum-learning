import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch.nn as nn

import math
import numpy as np
#from torch.multiprocessing import Process, set_start_method, Manager, set_sharing_strategy
from torch.utils.data.sampler import SubsetRandomSampler
import copy

from org_resnet import resnet56 as net
from bias_function import Update

if __name__ == '__main__':
    #set_start_method('spawn')   
    
    def average_weights(w, p):
        """
        Returns the average of the weights.
        """
        if len(w) < 2:
            return copy.deepcopy(w[0])
        p_total = sum(p)
        w_avg = copy.deepcopy(w[0])
        for key in w_avg.keys():
            if w_avg[key].type() == 'torch.cuda.LongTensor':
                continue
            w_avg[key] *= p[0]/p_total
            for i in range(1, len(w)):
                w_avg[key] += (p[i]/p_total) * w[i][key]
                #w_avg[key] = torch.true_divide(w_avg[key], len(w))
        return w_avg

    def EMA(w, delta=0.1):
        """
        Exponential Moving Averaging
        """
        w_avg = copy.deepcopy(w[0])
        for key in w_avg.keys():
            if w_avg[key].type() == 'torch.cuda.LongTensor':
                w_avg[key] = w[1][key]
                continue
            w_avg[key] = (1-delta)*w_avg[key] + delta*w[1][key]
        return w_avg



    test_loader = DataLoader(
                datasets.CIFAR10(
                        '../data/CIFAR10',
                        train=False,
                        download=False,
                        transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(
                                        (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
                                ),
                        ),
                batch_size=64, shuffle=False)#, pin_memory=True)


    device = torch.device("cuda:0")
    aggregated_model = net().to(device)
    global_model = net().to(device)
    
    best_acc = 0
    whole = [i for i in range(50000)]
    for Round in range(300):
        if len(whole) == 0:
            whole = [i for i in range(50000)]
            print("whole reset")
        print(len(whole))
        sample = np.random.choice(whole, 10000, replace=False).tolist()
        whole = list(set(whole) - set(sample))
        
        #Local_weight_manage = Manager()
        #local_weights = Local_weight_manage.list()
        #Local_loss_manage = Manager()
        #local_losses = Local_loss_manage.list()
        
        local_weights = []
        local_losses = []
        
        procs = []
        for partial in range(20):
            total_idx = sample[500*partial:500*(partial+1)]
            idx1 = np.random.choice(total_idx, 400, replace=False)
            idx2 = list(set(total_idx) - set(idx1))
            
            labeled_sampler = SubsetRandomSampler(list(idx1))
            unlabeled_sampler = SubsetRandomSampler(list(idx2))
            
            loader1 = DataLoader(datasets.CIFAR10(
                        "../data/CIFAR10",
                        train=True,
                        download=False,
                        transform=transforms.Compose([
                                transforms.RandomCrop(32, padding=4),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(
                                        (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
                                ),
                        ), batch_size=40, sampler=labeled_sampler, drop_last=True)#, num_workers=0)
            loader2 = DataLoader(datasets.CIFAR10(
                        "../data/CIFAR10",
                        train=True,
                        download=False,
                        transform=transforms.Compose([
                                transforms.RandomCrop(32, padding=4),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(
                                        (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
                                ),
                        ), batch_size=10, sampler=unlabeled_sampler, drop_last=True)#, num_workers=0)
            
            a, b = Update(partial, copy.deepcopy(aggregated_model), net(), device, loader1, loader2)
            
            local_weights.append(a)
            local_losses.append(b)
            
            if (partial+1) % 2 == 0:
                aggregated_weights = average_weights(local_weights, torch.ones(len(local_weights)).float())
                aggregated_model.load_state_dict(aggregated_weights)
                aggregated_model.to(device)
                
                #local_weights = []
                #local_losses = []
            
            #proc = Process(target=Update, args=(partial, copy.deepcopy(aggregated_model), net(), device, loader1, loader2, local_weights, local_losses))
            #proc.start()
            #procs.append(proc)
    
        #for proc in procs:
        #    proc.join()
        '''
        # aggregate
        aggregated_weights = average_weights(local_weights, torch.ones(len(local_weights)).float())
        aggregated_model.load_state_dict(aggregated_weights)
        aggregated_model.to(device)
        '''
        # global model
        global_weights = EMA([global_model.state_dict(), aggregated_model.state_dict()])
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        
        print("[Round:%d] [Loss:%f]" % ((Round+1), loss_avg), end=" ")
    
    
        with torch.no_grad():
            global_model.eval()
            correct1 = 0
            for x, y in test_loader:
                output = global_model(x.float().to(device))
                pred = output.argmax(1, keepdim=True)
                correct1 += pred.eq(y.long().to(device).view_as(pred)).sum().item()
            
            aggregated_model.eval()
            correct2 = 0
            for x, y in test_loader:
                output = aggregated_model(x.float().to(device))
                pred = output.argmax(1, keepdim=True)
                correct2 += pred.eq(y.long().to(device).view_as(pred)).sum().item()
            
            accuracy1 = correct1 / len(test_loader.dataset)
            accuracy2 = correct2 / len(test_loader.dataset)
        
            accuracy = max(accuracy1, accuracy2)
        
            if accuracy >= best_acc:
                print("[Global:%f] [Aggregated:%f] **Best**" % (accuracy1, accuracy2))
                best_acc = accuracy
            else:
                print("[Global:%f] [Aggregated:%f]" % (accuracy1, accuracy2))
    
    
    

        
        
    #scheduler.step()
    
    #torch.save({'model_state_dict': model.state_dict()}, "/data/ymh/gpu_test/resnet56_preact.pth")