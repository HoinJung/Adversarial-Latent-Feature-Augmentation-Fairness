import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from dataset import preprocess_data
from model import MLP_encoder,Linear,Identity
from utils import evaluate_latent
import itertools
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from model import Latent_Perturbation, SinkhornDistance
from utils import cov_adversarial_loss


def generate_lists(*args):
    current = 0
    result = []
    for length in args:
        new_list = list(range(current, current + length))
        result.append(new_list)
        current += length
    return result
        

class CustomDataset(Dataset):
    def __init__(self, data, target, sensitive_attribute):
        self.data = data
        self.target = target
        self.sensitive_attribute = sensitive_attribute.astype(int)

    def __len__(self):
        return len(self.target)

    def __getitem__(self, index):
        x = torch.tensor(self.data[index], dtype=torch.float32)
        y = torch.tensor(self.target[index], dtype=torch.float32)
        a = torch.tensor(self.sensitive_attribute[index], dtype=torch.float32)
        return x, y, a
class CustomDataset_latent(Dataset):
    def __init__(self, data, target, sensitive_attribute,latent_feature, perturbed_feature):
        self.data = data
        self.target = target
        self.sensitive_attribute = sensitive_attribute.astype(int)
        self.latent_feature = latent_feature
        self.perturbed_feature = perturbed_feature
    def __len__(self):
        return len(self.target)

    def __getitem__(self, index):
        x = torch.tensor(self.data[index], dtype=torch.float32)
        y = torch.tensor(self.target[index], dtype=torch.float32)
        h_c = torch.tensor(self.latent_feature[index], dtype=torch.float32)
        h_p = torch.tensor(self.perturbed_feature[index], dtype=torch.float32)
        a = torch.tensor(self.sensitive_attribute[index], dtype=torch.float32)
        return x, y, a, h_c, h_p

def latent_attack(args,  batch_X, batch_Y, batch_A,model, model_linear):
    model.eval()
    model_linear.eval()
    N = len(batch_X)

    indices_A_0 = torch.where(batch_A == 0)[0]
    indices_Y_1 = torch.where(batch_Y == 1)[0]
    indices_A_1 = torch.where(batch_A == 1)[0]
    indices_Y_0 = torch.where(batch_Y == 0)[0]
    
    num_samples =int(N//4)
    
    def sampler(A, Y):
        common_indices = torch.tensor(list(set(A.tolist()) & set(Y.tolist())))

        # Check if common_indices is not empty and has non-zero elements
        if len(common_indices) > 0 and torch.any(common_indices > 0):
            if len(common_indices) >= num_samples:
                sub_batch_idx = torch.multinomial(torch.ones_like(common_indices, dtype=torch.float), num_samples, replacement=False)
            else:
                # If not enough common indices, sample with replacement
                sub_batch_idx = torch.multinomial(torch.ones_like(common_indices, dtype=torch.float), num_samples, replacement=True)
            sub_batch_idx = common_indices[sub_batch_idx].tolist()
            return sub_batch_idx
        else:
            # Handle the case when common_indices is empty or invalid
            # Example: return an empty list or some default value
            return []

    batch_idx_0_0 = sampler(indices_A_0,indices_Y_0)
    batch_idx_0_1 = sampler(indices_A_0,indices_Y_1)
    batch_idx_1_0 = sampler(indices_A_1,indices_Y_0)
    batch_idx_1_1 = sampler(indices_A_1,indices_Y_1)
    batch_idx = batch_idx_0_0 + batch_idx_0_1 + batch_idx_1_0 +batch_idx_1_1
    
    X_00 = batch_X[batch_idx_0_0]
    X_10 = batch_X[batch_idx_1_0]
    X_01 = batch_X[batch_idx_0_1]
    X_11 = batch_X[batch_idx_1_1]
    A_00 = batch_A[batch_idx_0_0]
    A_10 = batch_A[batch_idx_1_0]
    A_01 = batch_A[batch_idx_0_1]
    A_11 = batch_A[batch_idx_1_1]
    Y_00 = batch_Y[batch_idx_0_0]
    Y_10 = batch_Y[batch_idx_1_0]
    Y_01 = batch_Y[batch_idx_0_1]
    Y_11 = batch_Y[batch_idx_1_1]
    
 
    
    fair_loss = cov_adversarial_loss()

    W, b = model_linear.classifier.weight, model_linear.classifier.bias
    
    latent_feature_00 = model(X_00)
    latent_feature_01 = model(X_01)
    latent_feature_10 = model(X_10)
    latent_feature_11 = model(X_11)

    distance_00 = abs(torch.matmul(W, latent_feature_00.t()) + b) / torch.norm(W, p=2)
    distance_01 = abs(torch.matmul(W, latent_feature_01.t()) + b) / torch.norm(W, p=2)
    distance_10 = abs(torch.matmul(W, latent_feature_10.t()) + b) / torch.norm(W, p=2)
    distance_11 = abs(torch.matmul(W, latent_feature_11.t()) + b) / torch.norm(W, p=2)
    epsilon = []
    epsilon += [torch.mean(distance_00)]
    epsilon += [torch.mean(distance_01)]
    epsilon += [torch.mean(distance_10)]
    epsilon += [torch.mean(distance_11)]
    
    epsilon =torch.tensor(epsilon)
        
    latent_feature = torch.concat([latent_feature_00,latent_feature_01,latent_feature_10,latent_feature_11])
    batch_A = torch.concat([A_00,A_01,A_10,A_11])
    batch_Y = torch.concat([Y_00,Y_01,Y_10,Y_11])
    batch_X = torch.concat([X_00,X_01,X_10,X_11])
    

    group = generate_lists(len(X_00),len(X_01),len(X_10),len(X_11))
    model_perturbation = torch.nn.Sequential( Latent_Perturbation(latent_feature.shape, epsilon, groups_indices=group))
    model_perturbation.cuda()
    for param in model_perturbation.parameters():
        param.requires_grad = True
    optimizer = optim.Adam(model_perturbation.parameters(), lr=0.1)
    
    sinkhorn = SinkhornDistance(eps=0.1, max_iter=10)
    sinkhorn.cuda()
    y_ = torch.tensor(batch_Y).cuda().float()
    y_max = torch.max(y_)
    y_min = torch.min(y_)
    
    for j, _ in enumerate(range(args.attack_iter)):
        
        model_perturbation.train()
        
        perturbed_latent, delta = model_perturbation(latent_feature) 
        output = model_linear(perturbed_latent)  
    
        loss_fair= fair_loss(batch_A, output, y_max,y_min)
        
        if args.alpha!=0:
            sh_ditance, p, c=  sinkhorn(latent_feature, perturbed_latent)
            loss =  - loss_fair  + args.alpha*sh_ditance
        else :
            loss = - loss_fair
        
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    perturbed_latent, delta = model_perturbation(latent_feature) 
    
    
    return perturbed_latent,latent_feature, batch_X, batch_Y,batch_A, delta

def run_latent_afa(args):
    

    model_name = args.model
    dataset = args.dataset
    batch_size = args.batch_size
    epochs = args.epochs
    num_exp = args.num_exp
    
    
    
    acc = []
    dp_gap = []
    eo_gap = []
   
    for i in tqdm(range(num_exp)):

        X_train, X_val, X_test, Y_train, Y_val, Y_test, A_train,A_val,A_test = preprocess_data(dataset)
        PATH = f'pretrained/{dataset}_{model_name}_base_latent.pth'
        PATH_linear = f'pretrained/{dataset}_{model_name}_base_latent_linear.pth'
        
        out_dim=1
            
        if model_name =='mlp':            
            model = MLP_encoder(input_size=X_train.shape[1],out_dim=out_dim).cuda()
            model_linear = Linear(128,out_dim).cuda()
        elif model_name=='logistic':
            model = Identity().cuda()
            model_linear = Linear(input_size=X_train.shape[1],out_dim=out_dim).cuda()
        
        model.load_state_dict(torch.load(PATH))
        model_linear.load_state_dict(torch.load(PATH_linear))

        acc_val_epoch = []
        dp_gap_val_epoch = []
        eo_gap_val_epoch  = []
        acc_test_epoch = []
        dp_gap_test_epoch = []
        eo_gap_test_epoch  = []
        cnt=0
        train_dataset = CustomDataset(X_train, Y_train,A_train)        
        train_loader = DataLoader(train_dataset, batch_size=batch_size*4, shuffle=True,drop_last=False)
    
        for param in model.parameters():
            param.requires_grad = False
        for param in model_linear.parameters():
            param.requires_grad = False
        
        X_list = []
        Y_list = []
        A_list = []
        perturbeds_features = []
        clean_features=[]
        delta_list = []
        for batch_idx, (batch_X, batch_Y,batch_A) in enumerate(train_loader):
            
            batch_X, batch_Y, batch_A = batch_X.cuda(), batch_Y.cuda(), batch_A.cuda()
            perturbed_latent, clean_latent,batch_X, batch_Y,batch_A,delta= latent_attack(args, batch_X, batch_Y, batch_A,model, model_linear)
            perturbeds_features.append(perturbed_latent.cpu().detach().numpy())
            clean_features.append(clean_latent.cpu().detach().numpy())
            X_list.append(batch_X.cpu().detach().numpy())
            Y_list.append(batch_Y.cpu().detach().numpy())
            A_list.append(batch_A.cpu().detach().numpy())
        
        X_p = np.array(list(itertools.chain(*X_list)))
        Y_p = np.array(list(itertools.chain(*Y_list)))
        A_p = np.array(list(itertools.chain(*A_list)))
        h_p = np.array(list(itertools.chain(*perturbeds_features)))
        h_c = np.array(list(itertools.chain(*clean_features)))
        
        criterion = nn.BCELoss()
        latent_dataset = CustomDataset_latent(X_p, Y_p,A_p,h_c,h_p)        
        latent_loader = DataLoader(latent_dataset, batch_size=batch_size, shuffle=True,drop_last=False)    
        for param in model_linear.parameters():
            param.requires_grad = True
            optimizer = optim.Adam(model_linear.parameters(), lr=args.lr)
            
        for epoch in tqdm(range(epochs)):
            for batch_idx, (batch_X, batch_Y,batch_A,batch_hc, batch_hp) in enumerate(latent_loader):
                model_linear.train()
                batch_X, batch_Y, batch_A = batch_X.cuda(), batch_Y.cuda(), batch_A.cuda()
                batch_hc,batch_hp = batch_hc.cuda(),batch_hp.cuda()
                
                output_p = model_linear(batch_hp)
                output_c = model_linear(batch_hc)
            
                loss = criterion(output_p, batch_Y) + criterion(output_c, batch_Y)
            
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if epoch>=0:
                
                acc_val, dp_gap_val, eo_gap_val = evaluate_latent(model,model_linear, X_val, Y_val, A_val)
                acc_test, dp_gap_test, eo_gap_test = evaluate_latent(model, model_linear,X_test, Y_test, A_test)

                acc_val_epoch.append(acc_val)
                dp_gap_val_epoch.append(dp_gap_val)
                eo_gap_val_epoch.append(eo_gap_val) 
                acc_test_epoch.append(acc_test)
                dp_gap_test_epoch.append(dp_gap_test)
                eo_gap_test_epoch.append(eo_gap_test) 
                
        
        
        idx = acc_val_epoch.index(max(acc_val_epoch))

        
        acc.append(acc_test_epoch[idx])
        dp_gap.append(dp_gap_test_epoch[idx])
        eo_gap.append(eo_gap_test_epoch[idx])
        
        print(f'{acc_test_epoch[idx]:.4f} {dp_gap_test_epoch[idx]:.4f} {eo_gap_test_epoch[idx]:.4f}')
    
    print((np.round(np.mean(acc,axis=0),4),np.round(np.mean(dp_gap,axis=0),4),np.round(np.mean(eo_gap,axis=0),4)),(np.round(np.std(acc,axis=0),4), np.round(np.std(dp_gap,axis=0),4),np.round(np.std(eo_gap,axis=0),4)))
    return (np.mean(acc),np.mean(dp_gap),np.mean(eo_gap)),(np.std(acc), np.std(dp_gap),np.std(eo_gap))


        
            


