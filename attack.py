import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import preprocess_data
from model import MLP, Perturbation, SinkhornDistance, LogisticRegression
from utils import cov_adversarial_loss



def run_attack(gamma,lamb1, lamb2, lamb3, epsilon, attack_iter,model_name, dataset,batch_limit, simple_aug=False ):
    
    PATH = f'pretrained/{dataset}_{model_name}_base.pth'
    # load data
    X_train, X_val, X_test, Y_train, Y_val, Y_test, A_train,A_val,A_test, sensitives = preprocess_data(dataset)
    # load pretrained model
    if model_name == 'mlp':
        model = MLP(input_size=len(X_train[0])).cuda()
    elif model_name=='logistic':
        model = LogisticRegression(input_size=len(X_train[0])).cuda()
    model.load_state_dict(torch.load(PATH))
    ## Freeze pretrained model
    for param in model.parameters():
        param.requires_grad = False
    acc_loss = nn.BCELoss()
    fair_loss = cov_adversarial_loss()

    batch_size = int(len(X_train)*gamma)
    
    if batch_size >= batch_limit:  
        N=batch_size//batch_limit
        batch_size = batch_limit
    else : 
        N=0
    
    poisoned_data_X = []
    poisoned_data_Y = []
    poisoned_data_A = []
    poisoned_data_sense = []

    for k in range(N+1):
        
        if k==N & N!=0:
            batch_size = (int(len(X_train)*gamma)%batch_limit)
        try : 
            batch_idx_0_0 = np.random.choice(set(np.where(A==0)[0]) & set(np.where(y==0)[0]), size=int(batch_size//4), replace=False).tolist()
            batch_idx_0_1 = np.random.choice(set(np.where(A==0)[0]) & set(np.where(y==1)[0]), size=int(batch_size//4), replace=False).tolist()
            batch_idx_1_0 = np.random.choice(set(np.where(A==1)[0]) & set(np.where(y==0)[0]), size=int(batch_size//4), replace=False).tolist()
            batch_idx_1_1 = np.random.choice(set(np.where(A==1)[0]) & set(np.where(y==1)[0]), size=int(batch_size//4), replace=False).tolist()
            batch_idx = batch_idx_0_0 + batch_idx_0_1 + batch_idx_1_0 +batch_idx_1_1
        except : 
            batch_idx = np.random.choice(np.where(A_train!=2)[0], size=int(batch_size), replace=False).tolist()
        # perturbation model
        model_perturbation = torch.nn.Sequential( Perturbation((len(batch_idx), X_train.shape[1]), epsilon))
        model_perturbation.cuda()
        for param in model_perturbation.parameters():
            param.requires_grad = True
        optimizer = optim.Adam(model_perturbation.parameters(), lr=1e-3)
        batch_X = X_train[batch_idx]
        batch_Y = Y_train[batch_idx]
        batch_A = A_train[batch_idx]
        y_ = torch.tensor(Y_train).cuda().float()
        y_max = torch.max(y_)
        y_min = torch.min(y_)
        
        batch_sensitives = sensitives[batch_idx]
        batch_X = torch.tensor(batch_X).cuda().float() 
        batch_Y = torch.tensor(batch_Y).cuda().float()
        if simple_aug:
            batch_Xp = batch_X
            poisoned_data_X.append(batch_Xp.cpu().detach().numpy())
            poisoned_data_Y.append(batch_Y.cpu().detach().numpy())
            poisoned_data_A.append(batch_A)
            continue
        
        # SinkhornDistance for Wasserstein distance
        sinkhorn = SinkhornDistance(eps=0.1, max_iter=1, dataset=dataset)
        sinkhorn.cuda()
        model.eval()
        curr_loss = 9999
        cnt = 0
        for j, _ in enumerate(range(attack_iter)):
            
            model_perturbation.train()
            sinkhorn.train()
            
            batch_Xp = model_perturbation(batch_X) 
            output = model(batch_Xp)  
            loss_acc = acc_loss(output, batch_Y)
            loss_fair= fair_loss(dataset,batch_Xp, batch_A, output, y_max,y_min)
            if lamb3!=0:
                mi, p, c=  sinkhorn(batch_X, batch_Xp)
                loss = -  lamb1 * loss_acc - lamb2*loss_fair  + lamb3*mi
            else :
                loss = -  lamb1 * loss_acc - lamb2*loss_fair 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_value = loss.item()
            
            if j > 1:
                if loss_value < curr_loss :
                    curr_loss = loss_value  
                    cnt=0
                else :
                    cnt+=1
                if cnt==200: #early stop
                    break      

        poisoned_data_X.append(batch_Xp.cpu().detach().numpy())
        poisoned_data_Y.append(batch_Y.cpu().detach().numpy())
        poisoned_data_A.append(batch_A)
        poisoned_data_sense.append(batch_sensitives)

    return (poisoned_data_X,poisoned_data_Y,poisoned_data_A,poisoned_data_sense)