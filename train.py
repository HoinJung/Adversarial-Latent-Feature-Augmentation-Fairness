import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from dataset import preprocess_data
from model import MLP,LogisticRegression
from utils import evaluate
import itertools
import os
from attack import run_attack




def run_train(args, base, attack):
    
    torch.cuda.set_device(f'cuda:{args.gpu_id}')
    
    pretrain=True
    save_model=False
    if base:
        pretrain=False
        save_model=True

    model_name = args.model
    dataset = args.dataset
    batch_size = args.batch_size
    epochs = args.epochs
    num_exp = args.num_exp
    epsilon = args.epsilon
    gamma = args.gamma
    lamb1 = args.lamb1
    lamb2 = args.lamb2
    lamb3 = args.lamb3
    batch_limit = args.batch_limit
    attack_iter = args.attack_iter

    acc = []
    dp_gap = []
    eo_gap = []

    PATH = f'pretrained/{dataset}_{model_name}_base.pth'
    
    if save_model: 
        num_exp = 1
        print("Training base model")
        if not os.path.exists('pretrained'):
            os.makedirs('pretrained')
    
    for i in tqdm(range(num_exp)):
        X_train, X_val, X_test, Y_train, Y_val, Y_test, A_train,A_val,A_test, sensitives = preprocess_data(dataset)
        if attack:
            if i == 0 :
                print("Retraining after attack-based augmentation")
            poisoned_data = run_attack(dataset = dataset, model_name = model_name, gamma = gamma,  lamb2 =lamb2,   lamb3 =lamb3, lamb1 = lamb1,\
            epsilon = epsilon, attack_iter = attack_iter, batch_limit = batch_limit)
            before = len(X_train)
            X_p, Y_p, A_p, S_p = poisoned_data
            X_p = np.array(list(itertools.chain(*X_p)))
            Y_p = np.array(list(itertools.chain(*Y_p)))
            A_p = np.array(list(itertools.chain(*A_p)))
            S_p = np.array(list(itertools.chain(*S_p)))
            X_train = np.concatenate([X_train, X_p], axis=0)
            Y_train = np.concatenate([Y_train, Y_p], axis=0)
            A_train = np.concatenate([A_train, A_p], axis=0)
            sensitives = np.concatenate([sensitives, S_p], axis=0)
            after = len(X_train)
            if i==0:

                print(f"Poisoned data augmentation: {before} -> {after}")


        # initialize model
        base_iter_num = int(len(X_train)/batch_size)
        iter_num = epochs * base_iter_num
        if model_name =='mlp':
            model = MLP(input_size=len(X_train[0])).cuda()

        elif model_name=='logistic':
            model = LogisticRegression(input_size=len(X_train[0])).cuda()

        if pretrain : 
            model.load_state_dict(torch.load(PATH))

        optimizer = optim.Adam(model.parameters(), lr=0.001)

                
        criterion = nn.BCELoss()

        acc_val_epoch = []
        dp_gap_val_epoch = []
        eo_gap_val_epoch  = []
        acc_test_epoch = []
        dp_gap_test_epoch = []
        eo_gap_test_epoch  = []
        
        model_save=[]
        
        
        for i in range(iter_num):   
            
            # Random Selection
            batch_idx = np.random.choice(range(len(X_train)), size=int(batch_size), replace=False).tolist()
            # make torch.tensor
            batch_X = X_train[batch_idx]
            batch_Y = Y_train[batch_idx]
            batch_A = A_train[batch_idx]
            # batch_sensitives = sensitives[batch_idx]
            
            batch_X = torch.tensor(batch_X).cuda().float() 
            batch_Y = torch.tensor(batch_Y).cuda().float()
            batch_A = torch.tensor(batch_A).cuda().float()
            model.train()
                
            output = model(batch_X)
                
            loss = criterion(output, batch_Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i>10:
                if i%1==0:    
                    acc_test, dp_gap_test, eo_gap_test = evaluate(model, X_test, Y_test, A_test)
                    acc_val, dp_gap_val, eo_gap_val = evaluate(model, X_val, Y_val, A_val)
                    
                    acc_val_epoch.append(acc_val)
                    dp_gap_val_epoch.append(dp_gap_val)
                    eo_gap_val_epoch.append(eo_gap_val) 
                    acc_test_epoch.append(acc_test)
                    dp_gap_test_epoch.append(dp_gap_test)
                    eo_gap_test_epoch.append(eo_gap_test) 
                    if save_model : 
                        model_save.append(model.state_dict())

        
        idx = dp_gap_val_epoch.index(min(dp_gap_val_epoch))
        
        acc.append(acc_test_epoch[idx])
        dp_gap.append(dp_gap_test_epoch[idx])
        eo_gap.append(eo_gap_test_epoch[idx])
        
        if save_model:
            torch.save(model_save[idx], PATH)
    
    return (np.mean(acc),np.mean(dp_gap),np.mean(eo_gap)),(np.std(acc), np.std(dp_gap),np.std(eo_gap))


        
            


