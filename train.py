import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from dataset import preprocess_data
from model import MLP_encoder,Linear,Identity
from utils import evaluate_latent
import os
from torch.utils.data import DataLoader
from dataset import CustomDataset




def run_train(args, base, attack):
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = args.device
    pretrain=False
    save_model=False
    if base:
        pretrain=False
        save_model=True
    
    model_name = args.model
    dataset = args.dataset
    batch_size = args.batch_size
    epochs = args.epochs
    num_exp = args.num_exp
    
    acc = []
    dp_gap = []
    eo_gap = []
    

    if save_model: 
        num_exp=1
        print("Training base model")
        if not os.path.exists('pretrained'):
            os.makedirs('pretrained')


    PATH = f'pretrained/{dataset}_{model_name}_base_latent.pth'
    PATH_linear = f'pretrained/{dataset}_{model_name}_base_latent_linear.pth'

    X_train, X_val, X_test, Y_train, Y_val, Y_test, A_train,A_val,A_test = preprocess_data(dataset)
    out_dim=1    
    criterion = nn.BCELoss()
    
    if model_name =='mlp':    
        model = MLP_encoder(input_size=X_train.shape[1],out_dim=out_dim).cuda()
        model_linear = Linear(128,out_dim).cuda()
    elif model_name=='logistic':
        model = Identity().cuda()
        model_linear = Linear(input_size=X_train.shape[1],out_dim=out_dim).cuda()
            
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    optimizer_linear = optim.Adam(model_linear.parameters(), lr=args.lr)
    
    
    acc_val_epoch = []
    dp_gap_val_epoch = []
    eo_gap_val_epoch  = []
    acc_test_epoch = []
    dp_gap_test_epoch = []
    eo_gap_test_epoch  = []
    confusion_test_epoch = []
        
    model_save=[]
    model_linear_save = []
    train_dataset = CustomDataset(X_train, Y_train,A_train)        
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        
    for i in tqdm(range(epochs)):
        for batch_idx, (batch_X, batch_Y,batch_A) in enumerate(train_loader):
            batch_X, batch_Y, batch_A = batch_X.to(device), batch_Y.to(device), batch_A.to(device)
            model.train()
            model_linear.train()
                
            h = model(batch_X)
            output = model_linear(h)

                        
            loss = criterion(output, batch_Y)
            optimizer.zero_grad()
            optimizer_linear.zero_grad()
            loss.backward()
            optimizer.step()
            optimizer_linear.step()
    
            if i>=0: 
                acc_test, dp_gap_test, eo_gap_test = evaluate_latent(model,model_linear, X_test, Y_test, A_test)
                acc_val, dp_gap_val, eo_gap_val = evaluate_latent(model, model_linear,X_val, Y_val, A_val)

                acc_val_epoch.append(acc_val)
                dp_gap_val_epoch.append(dp_gap_val)
                eo_gap_val_epoch.append(eo_gap_val) 
                acc_test_epoch.append(acc_test)
                dp_gap_test_epoch.append(dp_gap_test)
                eo_gap_test_epoch.append(eo_gap_test) 
                        
                if save_model : 
                    model_save.append(model.state_dict())
                    model_linear_save.append(model_linear.state_dict())

            
            idx = acc_val_epoch.index(max(acc_val_epoch))
            
            acc.append(acc_test_epoch[idx])
            dp_gap.append(dp_gap_test_epoch[idx])
            eo_gap.append(eo_gap_test_epoch[idx])
            
            if save_model:
                torch.save(model_save[idx], PATH)
                torch.save(model_linear_save[idx], PATH_linear)
    print((np.mean(acc,axis=0),np.mean(dp_gap,axis=0),np.mean(eo_gap,axis=0)),(np.std(acc,axis=0), np.std(dp_gap,axis=0),np.std(eo_gap,axis=0)))
    return (np.mean(acc),np.mean(dp_gap),np.mean(eo_gap)),(np.std(acc), np.std(dp_gap),np.std(eo_gap))


        
          