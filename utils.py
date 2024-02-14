import torch
import numpy as np
from sklearn.metrics import average_precision_score, accuracy_score
import torch
import copy



def evaluate_latent(model, model_linear, X_test, y_test, A_test):
    model.eval()
    model_linear.eval()
    
    latent = model(torch.tensor(X_test).cuda().float())
    output = model_linear(latent)
    A_test = A_test.reshape(-1)
    y_test = y_test.reshape(-1)
    pred = torch.squeeze(torch.where(output >= 0.5, 1, 0), 1).data.cpu().numpy()
    
    idx_0 = np.where(A_test == 0)[0]
    idx_1 = np.where(A_test == 1)[0]
    pred_1 = pred[idx_1] == 1
    pred_0 = pred[idx_0] == 1
    dp_gap = abs(pred_0.mean() - pred_1.mean())
    
    idx_00 = list(set(np.where(A_test == 0)[0]) & set(np.where(y_test == 0)[0]))  
    idx_01 = list(set(np.where(A_test == 0)[0]) & set(np.where(y_test == 1)[0]))  
    idx_10 = list(set(np.where(A_test == 1)[0]) & set(np.where(y_test == 0)[0]))  
    idx_11 = list(set(np.where(A_test == 1)[0]) & set(np.where(y_test == 1)[0]))  
    eo_tp_gap = pred[idx_11].mean() - pred[idx_01].mean()
    
    eo_tn_gap = pred[idx_00].mean() - pred[idx_10].mean()
    
    eo_gap = (abs(eo_tp_gap) + abs(eo_tn_gap)) 
    acc = accuracy_score(y_test, pred)

    return acc, dp_gap, eo_gap

class cov_adversarial_loss():
    def __call__(self,A, output, y_max, y_min):
        A = torch.tensor(A).cuda().float()
        
        sorted_y_c, sorted_y_c_indice = torch.sort(output,axis=0)
        m = 10
        
        del_y = (y_max - y_min)/m
        y_points = [y_min+(i*del_y) for i in range(m+1)]
        y_points[-1]+=1e-7
        d_list=[]
        y_list=[]
        
        for k in range(m):
            start_value = y_points[k]
            end_value = y_points[k+1]
            segment_y = torch.where(((start_value<=sorted_y_c) & (sorted_y_c<end_value)))[0]
            a = (torch.special.logit(end_value,eps=1e-7)-torch.special.logit(start_value,eps=1e-7))/(end_value-start_value)
            Y = sorted_y_c[segment_y]
            
            D = a*(Y-end_value)+ torch.special.logit(end_value,eps=1e-7)
        
            d_list.append(D)
            y_list.append(Y)
        
        signed_distance = torch.cat(d_list,0).gather(0,sorted_y_c_indice.argsort(0))
        
        z_i_z_bar = A-torch.mean(A)
        
        z_i_z_bar = torch.unsqueeze(z_i_z_bar,1)
        z_i_z_bar = torch.transpose(z_i_z_bar, 0, 1)
        
        d_i_d_bar = signed_distance - torch.mean(signed_distance)
        
        loss = torch.matmul(z_i_z_bar, d_i_d_bar)  /len(output)
        
        return abs(loss)
        
