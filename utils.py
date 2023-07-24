import torch
import numpy as np
from sklearn.metrics import average_precision_score, accuracy_score
import torch




def evaluate(model, X_test, y_test, A_test):
    model.eval()

    # calculate DP gap(demographic parity)
    idx_0 = np.where(A_test==0)[0]
    idx_1 = np.where(A_test==1)[0]

    X_test_0 = X_test[idx_0]
    X_test_1 = X_test[idx_1]
    X_test_0 = torch.tensor(X_test_0).cuda().float()
    X_test_1 = torch.tensor(X_test_1).cuda().float()

    pred_0 = model(X_test_0)
    pred_1 = model(X_test_1)

    dp_gap = pred_0.mean() - pred_1.mean()
    dp_gap = abs(dp_gap.data.cpu().numpy())

    # Equalized odds gap
    idx_00 = list(set(np.where(A_test==0)[0]) & set(np.where(y_test==0)[0]))
    idx_01 = list(set(np.where(A_test==0)[0]) & set(np.where(y_test==1)[0]))
    idx_10 = list(set(np.where(A_test==1)[0]) & set(np.where(y_test==0)[0]))
    idx_11 = list(set(np.where(A_test==1)[0]) & set(np.where(y_test==1)[0]))

    X_test_00 = X_test[idx_00]
    X_test_01 = X_test[idx_01]
    X_test_10 = X_test[idx_10]
    X_test_11 = X_test[idx_11]

    X_test_00 = torch.tensor(X_test_00).cuda().float()
    X_test_01 = torch.tensor(X_test_01).cuda().float()
    X_test_10 = torch.tensor(X_test_10).cuda().float()
    X_test_11 = torch.tensor(X_test_11).cuda().float()

    pred_00 = model(X_test_00)
    pred_01 = model(X_test_01)
    pred_10 = model(X_test_10)
    pred_11 = model(X_test_11)

    gap_0 = pred_00.mean() - pred_10.mean()
    gap_1 = pred_01.mean() - pred_11.mean()
    gap_0 = abs(gap_0.data.cpu().numpy())
    gap_1 = abs(gap_1.data.cpu().numpy())

    eo_gap = gap_0 + gap_1

    
    # calculate average precision
    X_test_cuda = torch.tensor(X_test).cuda().float()
    output = model(X_test_cuda)
    y_scores = output[:, 0].data.cpu().numpy()
    # ap = average_precision_score(y_test, y_scores)
    threshold = 0.5
    y_pred = np.where(y_scores>threshold, 1, 0)
    acc = accuracy_score(y_test, y_pred)

    return acc, dp_gap, eo_gap,




class cov_adversarial_loss():
    def __call__(self, dataset,  Xp, A, output, y_max, y_min):
        
        # signed_distance = torch.special.logit(output,eps=1e-7)
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
        z_i_z_bar = torch.FloatTensor(A - np.mean(A)).cuda()
        z_i_z_bar = torch.unsqueeze(z_i_z_bar,1)
        z_i_z_bar = torch.transpose(z_i_z_bar, 0, 1)
        
        d_i_d_bar = signed_distance - torch.mean(signed_distance)
        
        loss = torch.matmul(z_i_z_bar, d_i_d_bar)  /len(Xp)
        if dataset=='compas':
            loss = -loss
        return loss
        


