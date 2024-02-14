import torch
import torch.nn as nn
import torch.nn.functional as F
import typing as ty
from torch.autograd import Function
from warnings import warn
import torch.nn.init as nn_init
from torch.jit import script
from torch import Tensor



class MLP_encoder(nn.Module):
    def __init__(self, input_size,out_dim):
        super(MLP_encoder, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        h = F.relu(x)
        return h
    
class Linear(nn.Module):
    def __init__(self, input_size,out_dim):
        super(Linear, self).__init__()
        self.classifier = nn.Linear(input_size, out_dim)
    def forward(self, h):
        outputs = self.classifier(h)

        return torch.sigmoid(outputs)
        
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        self.f = nn.Linear(1,1) # dummy
    def forward(self, x):
        return x


class LogisticRegression(torch.nn.Module):
     def __init__(self, input_size, out_dim=1):
        super(LogisticRegression, self).__init__()
        self.classifier = torch.nn.Linear(input_size, out_dim)
     def forward(self, x):
        h=x
        outputs = self.classifier(h)
        return torch.sigmoid(outputs)


class Latent_Perturbation(nn.Module):
    def __init__(self, input_size, epsilon, groups_indices):
        super(Latent_Perturbation, self).__init__()
        self.epsilon = epsilon
        self.groups_indices = groups_indices  # List of indices for each group
        self.W = nn.Embedding(input_size[0], input_size[1])
        self._init_weight_()
        
    def _init_weight_(self):
        nn.init.xavier_uniform_(self.W.weight)
    
    def forward(self, x):
        
        with torch.no_grad():
            for i, indices in enumerate(self.groups_indices):
                group_weights = self.W.weight[indices]
                l2_norm = torch.norm(group_weights.view(group_weights.size(0),-1),dim=1,keepdim=True)
                normalized_weights = group_weights / torch.clamp(l2_norm / (self.epsilon[i]), min=1)
                self.W.weight[indices] = normalized_weights

        x = x + self.W.weight       
        
        return x , self.W.weight
    
class SinkhornDistance(nn.Module):
    """
    Given two empirical measures each with :math:`P_1` locations
    :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
    outputs an approximation of the regularized OT cost for point clouds.
    Args:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default: 'none'
    Shape:
        - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
        - Output: :math:`(N)` or :math:`()`, depending on `reduction`
    """
    def __init__(self, eps, max_iter, reduction='none'):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction
        
    def forward(self, x, y):
        # The Sinkhorn algorithm takes as input three variables :
        C = self._cost_matrix(x, y).cuda()  # Wasserstein cost function
        x_points = x.shape[-2]
        y_points = y.shape[-2]

        
        if x.dim() == 2:
            batch_size = 1
        else:
            batch_size = x.shape[0]

        # both marginals are fixed with equal weights
        mu = torch.empty(batch_size, x_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / x_points+1e-6).squeeze().cuda()
        nu = torch.empty(batch_size, y_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / y_points+1e-6).squeeze().cuda()
        
        u = torch.zeros_like(mu).cuda()
        v = torch.zeros_like(nu).cuda()
        # To check if algorithm terminates because of threshold
        # or max iterations reached
        actual_nits = 0
        # Stopping criterion
        thresh = 1e-1
        
        # Sinkhorn iterations
        for i in range(self.max_iter):
            u1 = u  # useful to check the update
            u = self.eps * (torch.log(mu+1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u
            v = self.eps * (torch.log(nu+1e-8) - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) + v
            err = (u - u1).abs().sum(-1).mean()

            actual_nits += 1
            if err.item() < thresh:
                break

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(self.M(C, U, V))
        # Sinkhorn distance
        cost = torch.sum(pi * C, dim=(-2, -1))

        if self.reduction == 'mean':
            cost = cost.mean()
        elif self.reduction == 'sum':
            cost = cost.sum()

        return cost, pi, C

    def M(self, C, u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    @staticmethod
    def _cost_matrix(x, y,p=2):
        "Returns the matrix of $|x_i-y_j|^p$."
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        
        
        C = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
        return C

    @staticmethod
    def ave(u, u1, tau):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1
        

