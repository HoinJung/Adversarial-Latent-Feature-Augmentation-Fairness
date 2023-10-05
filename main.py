import argparse
import os
from train import run_train
import warnings
from latent_adversarial import run_alfa
warnings.filterwarnings("ignore")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='mlp', type=str, help='model(mlp, logistic regression)')
    parser.add_argument('--dataset', default='german', type=str, help='dataset(adult, german, compas')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lam',default=0.001, type=float)
    parser.add_argument('--epochs', default=10, type=int, help='epochs for run_train')
    parser.add_argument('--num_exp', default=10, type=int, help='many experiments are needed for fair comparison')
    parser.add_argument('--epsilon',default=0.5, type=float, help='perturbation clipping')
    parser.add_argument('--lr',default=0.001, type=float)
    parser.add_argument('--alpha',default=0.1, type=float)
    parser.add_argument('--attack_iter',default=100, type=int, help='the number of iteration for the attack')
    parser.add_argument('--gpu_id',default='3', type=str, help='gpu id')
    parser.add_argument('--base', default=False, action=argparse.BooleanOptionalAction,help='whether run base model creating pretrained weight')
    parser.add_argument('--latent',default=False, action=argparse.BooleanOptionalAction,help='run attack')
    
    
    
    
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    if args.base : 
        base_result, base_std = run_train(args, base=True, attack=False)
    
     
    if args.latent:
        print("Run ALFA")
        latent_result, latent_std= run_alfa(args)
    
    

    print("*"*30 + "Result"+"*"*30)
    
    
    
    if args.latent:
        latent_score = { 'latent_acc':  round(latent_result[0],4), 'latent_dp_gap': round(latent_result[1],4), 'latent_eo_gap': round(latent_result[2],4),
        'latent_acc_std':  round(latent_std[0],4), 'latent_dp_gap_std': round(latent_std[1],4), 'latent_eo_gap_std': round(latent_std[2],4),
        }
        print(latent_score)   
