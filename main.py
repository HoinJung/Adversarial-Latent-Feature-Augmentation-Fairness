import argparse
import os
from train import run_train
import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='mlp', type=str, help='model(mlp, logistic regression)')
    parser.add_argument('--dataset', default='german', type=str, help='dataset(adult, german, compas')
    parser.add_argument('--gamma', default=1.0, type=float, help='gamma, to determine the number of poisoned data')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--lamb1',default=1, type=float, help='Use accuracy attack')
    parser.add_argument('--lamb2', default=5, type=float, help='lambda for the fairness loss')
    parser.add_argument('--lamb3', default=50, type=float, help='lambda for similiarity metric(MINE or Wasserstein)')
    parser.add_argument('--epochs', default=200, type=int, help='epochs for run_train')
    parser.add_argument('--num_exp', default=10, type=int, help='many experiments are needed for fair comparison')
    parser.add_argument('--epsilon',default=1, type=float, help='perturbation clipping')
    parser.add_argument('--attack_iter',default=100, type=int, help='the number of iteration for the attack')
    parser.add_argument('--gpu_id',default='0', type=str, help='gpu id')

    parser.add_argument('--base', default=False, action=argparse.BooleanOptionalAction,help='whether run base model creating pretrained weight')
    parser.add_argument('--compare', default=False, action=argparse.BooleanOptionalAction,help='based on the pretrained weight, run more epochs as the same number of the augmented experiment')
    parser.add_argument('--attack',default=False, action=argparse.BooleanOptionalAction,help='run attack')
    
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    print("*"*30 + "Options"+"*"*30)
    print("pretrain classifier",args.base)
    print("Compare Baseline:",args.compare)
    print("Attack:",args.attack)
    if args.dataset in ['adult']:
        args.batch_size =512
        args.batch_limit = 1024
    if args.dataset in ['compas']:
        args.batch_size =256
        args.batch_limit = 512
    if args.dataset in ['german','drug']:
        args.batch_size =64
        args.batch_limit = 128
    
    
    if args.base : 
        base_result, base_std = run_train(args, base=True, attack=False)
    if args.compare:
        compare_result, compare_std = run_train(args, base=False, attack=False)
    if args.attack:
        attack_result, attack_std= run_train(args, base=False, attack=True)
    

    print("*"*30 + "Result"+"*"*30)
    
    if args.compare:
        compare_score = { 'compare_acc':  round(compare_result[0],4), 'compare_dp_gap': round(compare_result[1],4), 'compare_eo_gap': round(compare_result[2],4),
        'compare_acc_std':  round(compare_std[0],4), 'compare_dp_gap_std': round(compare_std[1],4), 'compare_eo_gap_std': round(compare_std[2],4),
        }
        print(compare_score)
    
    if args.attack:
        attack_score = { 'attack_acc':  round(attack_result[0],4), 'attack_dp_gap': round(attack_result[1],4), 'attack_eo_gap': round(attack_result[2],4),
        'attack_acc_std':  round(attack_std[0],4), 'attack_dp_gap_std': round(attack_std[1],4), 'attack_eo_gap_std': round(attack_std[2],4),
        }
        print(attack_score)   
    
        
            
    

