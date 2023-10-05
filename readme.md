# Adversarial Latent Feature Augmentation for Fairness
## Introduction
ALFA aims to produce a augmented feature on the latent space by fairness attack. Fine-tuning on the augmented feature rotate the decision boundary recover the unfair region, where the false positive rate and false negative rate are disproportionately high.

## Install
  ```
cuda=11.6.1
numpy=1.23.5
pandas=1.5.3
python=3.10.9
torch=2.0.1
  ```
	```git clone ff```

## Run code
For a fair comparison, ALFA runs 10 times and present the mean and standard deviation for each evaluation metric.
```
python main.py --model {MODEL_NAME} --dataset {DATASET} --base  --latent --alpha {ALPHA} --lam {LAMBDA} --epsilon {EPSILON}
```
- MODEL_NAME: ```mlp```, ```logistic```
- DATASET:```adult```,```compas```,```german```, ```drug```
- LAMBDA, ALPHA : float numbers , ```ALPHA={0,0.1,1,10,100}```,  ```LAMBDA=[0,1]```
- EPSILON : perturbation range
- ATTACK_ITER: the number of iteration for attacking step
- ```--base```: creating a pretrained model which is necessary for adversarial latent feature augmentation. Once the pretrained model is obtained, you can drop the commad ```--base```.
- ```--latent```: necessary to execute the attack-based data augmentation for fairness.


ex)
```
python main.py --model logistic --dataset compas --latent --alpha 10 --lam 0.25 --attack_iter 100 --epochs 10 --epsilon 0.5 --base
python main.py --model mlp --dataset drug  --latent --alpha 1 --lam 0.75 --attack_iter 100 --epochs 100 --epsilon 0.5 --base
```
## Test
After the training, the evaluation will be conducted automatically.

## Citation
TBD

## LICENSE
TBD
