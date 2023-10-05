

## Run Code
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
