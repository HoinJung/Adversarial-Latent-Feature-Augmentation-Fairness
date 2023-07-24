
## Data preprocessing
All the raw data have been prepared.    
Run
```
sh data_prep.sh
```

## Run Code
```
python main.py --model {MODEL_NAME} --dataset {DATASET} --base --compare --attack --lamb1 {LAMBDA_1} --lamb2 {LAMBDA_2} --lamb3 {LAMBDA_3} --attack_iter {ATTACK_ITER} 
```
- MODEL_NAME: ```mlp```, ```logistic```
- DATASET:```adult```,```compas```,```german```, ```drug```
- LAMBDA1, LAMBDA2, LAMBDA3: float numbers to weigh each loss term
- ATTACK_ITER: the number of iteration for attacking step
- ```--base```: creating a pretrained model which is necessary for adversarial data augmentation. Once the pretrained model is obtained, you can drop the commad ```--base```.
- ```--compare```: to compare the performance with the baseline. If you don't want to check the performance of baseline, drop the command ```--compare```.
- ```--attack```: necessary to execute the attack-based data augmentation for fairness.


ex)
```
python main.py --model logistic --dataset german --base --compare --attack --lamb1 1 --lamb2 10 --lamb3 10 --attack_iter 1000
```
