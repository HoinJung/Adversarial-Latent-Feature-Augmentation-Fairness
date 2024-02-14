
## Data preprocessing

```
python preprocessing/adult_data_processing.py
python preprocessing/compas_data_processing.py
python preprocessing/german_data_processing.py
python preprocessing/drug__data_processing.py
```

## Run Code
- MODEL_NAME: ```mlp```, ```logistic```
- DATASET:```adult```,```compas```,```german```, ```drug```
- ALPHA: float numbers to weight the Sinkhorn loss
- ATTACK_ITER: the number of iteration for attacking step
- ```--base```: creating a pretrained model which is necessary for adversarial data augmentation. Once the pretrained model is obtained, you can drop the commad ```--base```.
- ```--attack```: necessary to execute the attack-based data augmentation for fairness.


ex)
```
python main.py --model logistic --dataset compas --alpha 0.1 --attack --attack_iter 10 --num_exp 1 --epochs 50 --base
```
