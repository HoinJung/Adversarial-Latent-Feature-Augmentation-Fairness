# Adversarial Latent Feature Augmentation for Fairness
## Introduction
ALFA aims to produce a augmented feature on the latent space by fairness attack. Fine-tuning on the augmented feature rotate the decision boundary recover the unfair region, where the false positive rate and false negative rate are disproportionately high.

## Install
### Requirements
  ```
cuda=11.6.1
numpy=1.23.5
pandas=1.5.3
python=3.10.9
torch=2.0.1
  ```
### Clone Repository
```
git clone https://github.com/hin1115/Adversarial-Latent-Feature-Augmentation-Fairness.git
```


## Data preprocessing

```
python preprocessing/adult_data_processing.py
python preprocessing/compas_data_processing.py
python preprocessing/german_data_processing.py
python preprocessing/drug__data_processing.py
```

## Run Code
For a fair comparison, ALFA runs 10 times and present the mean and standard deviation for each evaluation metric.
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

## Test
After the training, the evaluation will be conducted automatically.

## Citation
TBD

## LICENSE
TBD
