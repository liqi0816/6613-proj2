# Full Project Report

Please see [AI Project 2 Report](/README.assets/AI_Project2_Report.pdf)


# By the end of the project due, we rank 2nd of the new classes task in this competition!
![current ranking](/README.assets/current_ranking.png)<br>



# Summary

We followed the idea from *[Overcoming catastrophic forgetting in neural networks (EWC)](https://arxiv.org/abs/1612.00796)* and conducted experiments on *both* MINST *and* CORe50 datasets. Results shows that our model beats the Resnet-18 baseline by 2% on CORe50.

# Running on MNIST and Fashion MNIST dataset
```
python ewc_mnist.py
```

# Running on CoRe50 dataset
## Setup

Follow the [CVPR Starter](https://github.com/vlomonaco/cvpr_clvision_challenge) Instruction

```bash
sh fetch_data_and_setup.sh
conda env create -f environment.yml
conda activate clvision-challenge
```


## Run

```bash
python main_ewc.py --scenario="multi-task-nc" --sub_dir="ewc3000tt" [ --epochs=EPOCH ] [ --ewc_weight=EWC_WEIGHT ] [ --ewc_explosion_multr_cap=EWC_EXPLOSION_MULTR_CAP ]
```