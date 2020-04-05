# Full Project Report

Please see [`AI Project2 Report.pdf`](README.assets/AI Project2 Report.pdf)

# Summary

We followed the idea from *[Overcoming catastrophic forgetting in neural networks (EWC)](https://arxiv.org/abs/1612.00796)* and conducted experiments on *both* MINST *and* CORe50 datasets. Results shows that our model beats the Resnet-18 baseline by 2% on CORe50.

# Theory

In brains, synaptic consolidation enables continual learning by reducing the plasticity of synapses that are vital to previously learned tasks. We implement an algorithm that performs a similar operation in artificial neural networks by constraining important parameters to stay close to their old values. 

Suppose we have two sequentially happened tasks $A$ and $B$. While learning task $B$, EWC will protect the performance in task $A$ by constraining the parameters to stay in a region of low error for task $A$.

From a probabilistic perspective: optimizing the parameters is tantamount to finding their most probable values given some data $D$. We can compute this conditional probability $\Pr(\theta | D)$  from the prior probability of the parameters $\Pr(\theta)$ and the probability of the data $\Pr(D|\theta)$ by using Bayes’ rule:
$$ \log \Pr(\theta|D) = \log \Pr(D|\theta) + \log \Pr(\theta) - \log \Pr(D) $$

Here is applying Bayes’ rule on two following tasks A & B
$$ \log \Pr(\theta|D) = \log \Pr(D_B|\theta) + \log \Pr(\theta|D_A) - \log \Pr(D_B) $$

$\Pr(\theta | D_A)$: All the information about task A must therefore have been absorbed into the posterior distribution $\Pr(\theta | D_A)$. 
we approximate the posterior as a Gaussian distribution with mean given by the parameters $\theta_A^*$ and a diagonal precision given by the diagonal of the Fisher information matrix $F$.

The function L that we minimize in EWC is:
$$ \Large L(\theta ) = L_{B} (\theta ) + \sum_{i}^{n} \frac{\lambda }{2} F_{i} (\theta _{i} - \theta _{A,i}^{*})^{2} $$




This picture shows the ideal goal of  EWC:

![](README.assets/F1.large.jpg)

# Implementation and Experiment

The CORe50 dataset is a resource demanding one and cannot fit in a regular personal computer. Therefore, we employed a google cloud compute engine with 8 vCPU, 54GB RAM and a Nvidia T4 GPU to carry out this part of the experiment. The implementation used the Torch framework, and source code can be found online[^1]. 

We used the image preprocessing part in starter pack as-is. The preprocessing function serves to satisfy requirements assumed by torchvision pretrained models[^2]. It pads mini-batches of images to at least 224x224 pixels, normalizes images with `mean = [0.485, 0.456, 0.406]` and `std = [0.229, 0.224, 0.225]`. At the end it also shuffles the batch.

The base model is namely ResNet-18, pre-trained by pytorch. It is the most basic ResNet structure, with 17 convolutional layers and 1 fully connected layer. Experiments showed that the baseline model from CVPR competition starter pack could already achieve an accuracy of 1.0 within each subtask. For a better comparison, we kept the same base model, on the top of which added EWC mechanism. The EWC module is shared between MINST experiment and CORe50 experiment.

![resnet-18](README.assets/resnet-18.png)

We trained the network with the data loader bundled in the starter pack. The loader will split 50 classes into 9 tasks/batches, and feed them into the network successively. A portion of the dataset will become validation set, and the same validation set is kept between tasks. 

There are two major hyper-parameter in our model, `ewc_explosion_multr_cap`, which controls how smooth EWC loss should be, and `ewc_weight`, which controls how important EWC loss is. In experiment We frequently observe gradient explosion and loss going to `NaN`. We borrowed the idea of gradient clipping[^3], setting a maximum for EWC loss. To further smoothen the loss curve, we use $\arctan$ as the clipping function. We empirically chose `ewc_explosion_multr_cap` between 5-25 and `ewc_weight` between 5-5000 in our experiment.

The experiment result are shown as in the following two figures:

![](README.assets/average_acc_mtx.svg)![](README.assets/final_acc_mtx.svg)

(due to limit of computation resources, some cells may contain 0 which indicates missing experiment)

As shown above, the best result is obtained with `ewc_explosion_multr_cap=15` and `ewc_weight=50`. The experiment log is attach as appendix. The final result is

```
------------------------------------------
Avg. acc: [0.9471604938271605]
------------------------------------------
Training Time: 7.673416260878245m
Average Accuracy Over Time on the Validation Set: 0.5267215363511659
Total Training/Test time: 7.673416260878245 Minutes
Average RAM Usage: 21473.329210069445 MB
Max RAM Usage: 29456.984375 MB
Experiment completed.
```

Note that while "`Avg. acc:`" is actually the final accuracy after all tasks/batches averaged *over all classes*.

The baseline result is

```
------------------------------------------
Avg. acc: [0.922716049382716]
------------------------------------------
Training Time: 7.656683707237244m
Average Accuracy Over Time on the Validation Set: 0.5145679012345679
Total Training/Test time: 7.656683707237244 Minutes
Average RAM Usage: 21424.665364583332 MB
Max RAM Usage: 29409.3671875 MB
Experiment completed.
```

which shows that our model performs 2% better after *all* tasks/batches.

# Technical Information

## 

[^1]: https://github.com/liqi0816/6613-proj2
[^2]: https://pytorch.org/docs/stable/torchvision/models.html
[^3]: Section 5.2.4, Vanishing and Exploding Gradients, [Neural Network Methods in Natural Language Processing](http://amzn.to/2fwTPCn), 2017.