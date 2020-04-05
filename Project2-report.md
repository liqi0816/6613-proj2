## Project 2 - Continual Learning (CL) for Robotic Perception Report

Zhu, Haoran - hz1922  
Li, Qi - ql1045  
Zhao, Jingyi - jz2216  

For project 2, we followed the work from paper:  [*Overcoming catastrophic forgetting in neural networks (EWC)*](https://arxiv.org/abs/1612.00796), as well as learnt the CL knowledge from paper: Continual Lifelong Learning with Neural Networks: A Review.

Thus, in this project we use EWC(the algorithm from the paper) for CL. We start by addressing the problem of whether elastic weight consolidation could allow deep neural networks to learn a set of complex tasks without catastrophic forgetting. 

### Algorithm Description

Here are 3 parts: idea of EWC, applications of EWC in supervised learning, and reinforcement learning.

**Continual Learning (CL)**: The ability to continually learn over time by accommodating new knowledge while retaining previously learned experiences is referred to as continual or lifelong learning.

**Elastic weight consolidation (EWC)**: an algorithm analogous to *synaptic consolidation* for artificial neural networks, which we refer to as elastic weight consolidation (EWC for short). This algorithm slows down learning on certain weights based on how important they are to multitask learning.  EWC can be used in supervised learning and reinforcement learning problems to train several tasks sequentially without forgetting older ones, in marked contrast to previous deep-learning techniques.

#### 1 Elastic weight consolidation
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

![](https://www.pnas.org/content/114/13/3521/F1.large.jpg "EWC purpose") 


#### 2 EWC for supervised learning 
We trained a fully connected multilayer neural network on several supervised learning tasks in sequence. Within each task, we trained the neural network in the traditional way, namely by shuffling the data and processing it in small batches. After a fixed amount of training on each task, however, we allowed no further training on that task’s dataset. Using EWC, and thus take into account how important each weight is to task A, the network can learn task B well without forgetting task A.

### Training Experiment

#### MNIST

#### CoRe50

The CoRe50 dataset is a resource demanding one and cannot fit in a regular personal computer. Therefore, we employed a google cloud compute engine with 8 vCPU, 54GB RAM and a Nvidia T4 GPU to carry out this part of the expriment. The implementation used the Torch framework and source code can be found online[^1]. Expriments showed that the baseline model from CVPR competation starter pack could already achieve an accuracy of 1.0 within each subtask. For a better comparison, we kept the same base model, on the top of which added EWC mechanism.

We used the image preprocessing part in starter pack as-is. It 

The base model is namely ResNet-18, pre-trained by pytorch. 

We trained the network 

We empirically chose to 

### Conclusion
EWC allows knowledge of previous tasks to be protected during new learning, thereby avoiding catastrophic forgetting of old abilities. It does so by selectively decreasing the plasticity of weights, and thus has parallels with neurobiological models of synaptic consolidation. 

EWC has a run time which is linear in both the number of parameters and the number of training examples. Most notably by approximating the posterior distribution of the parameters on a task (i.e. the weight uncertainties) by a factorized Gaussian. With EWC, three values have to be stored for each synapse: the weight itself, its variance and its mean. 

The ability to learn tasks in succession without forgetting is a core component of biological and artificial intelligence. 

------
Thanks the help from professor, TAs, teammates and the paper.

[^1]: https://github.com/liqi0816/6613-proj2