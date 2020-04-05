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

For two sequentially happened tasks `A` and `B`. While learning task B, EWC therefore protects the performance in task A by constraining the parameters to stay in a region of low error for task A.  
 
From a probabilistic perspective: optimizing the parameters is tantamount to finding their most probable values given some data D. We can compute this conditional probability p(θ|D) from the prior probability of the parameters p(θ) and the probability of the data p(D|θ) by using Bayes’ rule:

> log p(θ|D) = log p(D|θ) + log p(θ) − log p(D)  

Here is applying Bayes’ rule on two following tasks A & B
> log p(θ|D) = log p(DB|θ) + log p(θ|DA) − log p(DB)

`p(θ|DA)`: All the information about task A must therefore have been absorbed into the posterior distribution p(θ|DA). 
we approximate the posterior as a Gaussian distribution with mean given by the parameters `θA*` and a diagonal precision given by the diagonal of the Fisher information matrix `F`.

The function L that we minimize in EWC is:
> L(θ)=L (θ)+ 􏰂Sumλ/2Fi(θi −θA,i* )^2   


<img src="http://chart.googleapis.com/chart?cht=tx&chl=\Large L(\theta ) = L_{B} (\theta ) + \sum_{i}^{n} \frac{\lambda }{2} F_{i} (\theta _{i} - \theta _{A,i}^{*})^{2}" style="border:none;">


This picture shows the ideal goal of  EWC:

![](https://www.pnas.org/content/114/13/3521/F1.large.jpg "EWC purpose") 


#### 2 EWC for supervised learning 
We trained a fully connected multilayer neural network on several supervised learning tasks in sequence. Within each task, we trained the neural network in the traditional way, namely by shuffling the data and processing it in small batches. After a fixed amount of training on each task, however, we allowed no further training on that task’s dataset. Using EWC, and thus take into account how important each weight is to task A, the network can learn task B well without forgetting task A.

#### 3 EWC for reinforcement learning  
In RL, agents dynamically interact with the environment in order to develop a policy that maximizes cumulative future reward. Previous reinforcement learning approaches to continual learning have either relied on either adding capacity to the network or on learning each task in separate networks, which are then used to train a single network that can play all games. In contrast, the EWC approach presented here makes use of a single network with fixed resources (i.e. network capacity) and has minimal computational overhead.

Specifically, EWC used an agent very similar to that described in [van Hasselt et al., 2016] with few differences: (a) a network with more parameters, (b) a smaller transition table, (c) task-specific bias and gains at each layer, (d) the full action set in Atari, (e) a task-recognition model, and (e) the EWC penalty. Full details of hyper-parameters are described in Appendix app:atari. 

The two most important modifications to the agent: the **task-recognition module**, and the implementation of the **EWC penalty**. 
 
**Task-recognition module**: treat the task context as the latent variable of a Hidden Markov Model. we allow for the addition of new generative models if they explain recent data better than the existing pool of models by using a training procedure.  
**EWC penalty**: At each task switch: compute the Fisher information matrix, and a penalty is added with anchor point given by the current value of the parameters and with weights given by the Fisher information matrix times a scaling factor λ.  

We also allowed the DQN agents to maintain separate short-term memory buffers for each inferred task: these allow action values for each task to be learned off-policy using an experience replay mechanism. As such, the overall system has memory on two time-scales: over **short time-scales**, the experience replay mechanism allows learning in DQN to be based on the interleaved and uncorrelated experiences. At **longer time scales**, know-how across tasks is consolidated by using EWC. Finally, we allowed a small number of network parameters to be game-specific, rather than shared across games. In particular, we allowed each layer of the network to have biases and per element multiplicative gains that were specific to each game.

### Training Experiment

#### MNIST

#### CoRe50


### Conclusion
EWC allows knowledge of previous tasks to be protected during new learning, thereby avoiding catastrophic forgetting of old abilities. It does so by selectively decreasing the plasticity of weights, and thus has parallels with neurobiological models of synaptic consolidation. 


EWC has a run time which is linear in both the number of parameters and the number of training examples. Most notably by approximating the posterior distribution of the parameters on a task (i.e. the weight uncertainties) by a factorized Gaussian. With EWC, three values have to be stored for each synapse: the weight itself, its variance and its mean. 

The ability to learn tasks in succession without forgetting is a core component of biological and artificial intelligence. 

------
Thanks the help from professor, TAs, teammates and the paper.