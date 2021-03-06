
# -*- coding: utf-8 -*-
"""EWC.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/12cvrWSH0i5LYE8c4ATDfMP-g02fd5c0t
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch import autograd

# hyperparameters
# loss function criterion
criterion = nn.NLLLoss()
# learning rate
lr = 0.001
# set how important old task is to new task
old_new_rate = 10000   
# optimizer
optimizer_name = "Adam"
epoch = 1
train_batch_size = 64
device = "cuda"
log_interval = 1000
max_iter_per_epoch = 100

class EWC:
  # initialize parameters
  def __init__(self, model, old_new_rate):
    self.model  = model.to(device)
    self.old_new_rate = old_new_rate
    self.approximate_mean = 0
    self.approximate_fisher_information_matrix = 0
  
  # function to compute loss regarding to previous task, use an approximate mean and fisher matrix to simplify compute
  def get_old_task_loss(self):
    try:
      losses = []
      for param_name, param in self.model.named_parameters():

        _buff_param_name = param_name.replace('.', '__')        
        estimated_mean = getattr(self.model, '{}_estimated_mean'.format(_buff_param_name))
        estimated_fisher = getattr(self.model, '{}_estimated_fisher'.format(_buff_param_name))
        losses.append((estimated_fisher * (param - estimated_mean) ** 2).sum())
      return (old_new_rate / 2) * sum(losses) 
    except Exception:
      return 0


  # training given model with data
  def train(self, data, target):
    if optimizer_name =="Adam":
      optimizer = optim.Adam(self.model.parameters(), lr=lr)
      output = self.model(data).to(device)

      optimizer.zero_grad()
      loss_new_task = criterion(output, target)
      loss_old_task = self.get_old_task_loss()
      loss = loss_new_task + loss_old_task
      loss.backward()
      optimizer.step()

  # update approximate mean and fisher information matrix
  # use this function after training is over
  def update(self, current_ds, batch_size, num_batch):
    # update approximate mean
    for param_name, param in self.model.named_parameters():
            _buff_param_name = param_name.replace('.', '__')
            self.model.register_buffer(_buff_param_name+'_estimated_mean', param.data.clone())
    
    # update approximate fisher information matrix
    dl = DataLoader(current_ds, batch_size, shuffle=True)
    log_liklihoods = []
    for i, (input, target) in enumerate(dl):
        if i > num_batch:
              break
        input = input.to(device)
        target = target.to(device)
        self.model = self.model.to(device)
        output = F.log_softmax(self.model(input), dim=1)
        log_liklihoods.append(output[:, target])
    log_likelihood = torch.cat(log_liklihoods).mean()
    grad_log_liklihood = autograd.grad(log_likelihood, self.model.parameters())
    _buff_param_names = [param[0].replace('.', '__') for param in self.model.named_parameters()]
    for _buff_param_name, param in zip(_buff_param_names, grad_log_liklihood):
        self.model.register_buffer(_buff_param_name+'_estimated_fisher', param.data.clone() ** 2)