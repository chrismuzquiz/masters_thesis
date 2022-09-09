import torch
import numpy as np

from sklearn.metrics import mean_squared_error
from scipy.stats.stats import pearsonr

def tensor_maes(t1, t2):
    return torch.mean(torch.abs(t1-t2), 0)

def tensor_rmses(t1, t2):
    """
    function assumes t1.shape = t2.shape and t1.dim() = t2.dim() = 2
    :param t1:
    :param t2:
    :return: tensor of rmses for each pair of respective columns
    """

    its = t1.size()[1]
    rmses = []

    for i in range(its):
        rmse = mean_squared_error(t1[:,i].numpy(),t2[:,i].numpy(),squared=False)
        rmses.append(rmse)

    return torch.tensor(rmses).float()

def tensor_pearsonr(t1,t2):
    """
    function assumes t1.shape = t2.shape and t1.dim() = t2.dim() = 2
    :param t1:
    :param t2:
    :return: tensor of pearsonr coefficients for each pair of respective columns
    """
    try:
        its = t1.size()[1]
        pearsonrs = []

        for i in range(its):
            pr = pearsonr(t1[:,i].numpy(),t2[:,i].numpy())[0]
            pearsonrs.append(pr)

        return torch.tensor(pearsonrs).float()
    except:
        print("pearson r cannot be computed")
        return torch.tensor([])