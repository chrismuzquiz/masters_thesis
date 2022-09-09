"""
Independent Single-output GPs with Basic Kernels

* The string kernel is not used in these models due to computational limitations
"""

import pandas as pd

import torch
from tensor_error_metrics import tensor_maes, tensor_rmses, tensor_pearsonr

import gpytorch
from polynomial_kernel_wrapper import \
    LinearKernel3, LinearKernel4, LinearKernel5, \
    Polynomial2Kernel3, Polynomial2Kernel4, Polynomial2Kernel5
from rbf_kernel_wrapper import RBFKernel3, RBFKernel4, RBFKernel5
from matern_kernel_wrapper import MaternKernel3, MaternKernel4, MaternKernel5
from vectorized_string_kernel_wrapper import \
    VectorizedStringKernel0, VectorizedStringKernel1, VectorizedStringKernel2

from copy import deepcopy

import itertools

import time

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, choice_kernel):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(choice_kernel)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# with gpytorch.settings.fast_computations(covar_root_decomposition=False, log_prob=False, solves=False):
if __name__ == '__main__':

    filename = 'independent_sogp_basic.tsv'

    df = pd.read_csv("affectivetext_data.csv")
    labels = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "valence"]
    df = df.reindex(columns=["headline", *labels])

    # df.ndim = 2
    # df.shape = (1250, 8)
    n = df.shape[0]  # = 1250 data points in total

    torch.set_printoptions(precision=3, edgeitems=3, sci_mode=False)
    # Y contains channels of emotion scores and valence scores
    Y = df[labels]
    Y = torch.tensor(Y.values).contiguous().float()  # torch.Size([1250, 7])
    tasks = Y.shape[1]

    # transform output values to a [0,1] scale for emotion scores and [-1,1] scale for valence scores
    Y_raw = Y
    Y = Y / 100

    X = torch.load('input_tensor.pt').contiguous()

    doc2vec_dim = 4
    sizes = (1, 88, 15, 50, doc2vec_dim, doc2vec_dim)
    _distinct_embs = len(sizes)
    _breaks = [sum(sizes[:i]) for i in range(_distinct_embs + 1)]
    indices = ((_breaks[i], _breaks[i + 1]) for i in range(_distinct_embs))
    (X0, X1, X2, X3, X4, X5) = (X[:, slice(*i)] for i in indices)

    separation = 250
    X_train, X_test = X[:separation], X[separation:]
    Y_train, Y_test = Y[:separation], Y[separation:]

    # training iterations kept constant
    training_iterations = 100

    I = torch.eye(separation)

    single_task_kernels = [
        LinearKernel3(sizes), LinearKernel4(sizes), LinearKernel5(sizes),
        Polynomial2Kernel3(sizes), Polynomial2Kernel4(sizes), Polynomial2Kernel5(sizes),
        RBFKernel3(sizes), RBFKernel4(sizes), RBFKernel5(sizes),
        MaternKernel3(sizes, nu=1 / 2), MaternKernel4(sizes, nu=1 / 2), MaternKernel5(sizes, nu=1 / 2),
        MaternKernel3(sizes, nu=3 / 2), MaternKernel4(sizes, nu=3 / 2), MaternKernel5(sizes, nu=3 / 2),
        MaternKernel3(sizes, nu=5 / 2), MaternKernel4(sizes, nu=5 / 2), MaternKernel5(sizes, nu=5 / 2),
        # VectorizedStringKernel0(sizes, n=4),
        # VectorizedStringKernel1(sizes, data=X1, n=6),
        # VectorizedStringKernel2(sizes, data=X2, n=3),
    ]

    single_task_kernel_ids = [
        "Linear[3]", "Linear[4]", "Linear[5]",
        "Polynomial2[3]", "Polynomial2[4]", "Polynomial2[5]",
        "RadialBasis[3]", "RadialBasis[4]", "RadialBasis[5]",
        "Matern(nu=1/2)[3]", "Matern(nu=1/2)[4]", "Matern(nu=1/2)[5]",
        "Matern(nu=3/2)[3]", "Matern(nu=3/2)[4]", "Matern(nu=3/2)[5]",
        "Matern(nu=5/2)[3]", "Matern(nu=5/2)[4]", "Matern(nu=5/2)[5]",
        # "String[0]", "String[1]", "String[2]",
    ]

    kernel_it = list(zip(single_task_kernels, single_task_kernel_ids))

    ######################################################################################################
    # Independent single-output GPs with task-specific and aggregate metrics
    ######################################################################################################

    runtime = time.time()

    headers = [f"Kernel ({training_iterations})"] \
              + list("_".join(i) for i in itertools.product(
        ["loocv_MAE", "loocv_RMSE", "loocv_NLPL", "test_MAE", "test_RMSE"], labels + ["total"])) \
              + list("_".join(i) for i in itertools.product(["test_PearsonR"], labels)) \
              + labels
    df_header = pd.DataFrame([headers])
    df_header.to_csv(filename, sep='\t', header=False, index=False, mode='a')

    for k_it in kernel_it:
        runtime = time.time()

        curr_kernel, kernel_id = k_it

        K = torch.zeros(tasks, separation, separation)
        mean_predictions = torch.zeros(n - separation, tasks)
        param_info = []

        for task in range(tasks):
            print(kernel_id, '(' + labels[task] + ')', end=" ")

            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            model = ExactGPModel(X_train, Y_train[:, task], likelihood, deepcopy(curr_kernel))

            # Find optimal model hyperparameters
            model.train()
            likelihood.train()

            # Use the adam optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

            # "Loss" for GPs - the marginal log likelihood
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

            for i in range(training_iterations):
                optimizer.zero_grad()
                output = model(X_train)
                loss = -mll(output, Y_train[:, task])
                loss.backward()
                print('| Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()), end=" ")
                optimizer.step()
            print()

            # torch.Size([1000, 1000])
            K[task] = model.covar_module(X_train).evaluate() + I * model.likelihood.noise

            # Set into eval mode
            model.eval()
            likelihood.eval()

            param_info.append(str(model.state_dict()))

            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                predictions = likelihood(model(X_test))
            mean_predictions[:, task] = predictions.mean


        # loocv error metrics
        K_inv = torch.inverse(K)  # torch.Size([7, 1000, 1000])
        Y_train_tensor = (Y_train.T).unsqueeze(2)  # torch.Size([7, 1000, 1])
        denom = torch.diagonal(K_inv, dim1=-2, dim2=-1).T  # torch.Size([1000, 7])
        mu = Y_train - torch.bmm(K_inv, Y_train_tensor).squeeze().T / denom  # torch.Size([1000, 7])
        resid = (Y_train - mu)  # torch.Size([1000, 7])

        sigma2 = torch.ones(list(Y_train.shape)) / denom

        neg_2_pred_log_prob = sigma2.log() \
                              + resid.pow(2) / sigma2 \
                              + (torch.ones(list(Y_train.shape)) * torch.pi * 2).log()
        loocv_nlpl = 0.5 * torch.sum(neg_2_pred_log_prob, 0)

        resid = resid * 100  # torch.Size([1000, 7])
        loocv_mae = torch.mean(torch.abs(resid), 0)  # torch.Size( [7])
        loocv_rmse = torch.mean(resid.pow(2), 0).sqrt()

        # test error metrics
        test_maes = tensor_maes(mean_predictions, Y_test) * 100
        test_rmses = tensor_rmses(mean_predictions, Y_test) * 100
        test_pearsonrs = tensor_pearsonr(mean_predictions, Y_test)

        # format the data
        summable_error_metrics = [loocv_mae.detach(), loocv_rmse.detach(), loocv_nlpl.detach(), test_maes,
                                  test_rmses]
        sums = [torch.sum(m, 0, keepdim=True) for m in summable_error_metrics]
        error_metrics = torch.cat([torch.cat([i, j]) for i, j in zip(summable_error_metrics, sums)]).tolist() \
                        + test_pearsonrs.tolist()
        model_info = [kernel_id, *error_metrics, *param_info]

        # add data to dataframe
        frame = pd.DataFrame([model_info])
        frame.to_csv(filename, sep='\t', header=False, index=False, mode='a')