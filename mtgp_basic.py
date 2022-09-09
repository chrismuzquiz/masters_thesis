"""
Independent Single-output GPs with Basic Kernels

* The string kernel is not used in these models due to computational limitations
* We drop MAT(nu=1.5) and MAT(nu=2.5) since they are comparable in performance MAT(nu=0.5)


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

from copy import deepcopy

import itertools

import time

class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, n_tasks, choice_kernel, kernel_rank):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ZeroMean(), num_tasks=n_tasks
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            choice_kernel, num_tasks=n_tasks, rank=kernel_rank
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


if __name__ == '__main__':
    total_runtime = time.time()

    filename = 'mtgp_basic.tsv'

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

    base_kernel_objs = [
        LinearKernel3(sizes), LinearKernel4(sizes), LinearKernel5(sizes),
        Polynomial2Kernel3(sizes), Polynomial2Kernel4(sizes), Polynomial2Kernel5(sizes),
        RBFKernel3(sizes), RBFKernel4(sizes), RBFKernel5(sizes),
        MaternKernel3(sizes, nu=1 / 2), MaternKernel4(sizes, nu=1 / 2), MaternKernel5(sizes, nu=1 / 2),
        # MaternKernel3(sizes, nu=3/2), MaternKernel4(sizes, nu=3/2), MaternKernel5(sizes, nu=3/2),
        # MaternKernel3(sizes, nu=5/2), MaternKernel4(sizes, nu=5/2), MaternKernel5(sizes, nu=5/2),
    ]

    base_kernel_ids = [
        "Linear[3]", "Linear[4]", "Linear[5]",
        "Polynomial2[3]", "Polynomial2[4]", "Polynomial2[5]",
        "RadialBasis[3]", "RadialBasis[4]", "RadialBasis[5]",
        "Matern(nu=1/2)[3]", "Matern(nu=1/2)[4]", "Matern(nu=1/2)[5]",
        # "Matern(nu=3/2)[3]", "Matern(nu=3/2)[4]", "Matern(nu=3/2)[5]",
        # "Matern(nu=5/2)[3]", "Matern(nu=5/2)[4]", "Matern(nu=5/2)[5]",
    ]

    base_kernels = list(zip(base_kernel_objs, base_kernel_ids))


    headers = [f"Kernel ({training_iterations})"] \
              + list("_".join(i) for i in itertools.product(
        ["loocv_MAE", "loocv_RMSE", "loocv_NLPL", "test_MAE", "test_RMSE"], labels + ["total"])) \
              + list("_".join(i) for i in itertools.product(["test_PearsonR"], labels)) \
              + ["model", "K_t"]
              # + labels \

    df_header = pd.DataFrame([headers])
    df_header.to_csv(filename, sep='\t', header=False, index=False, mode='a')

    for k_it in base_kernels:

        curr_kernel, kernel_id = k_it

        for curr_kernel_rank in range(2,tasks+1):

            curr_kernel_id = kernel_id + f"-{curr_kernel_rank}"

            param_info = []

            print(f"{curr_kernel_id}", end=" ")


            likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=tasks, has_global_noise=True)
            model = MultitaskGPModel(X_train, Y_train, likelihood, tasks, deepcopy(curr_kernel), curr_kernel_rank)

            # Find optimal model hyperparameters
            model.train()
            likelihood.train()

            # Use the adam optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

            # "Loss" for GPs - the marginal log likelihood
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

            for i in range(training_iterations):
                optimizer.zero_grad()
                output = model(X_train)
                loss = -mll(output, Y_train)
                loss.backward()
                # print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
                optimizer.step()

            I = torch.eye(separation)

            covar_i = model.covar_module.task_covar_module.covar_matrix.evaluate() # K_t
            covar_x = model.covar_module.data_covar_module.forward(X_train, X_train) # K_x
            K_f = torch.kron(covar_i,covar_x)
            noise = torch.kron(torch.diag(model.likelihood.task_noises), I) # D kron I
            K_y_inv = torch.inverse(K_f + noise)

            Y_vec = torch.flatten(Y_train.T).unsqueeze(1)  # torch.Size([1750, 1]) # vector of columns

            denom = torch.diag(K_y_inv).unsqueeze(1) # torch.Size([1750, 1])

            mu = Y_vec - (K_y_inv @ Y_vec) / denom

            resid = Y_vec - mu
            resid = resid.reshape(tasks,separation).T

            sigma2 = torch.ones(list(Y_train.shape)) / denom.reshape(tasks,separation).T

            neg_2_pred_log_prob = sigma2.log() \
                                  + resid.pow(2) / sigma2 \
                                  + (torch.ones(list(Y_train.shape)) * torch.pi * 2).log()
            loocv_nlpl = 0.5 * torch.sum(neg_2_pred_log_prob, 0)

            resid = resid * 100  # torch.Size([1000, 7])
            loocv_mae = torch.mean(torch.abs(resid), 0)  # torch.Size( [7])
            loocv_rmse = torch.mean(resid.pow(2), 0).sqrt()

            # Set into eval mode
            model.eval()
            likelihood.eval()

            param_info.append(str(model.state_dict()))

            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                predictions = likelihood(model(X_test))
            mean_predictions = predictions.mean

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
            model_info = [curr_kernel_id, *error_metrics, *param_info, str(covar_i.detach())]

            # add data to dataframe
            frame = pd.DataFrame([model_info])
            frame.to_csv(filename, sep='\t', header=False, index=False, mode='a')

            if kernel_id == "Matern(nu=1/2)[3]" and curr_kernel_rank == 7:
                torch.save(mean_predictions, "matern7_preds.pt")


    total_runtime = time.time() - total_runtime
    print(total_runtime)


















