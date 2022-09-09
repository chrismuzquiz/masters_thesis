"""
Multi-task GPs using LMC with Basic Kernels

LMCs are constructed by means of forward selection

* The string kernel is not used in these models due to computational limitations
* We drop MAT(nu=1.5) and MAT(nu=2.5) since they are comparable in performance MAT(nu=0.5)
* We introduce the compound kernels that were obtained in independent_sogp_compound.py in our forward selection
    to produce LMC kernels
"""

import pandas as pd

import torch
from tensor_error_metrics import tensor_maes, tensor_rmses, tensor_pearsonr

import gpytorch
from gpytorch.kernels import ScaleKernel
from polynomial_kernel_wrapper import \
    LinearKernel3, LinearKernel4, LinearKernel5, \
    Polynomial2Kernel3, Polynomial2Kernel4, Polynomial2Kernel5
from rbf_kernel_wrapper import RBFKernel3, RBFKernel4, RBFKernel5
from matern_kernel_wrapper import MaternKernel3, MaternKernel4, MaternKernel5

from copy import deepcopy

import itertools

import time

class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, n_tasks, lmc_kernels, covar_ranks):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=n_tasks
        )
        self.covar_module = gpytorch.kernels.LCMKernel(
            lmc_kernels, num_tasks=n_tasks, rank=covar_ranks
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x, interleaved=True)

if __name__ == '__main__':
    total_runtime = time.time()

    filename = 'mtgp_lmc2.tsv'

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


    def ker_add(k1,k2):
        """
        Get the additive kernel and the LISP (list processing) symbolic expression (S-expression) which uses prefix notation
        :param k1:
        :param k2:
        :return:
        """
        return ScaleKernel(deepcopy(k1[0]))+ScaleKernel(deepcopy(k2[0])), f"+({k2[1]},{k1[1]})"

    def ker_mult(k1,k2):
        """
        Get the product kernel and the LISP (list processing) symbolic expression (S-expression) which uses prefix notation
        :param k1:
        :param k2:
        :return:
        """
        return ScaleKernel(deepcopy(k1[0]))*ScaleKernel(deepcopy(k2[0])), f"*({k2[1]},{k1[1]})"




    base_kernel_objs = [
        LinearKernel3(sizes), LinearKernel4(sizes), LinearKernel5(sizes),
        Polynomial2Kernel3(sizes), Polynomial2Kernel4(sizes), Polynomial2Kernel5(sizes),
        RBFKernel3(sizes), RBFKernel4(sizes), RBFKernel5(sizes),
        MaternKernel3(sizes, nu=1 / 2), MaternKernel4(sizes, nu=1 / 2), MaternKernel5(sizes, nu=1 / 2),
        # MaternKernel3(sizes, nu=3/2), MaternKernel4(sizes, nu=3/2), MaternKernel5(sizes, nu=3/2),
        # MaternKernel3(sizes, nu=5/2), MaternKernel4(sizes, nu=5/2), MaternKernel5(sizes, nu=5/2),
    ] + [
        ScaleKernel(MaternKernel3(sizes, nu=1 / 2))+ScaleKernel(MaternKernel3(sizes, nu=1 / 2))\
        +ScaleKernel(MaternKernel3(sizes, nu=1 / 2))+ScaleKernel(MaternKernel3(sizes, nu=1 / 2)),

        ScaleKernel(MaternKernel3(sizes, nu=1 / 2)) + ScaleKernel(RBFKernel3(sizes)) + ScaleKernel(MaternKernel3(sizes, nu=1 / 2)),

        ScaleKernel(MaternKernel3(sizes, nu=1 / 2)) + ScaleKernel(Polynomial2Kernel4(sizes)) \
        + ScaleKernel(MaternKernel3(sizes, nu=1 / 2)) + ScaleKernel(MaternKernel3(sizes, nu=1 / 2)),

        ScaleKernel(MaternKernel3(sizes, nu=1 / 2)) + ScaleKernel(LinearKernel4(sizes)) \
        + ScaleKernel(MaternKernel3(sizes, nu=1 / 2)) + ScaleKernel(MaternKernel3(sizes, nu=1 / 2)),

        ScaleKernel(MaternKernel3(sizes, nu=1 / 2)) * ScaleKernel(RBFKernel3(sizes)),

        ScaleKernel(MaternKernel3(sizes, nu=1 / 2)) + ScaleKernel(Polynomial2Kernel4(sizes)),
    ]

    base_kernel_ids = [
        "Linear[3]", "Linear[4]", "Linear[5]",
        "Polynomial2[3]", "Polynomial2[4]", "Polynomial2[5]",
        "RadialBasis[3]", "RadialBasis[4]", "RadialBasis[5]",
        "Matern(nu=1/2)[3]", "Matern(nu=1/2)[4]", "Matern(nu=1/2)[5]",
        # "Matern(nu=3/2)[3]", "Matern(nu=3/2)[4]", "Matern(nu=3/2)[5]",
        # "Matern(nu=5/2)[3]", "Matern(nu=5/2)[4]", "Matern(nu=5/2)[5]",
    ] + [
        "{Matern(nu=1/2)[3]+Matern(nu=1/2)[3]+Matern(nu=1/2)[3]+Matern(nu=1/2)[3]}",
        "{Matern(nu=1/2)[3]+RadialBasis[3]+Matern(nu=1/2)[3]}",
        "{Matern(nu=1/2)[3]+Polynomial2[4]+Matern(nu=1/2)[3]+Matern(nu=1/2)[3]}",
        "{Matern(nu=1/2)[3]+Linear[4]+Matern(nu=1/2)[3]+Matern(nu=1/2)[3]}",
        "{Matern(nu=1/2)[3]*RadialBasis[3]}",
        "{Matern(nu=1/2)[3]+Polynomial2[4]}",
    ]

    base_kernels = list(zip(base_kernel_objs, base_kernel_ids))



    ##################################################################################

    I = torch.eye(separation)
    Y_vec = torch.flatten(Y_train).unsqueeze(1)  # torch.Size([1750, 1]) # vector of columns

    # input list of kernel ids
    def lmc_kernel_id_str(ids,task_ranks, noise_rank):
        return '(' + '+'.join(i for i in ids) + ')[' + ','.join(str(r) for r in task_ranks) + ']' + f"({noise_rank})"

    ##################################################################################




    headers = [f"Kernel ({training_iterations})"] \
              + list("_".join(i) for i in itertools.product(
        ["loocv_MAE", "loocv_RMSE", "loocv_NLPL", "test_MAE", "test_RMSE"], labels + ["total"])) \
              + list("_".join(i) for i in itertools.product(["test_PearsonR"], labels)) \
              + ["model", "K_t"]
    # + labels \

    df_header = pd.DataFrame([headers])
    df_header.to_csv(filename, sep='\t', header=False, index=False, mode='a')

    # starting point

    lmc_kernels = [deepcopy(base_kernel_objs[9])]
    covar_ranks = [1]
    noise_rank = 0
    lmc_kernel_ids = [base_kernel_ids[9]]
    curr_best_loocv_rmse = 100000

    iters = 0

    while True:

        # curr_lmc = ()
        prev_best_loocv_rmse = curr_best_loocv_rmse
        curr_loocv_rmses = []

        new_lmc_kernels = []
        new_covar_ranks = []
        new_lmc_kernel_ids = []
        new_noise_ranks = []

        for k_i, k_it in enumerate(base_kernels):

            curr_kernel, kernel_id = k_it

            curr_lmc_kernels = [deepcopy(lmc_k) for lmc_k in lmc_kernels]
            curr_lmc_kernels.append(deepcopy(curr_kernel))
            curr_covar_ranks = [*covar_ranks,1]
            curr_kernel_ids = [*lmc_kernel_ids,kernel_id]
            curr_noise_rank = noise_rank

            new_lmc_kernels.append(curr_lmc_kernels)
            new_covar_ranks.append(curr_covar_ranks)
            new_lmc_kernel_ids.append(curr_kernel_ids)
            new_noise_ranks.append(curr_noise_rank)

            curr_kernel_id_str = lmc_kernel_id_str(curr_kernel_ids, curr_covar_ranks, curr_noise_rank)
            param_info = []

            print(curr_kernel_id_str)

            likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=tasks, has_global_noise=True,
                                                                          rank=curr_noise_rank)
            model = MultitaskGPModel(X_train, Y_train, likelihood, tasks, curr_lmc_kernels, curr_covar_ranks)

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

            # loocv
            K_f = model.forward(X_train).covariance_matrix
            if curr_noise_rank > 0:
                noise = torch.kron(model.likelihood.task_noise_covar, I)  # D kron I
            else:
                noise = torch.kron(torch.diag(model.likelihood.task_noises), I)  # D kron I
            K_y_inv = torch.inverse(K_f + noise)

            denom = torch.diag(K_y_inv).unsqueeze(1)  # torch.Size([1750, 1])
            mu = Y_vec - (K_y_inv @ Y_vec) / denom

            resid = Y_vec - mu
            resid = resid.reshape(tasks, separation).T

            sigma2 = torch.ones(list(Y_train.shape)) / denom.reshape(tasks, separation).T

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
            model_info = [curr_kernel_id_str, *error_metrics, *param_info]

            # add data to dataframe
            frame = pd.DataFrame([model_info])
            frame.to_csv(filename, sep='\t', header=False, index=False, mode='a')

            curr_loocv_rmses.append(sums[1].item())

        # iterate over incrementing ranks
        for r_i, c_rank in enumerate(covar_ranks):

            curr_lmc_kernels = [deepcopy(lmc_k) for lmc_k in lmc_kernels]
            curr_covar_ranks = [*covar_ranks]
            if c_rank >= 7:
                continue
            else:
                curr_covar_ranks[r_i] = c_rank + 1
                curr_kernel_ids = [*lmc_kernel_ids]
                curr_noise_rank = noise_rank

                new_lmc_kernels.append(curr_lmc_kernels)
                new_covar_ranks.append(curr_covar_ranks)
                new_lmc_kernel_ids.append(curr_kernel_ids)
                new_noise_ranks.append(curr_noise_rank)

                curr_kernel_id_str = lmc_kernel_id_str(curr_kernel_ids, curr_covar_ranks, curr_noise_rank)

                param_info = []

                print(curr_kernel_id_str)

                likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=tasks, has_global_noise=False,
                                                                              rank=curr_noise_rank)
                model = MultitaskGPModel(X_train, Y_train, likelihood, tasks, curr_lmc_kernels, curr_covar_ranks)

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

                # loocv
                K_f = model.forward(X_train).covariance_matrix
                if curr_noise_rank > 0:
                    noise = torch.kron(model.likelihood.task_noise_covar, I)  # D kron I
                else:
                    noise = torch.kron(torch.diag(model.likelihood.task_noises), I)  # D kron I
                K_y_inv = torch.inverse(K_f + noise)

                denom = torch.diag(K_y_inv).unsqueeze(1)  # torch.Size([1750, 1])
                mu = Y_vec - (K_y_inv @ Y_vec) / denom

                resid = Y_vec - mu
                resid = resid.reshape(tasks, separation).T

                sigma2 = torch.ones(list(Y_train.shape)) / denom.reshape(tasks, separation).T

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
                model_info = [curr_kernel_id_str, *error_metrics, *param_info]

                # add data to dataframe
                frame = pd.DataFrame([model_info])
                frame.to_csv(filename, sep='\t', header=False, index=False, mode='a')

                curr_loocv_rmses.append(sums[1].item())

        for rank_up in range(1):
            if noise_rank == 7:
                continue

            curr_lmc_kernels = [deepcopy(lmc_k) for lmc_k in lmc_kernels]
            curr_covar_ranks = [*covar_ranks]
            curr_kernel_ids = [*lmc_kernel_ids]
            curr_noise_rank = noise_rank + 1

            new_lmc_kernels.append(curr_lmc_kernels)
            new_covar_ranks.append(curr_covar_ranks)
            new_lmc_kernel_ids.append(curr_kernel_ids)
            new_noise_ranks.append(curr_noise_rank)

            curr_kernel_id_str = lmc_kernel_id_str(curr_kernel_ids, curr_covar_ranks, curr_noise_rank)
            param_info = []

            print(curr_kernel_id_str)

            likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=tasks, has_global_noise=True,
                                                                          rank=curr_noise_rank)
            model = MultitaskGPModel(X_train, Y_train, likelihood, tasks, curr_lmc_kernels, curr_covar_ranks)

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

            # loocv
            K_f = model.forward(X_train).covariance_matrix
            noise = torch.kron(model.likelihood.task_noise_covar, I)  # D kron I
            K_y_inv = torch.inverse(K_f + noise)

            denom = torch.diag(K_y_inv).unsqueeze(1)  # torch.Size([1750, 1])
            mu = Y_vec - (K_y_inv @ Y_vec) / denom

            resid = Y_vec - mu
            resid = resid.reshape(tasks, separation).T

            sigma2 = torch.ones(list(Y_train.shape)) / denom.reshape(tasks, separation).T

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
            model_info = [curr_kernel_id_str, *error_metrics, *param_info]

            # add data to dataframe
            frame = pd.DataFrame([model_info])
            frame.to_csv(filename, sep='\t', header=False, index=False, mode='a')

            curr_loocv_rmses.append(sums[1].item())

        curr_best_loocv_rmse = min(curr_loocv_rmses)
        best_loocv_index = curr_loocv_rmses.index(curr_best_loocv_rmse)

        iters += 1

        if prev_best_loocv_rmse < curr_best_loocv_rmse:
            break
        # if iters >= 10:
        #     break

        lmc_kernels = new_lmc_kernels[best_loocv_index]
        covar_ranks = new_covar_ranks[best_loocv_index]
        lmc_kernel_ids = new_lmc_kernel_ids[best_loocv_index]
        noise_rank = new_noise_ranks[best_loocv_index]

    print("BEST: " + lmc_kernel_id_str(lmc_kernel_ids, covar_ranks, noise_rank))


total_runtime = time.time() - total_runtime
print(total_runtime)









