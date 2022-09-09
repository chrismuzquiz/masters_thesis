"""
Independent Single-output GPs with Compound Kernels

* The string kernel is not used in these models due to computational limitations
* We drop MAT(nu=1.5) and MAT(nu=2.5) since they are comparable in performance MAT(nu=0.5)

The compound kernels are constructed using forward selection. We drop MAT(nu=1.5) and MAT(nu=2.5)
since they are comparable in performance MAT(nu=0.5)
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

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, choice_kernel):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(choice_kernel)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)



if __name__ == '__main__':

    filename1 = 'independent_sogp_compound_total.tsv'
    filename2 = 'independent_sogp_compound_task_specific.tsv'

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

    I = torch.eye(separation)

    # training iterations kept constant
    training_iterations = 100


    base_kernel_objs = [
        LinearKernel3(sizes), LinearKernel4(sizes), LinearKernel5(sizes),
        Polynomial2Kernel3(sizes), Polynomial2Kernel4(sizes), Polynomial2Kernel5(sizes),
        RBFKernel3(sizes), RBFKernel4(sizes), RBFKernel5(sizes),
        MaternKernel3(sizes, nu=1/2), MaternKernel4(sizes, nu=1/2), MaternKernel5(sizes, nu=1/2),
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

    def ker_base(k1):
        return ScaleKernel(deepcopy(k1[0])), f"{k1[1]}"

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



    def find2nd(string, substring):
        return string.find(substring, string.find(substring) + 1)
    def add(a, b):
        return '+(' + a + ',' + b + ')'
    def mul(a, b):
        return '*(' + a + ',' + b + ')'

    def expand(ex):
        comma_1 = ex.find(',')
        if comma_1 == -1:
            return ex

        comma_2 = find2nd(ex, ',')
        if comma_2 == -1:
            return ex

        op1, op2 = ex[0], ex[comma_1 + 1]
        A = ex[2:comma_1]
        if op1 == '+':
            B = ex[comma_1 + 1:-1]
            return add(A, expand(B))
        else:  # op1 == '*'
            B = ex[comma_1 + 1 + 2:comma_2]
            C = ex[comma_2 + 1:-2]
            return add(expand(mul(A, B)), expand(mul(A, C)))

    def mid_comma_i(ex):
        i = 2
        p = 0
        while (True):
            c = ex[i]
            if c == '(':
                p -= 1
            elif c == ')':
                p += 1
            elif c == ',' and p == 0:
                return i
            i += 1

    def has_linear_add_term(ex):
        # print(ex)
        op = ex[0]
        if op != '*' and op != '+':
            return True if "Linear" in ex else False

        comma_m = mid_comma_i(ex)
        A = ex[2:comma_m]
        B = ex[comma_m + 1:-1]
        if op == "+":
            return has_linear_add_term(A) + has_linear_add_term(B)
        else:  # op == '*':
            return False

    def has_add_linear_component(ex):
        return has_linear_add_term(expand(ex))

    ######################################################################################################
    # Independent kernel compound produced by summing the loocv rmse of each task
    ######################################################################################################
    runtime = time.time()

    headers = [f"Kernel ({training_iterations})"] \
              + list("_".join(i) for i in itertools.product(
        ["loocv_MAE", "loocv_RMSE", "loocv_NLPL", "test_MAE", "test_RMSE"], labels + ["total"])) \
              + list("_".join(i) for i in itertools.product(["test_PearsonR"], labels)) \
              + labels
    df_header = pd.DataFrame([headers])
    df_header.to_csv(filename1, sep='\t', header=False, index=False, mode='a')

    curr_best_kernel = None # base kernels are cycled in first iteration
    curr_best_loocv_rmse = 100000 # arbitrarily high value
    num_compounds = -1 # counter increments with each iteration

    while True:
        prev_best_kernel = curr_best_kernel
        prev_best_loocv_rmse = curr_best_loocv_rmse
        num_compounds += 1

        # start with base kernels and go from there
        if num_compounds == 0:
            compounded_kernels = [ker_base(base_kernel) for base_kernel in base_kernels]
        else: # num_compounds > 0
            compounded_kernels = [ker_add(curr_best_kernel, base_kernel) for base_kernel in base_kernels] \
                                 + [ker_mult(curr_best_kernel, base_kernel) for base_kernel in base_kernels]

        loocv_rmses = []

        for compounded_kernel in compounded_kernels:
            curr_kernel, kernel_id = compounded_kernel

            # this could be optimized, but it does the job
            if kernel_id[2:8] == "Matern" \
                    and kernel_id[0] == '*' \
                    and "Matern" in kernel_id[kernel_id.find(','):]\
                or kernel_id[2:13] == "RadialBasis" \
                    and kernel_id[0] == '*' \
                    and "RadialBasis" in kernel_id[kernel_id.find(','):]\
                or kernel_id[2:8] == "Linear" \
                    and kernel_id[0] == '+' \
                    and has_add_linear_component(kernel_id[kernel_id.find(',') + 1:-1]):
                print(f"skipping \t\t\t{kernel_id}")
                loocv_rmses.append(900000)
                continue
            else:
                print(kernel_id)

            K = torch.zeros(tasks, separation, separation)
            mean_predictions = torch.zeros(n - separation, tasks)
            param_info = []

            for task in range(tasks):

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
                    # print('| Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()), end=" ")
                    optimizer.step()

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
            frame.to_csv(filename1, sep='\t', header=False, index=False, mode='a')

            loocv_rmses.append(sums[1].item())

        curr_best_loocv_rmse = min(loocv_rmses)
        best_kernel_index = loocv_rmses.index(curr_best_loocv_rmse)
        curr_best_kernel = compounded_kernels[best_kernel_index]

        # iterate again if there is improvement
        # if curr_best_loocv_rmse > prev_best_loocv_rmse * 0.97:
        if curr_best_loocv_rmse > prev_best_loocv_rmse:
            curr_best_kernel = prev_best_kernel
            break
        # iterations not to exceed 4 compounds
        if num_compounds >= 4:
            break

    runtime = round(time.time() - runtime, 2)
    record = f"(total)\tBest: {curr_best_kernel[1]}, {str(runtime)}\n"
    print(record)
    with open('independent_sogp_compound_runtimes.txt', 'w') as f:
        f.write(record)

    ######################################################################################################
    # Independent kernel compound by observing the loocv rmse for each task individually
    ######################################################################################################


    for task in range(tasks):
        runtime = time.time()

        label = labels[task]
        headers = [f"Kernel ({training_iterations})"] \
                  + list("_".join(i) for i in itertools.product(
            ["loocv_MAE", "loocv_RMSE", "loocv_NLPL", "test_MAE", "test_RMSE", "test_PearsonR"], [label])) \
                  + [label]
        df_header = pd.DataFrame([headers])
        df_header.to_csv(filename2, sep='\t', header=False, index=False, mode='a')

        curr_best_kernel = None  # base kernels are cycled in first iteration
        curr_best_loocv_rmse = 100000  # arbitrarily high value
        num_compounds = -1  # counter increments with each iteration

        while True:
            prev_best_kernel = curr_best_kernel
            prev_best_loocv_rmse = curr_best_loocv_rmse
            num_compounds += 1

            # start with base kernels and go from there
            if num_compounds == 0:
                compounded_kernels = [ker_base(base_kernel) for base_kernel in base_kernels]
            else:  # num_compounds > 0
                compounded_kernels = [ker_add(curr_best_kernel, base_kernel) for base_kernel in base_kernels] \
                                     + [ker_mult(curr_best_kernel, base_kernel) for base_kernel in base_kernels]

            loocv_rmses = []

            for compounded_kernel in compounded_kernels:
                curr_kernel, kernel_id = compounded_kernel

                # this could be optimized, but it does the job
                if kernel_id[2:8] == "Matern" \
                        and kernel_id[0] == '*' \
                        and "Matern" in kernel_id[kernel_id.find(','):] \
                    or kernel_id[2:13] == "RadialBasis" \
                        and kernel_id[0] == '*' \
                        and "RadialBasis" in kernel_id[kernel_id.find(','):] \
                    or kernel_id[2:8] == "Linear" \
                        and kernel_id[0] == '+' \
                        and has_add_linear_component(kernel_id[kernel_id.find(',') + 1:-1]):
                    print(f"skipping \t\t\t{kernel_id}")
                    loocv_rmses.append(900000)
                    continue
                else:
                    print(kernel_id)

                mean_predictions = torch.zeros(n - separation, 1)
                param_info = []

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
                    # print('| Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()), end=" ")
                    optimizer.step()

                # loocv error metrics
                K = model.covar_module(X_train).evaluate() + I * model.likelihood.noise  # torch.Size([1000, 1000])
                K_inv = torch.inverse(K)  # torch.Size([1000, 1000])
                Y_train_vector = Y_train[:, task].unsqueeze(1)  # torch.Size([1000, 1])
                denom = torch.diagonal(K_inv).unsqueeze(1)  # torch.Size([1000, 1])
                mu = Y_train_vector - torch.mm(K_inv, Y_train_vector) / denom
                resid = (Y_train_vector - mu)
                sigma2 = torch.ones(list(Y_train_vector.shape)) / denom

                neg_2_pred_log_prob = sigma2.log() \
                                      + resid.pow(2) / sigma2 \
                                      + (torch.ones(list(Y_train_vector.shape)) * torch.pi * 2).log()
                loocv_nlpl = 0.5 * torch.sum(neg_2_pred_log_prob)  # torch.Size([])

                resid = resid * 100
                loocv_mae = torch.mean(torch.abs(resid), 0)  # torch.Size([1])

                loocv_rmse = torch.mean(resid.pow(2), 0).sqrt()  # torch.Size([1])



                # Set into eval mode
                model.eval()
                likelihood.eval()

                param_info = str(model.state_dict())

                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    predictions = likelihood(model(X_test))
                mean_predictions[:, 0] = predictions.mean

                Y_test_obs = Y_test[:, task].unsqueeze(1)

                # test error metrics
                test_maes = tensor_maes(mean_predictions, Y_test_obs) * 100
                test_rmses = tensor_rmses(mean_predictions, Y_test_obs) * 100
                test_pearsonrs = tensor_pearsonr(mean_predictions, Y_test_obs)

                # format the data
                error_metrics = [loocv_mae.item(), loocv_rmse.item(), loocv_nlpl.item(),
                                 test_maes.item(), test_rmses.item(), test_pearsonrs.item()]
                model_info = [kernel_id, *error_metrics, param_info]

                # add data to dataframe
                frame = pd.DataFrame([model_info])
                frame.to_csv(filename2, sep='\t', header=False, index=False, mode='a')

                loocv_rmses.append(loocv_rmse.item())

            curr_best_loocv_rmse = min(loocv_rmses)
            best_kernel_index = loocv_rmses.index(curr_best_loocv_rmse)
            curr_best_kernel = compounded_kernels[best_kernel_index]

            # iterate again if there is improvement
            # if curr_best_loocv_rmse > prev_best_loocv_rmse * 0.97:
            if curr_best_loocv_rmse > prev_best_loocv_rmse:
                curr_best_kernel = prev_best_kernel
                break
            # iterations not to exceed 4 compounds
            if num_compounds >= 4:
                break

        runtime = round(time.time() - runtime, 2)
        record = f"({label})\tBest: {curr_best_kernel[1]}, {str(runtime)}\n"
        print(record)
        with open('independent_sogp_compound_runtimes.txt', 'a') as f:
            f.write(record)
