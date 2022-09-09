import torch
from gpytorch.kernels.matern_kernel import MaternKernel

class MaternKernelWrapper(MaternKernel):

    def __init__(self, embedding_sizes, X_i, nu=2.5, batch_shape=torch.Size([]), **kwargs):
        _distinct_embs = len(embedding_sizes)
        _breaks = [sum(embedding_sizes[:i]) for i in range(_distinct_embs + 1)]
        self.i_slice = slice(*(_breaks[X_i],_breaks[X_i+1]))
        super().__init__(
            nu = nu,
            ard_num_dims = None,
            batch_shape = batch_shape,
            active_dims = None,
            lengthscale_prior = None,
            lengthscale_constraint = None,
            eps = 1e-6,
            **kwargs)

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        return super().forward(x1[:,self.i_slice], x2[:,self.i_slice], diag=False, last_dim_is_batch=False, **params)

    def __call__(self, x1, x2=None, diag=False, last_dim_is_batch=False, **params):
        return super().__call__(self, x1[:,self.i_slice], x2=None, diag=False, last_dim_is_batch=False, **params)

class MaternKernel3(MaternKernelWrapper):
    def __init__(self, embedding_sizes, nu = 2.5, batch_shape=torch.Size([]), **kwargs):
        super().__init__(embedding_sizes, X_i=3, nu=nu, batch_shape=batch_shape)

class MaternKernel4(MaternKernelWrapper):
    def __init__(self, embedding_sizes, nu = 2.5, batch_shape=torch.Size([]), **kwargs):
        super().__init__(embedding_sizes, X_i=4, nu=nu, batch_shape=batch_shape)

class MaternKernel5(MaternKernelWrapper):
    def __init__(self, embedding_sizes, nu = 2.5, batch_shape=torch.Size([]), **kwargs):
        super().__init__(embedding_sizes, X_i=5, nu=nu, batch_shape=batch_shape)