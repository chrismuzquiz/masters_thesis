import torch
from gpytorch.kernels.rbf_kernel import RBFKernel

class RBFKernelWrapper(RBFKernel):

    def __init__(self, embedding_sizes, X_i, batch_shape=torch.Size([]), **kwargs):
        _distinct_embs = len(embedding_sizes)
        _breaks = [sum(embedding_sizes[:i]) for i in range(_distinct_embs + 1)]
        self.i_slice = slice(*(_breaks[X_i],_breaks[X_i+1]))
        super().__init__(
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

class RBFKernel3(RBFKernelWrapper):
    def __init__(self, embedding_sizes, batch_shape=torch.Size([]), **kwargs):
        super().__init__(embedding_sizes, X_i=3, batch_shape=batch_shape)

class RBFKernel4(RBFKernelWrapper):
    def __init__(self, embedding_sizes, batch_shape=torch.Size([]), **kwargs):
        super().__init__(embedding_sizes, X_i=4, batch_shape=batch_shape)

class RBFKernel5(RBFKernelWrapper):
    def __init__(self, embedding_sizes, batch_shape=torch.Size([]), **kwargs):
        super().__init__(embedding_sizes, X_i=5, batch_shape=batch_shape)