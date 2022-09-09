import torch
import gpytorch

from gpytorch.constraints.constraints import Interval, Positive
from gpytorch.priors import Prior
from gpytorch.kernels import Kernel

class VectorizedStringKernel(gpytorch.kernels.Kernel):
    is_stationary = False
    has_lengthscale = False

    gap_constraint = Interval(1e-4, 1, initial_value=1/2)
    match_constraint = Interval(1e-4, 1, initial_value=1/2)
    ngram_constraint = Positive(initial_value=1)

    def __init__(self,
                 mode=1,
                 data=None,
                 n=5,
                 ard_num_dims=None,
                 batch_shape=torch.Size([]),
                 active_dims=None,
                 lengthscale_prior=None,
                 lengthscale_constraint=None,
                 eps=1e-6,
                 **kwargs):
        """
        :param mode: choose type of embedding used in the string sequence kernel
        0 - word-level glove embedding (requires X0 = Index encoding of strings to reference GloVe embeddings)
        1 - character-level one-hot encoding (requires X1 = Character-level integer encoding)
        2 - word-level one-hot encoding (requires X2 = Word-level integer encoding)
        mode 1 is default: if 0 or 2 is not selected, then mode 1 is used.
        :param n: ngram order. Positive integer needed. Default is 5
        :param gap: gap decay. lambda_g constraint is (0,1]
        :param match: match decay. lambda_m constraint is (0,1]
        """
        self.mode = mode

        self.n = n

        mu_vector = [(1+2*i)/(2*self.n) for i in range(self.n)]
        self.mu = torch.tensor(mu_vector) # for testing purposes
        mu_vector.reverse()
        self.nu = torch.tensor(mu_vector)
            # n = 5 => mu = [0.1, 0.3, 0.5, 0.7, 0.9]
            # n = 3 => mu = [1/3, 3/6, 5/6]

        # VectorizedStringKernel, self
        super().__init__(ard_num_dims=None,
                         batch_shape=torch.Size([]),
                         active_dims=None,
                         lengthscale_prior=None,
                         lengthscale_constraint=None,
                         # self.raw_gap_constraint,
                         # self.raw_match_constraint,
                         eps=1e-6,
                         **kwargs)

        self.register_parameter(name="raw_gap", parameter=torch.nn.Parameter(torch.zeros(1)))
        self.register_constraint("raw_gap", self.gap_constraint)
        self.raw_gap.requires_grad = True

        self.register_parameter(name="raw_match", parameter=torch.nn.Parameter(torch.zeros(1)))
        self.register_constraint("raw_match", self.match_constraint)
        self.raw_match.requires_grad = True

        if True:
            self.register_parameter(name="raw_ngram", parameter=torch.nn.Parameter(torch.zeros(self.n)))
            self.register_constraint("raw_ngram", self.ngram_constraint)
            self.raw_ngram.requires_grad = True

        self.n_data = 1250

        # initialize self.max_len and self.E
        if self.mode == 0:
            self.max_len = 50

            embs = torch.load('word-level_glove_embeddings.pt')
            self.E = tuple(self.remove_padding0(emb) for emb in embs)

        else: # self.mode == 1 | self.mode == 2
            if self.mode == 2:
                self.I = torch.eye(5000)
                self.max_len = 15
            else: # self.mode == 1
                self.I = torch.eye(53)
                self.max_len = 88

            self.E = tuple(self.embed(self.remove_padding(data[i])) for i in range(self.n_data))
        self.S = [[torch.mm(self.E[i], self.E[j].T) for j in range(self.n_data)] for i in range(self.n_data)]

        # setup for self.D
        self.power = torch.ones(self.max_len, self.max_len)
        self.upp_tri = torch.triu(self.power, 1)
        row = torch.stack([torch.ones(self.max_len) * i for i in range(self.max_len)])
        col = row.T

        for i in range(self.max_len - 1):
            self.power[col-i-1 == row] = i
        print("initialized")

    @property
    def gap(self):
        return self.raw_gap_constraint.transform(self.raw_gap)

    @gap.setter
    def gap(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_gap)
        self.initialize(raw_gap=self.raw_gap_constraint.inverse_transform(value))

    @property
    def match(self):
        return self.raw_match_constraint.transform(self.raw_match)

    @match.setter
    def match(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_match)
        self.initialize(raw_match=self.raw_match_constraint.inverse_transform(value))

    @property
    def ngram(self):
        return self.raw_ngram_constraint.transform(self.raw_ngram)

    @ngram.setter
    def ngram(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_ngram)
        self.initialize(raw_ngram=self.raw_ngram_constraint.inverse_transform(value))

    def print_params(self):
        print(f"{self.gap = }, {self.match = }, {self.ngram = }")

    def pretty_params(self):
        return f"{round(self.gap.item(),2)},{round(self.match.item(),2)},{tuple(self.ngram.tolist())}"

    def update_D(self):
        """
        D = { gap ** (j-i+1) if i < j and 0 otherwise
        :return:
        """
        self.D = (self.gap * self.upp_tri) ** self.power

    def lps_D(self, n):
        """
        Get leading principal submatrix of D with order n
        :param n:
        :return:
        """
        return self.D[:n, :n]

    def k(self, i1, i2, match2):
        """
        Auxillary function for forward
        :return:
        """
        S = self.S[i1][i2]
        n1, n2 = S.shape
        D1T = self.lps_D(n1).T
        D2 = self.lps_D(n2)
        K = torch.ones((self.n, n1, n2))
        if n1 >= n2:
            for i in range(1,self.n):
                K[i] = match2 * torch.mm(torch.mm(D1T, (S * K[i-1])), D2)
        else: # n2 > n1
            for i in range(1,self.n):
                K[i] = match2 * torch.mm(D1T, torch.mm((S * K[i-1]), D2))

        return torch.sum(S * K, [1, 2])

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        """
        Method returns a tensor with shape (x1.size()[0], x2.size()[0])
        Method requires tensors to be inputs corresponding to indices of data.
        :param x1:
        :param x2:
        :return:
        """
        self.update_D()

        a1, a2 = x1.size()[0], x2.size()[0]
        match2 = self.match ** 2

        k_matrix = torch.zeros((a1, a2, self.n))
        x1_ = x1.squeeze().int().tolist()
        if isinstance(x1_,int):
            x1_ = [x1_]
        x2_ = x2.squeeze().int().tolist()
        if isinstance(x2_,int):
            x2_ = [x2_]

        for i in range(a1):
            print(f"{i}", end=" ")
            i1 = x1_[i]
            for j in range(a2):
                k_matrix[i, j, :] = self.k(i1,x2_[j],match2)

        K = torch.tensordot(self.ngram, k_matrix, dims=([0], [2]))

        return match2 * K


    def remove_padding0(self, x):
        """
        Intended for use on tensors with ndim > 1
        :return:
        """
        index = x.shape[0] - 1
        while (torch.equal(x[index], torch.zeros(50))):
            index -= 1
        return x[:index + 1]

    def remove_padding(self, x):
        """
        Intended for use on length-n tensors
        :return:
        """
        index = x.shape[0] - 1
        while (x[index] == 0):
            index -= 1
        return x[:index + 1]

    def embed(self, x):
        """
        Intended for use on tensors
        :return:
        """
        return torch.stack([self.I[num] for num in x.int().tolist()])

