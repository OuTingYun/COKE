# import math
import torch
# import torch.nn.functional as F
# from time import time
# from torch import nn
# from torch.optim import Adam
# from torch.utils.data import DataLoader
from sklearn.preprocessing import PolynomialFeatures
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
# from pygam import LinearGAM, s
import numpy as np

class GPR_mine:
    def __init__(self, optimize=False):
        self.is_fit = False
        self.train_X, self.train_y = None, None
        self.params = {"l": 1, "sigma_f": 1}
        self.optimize = optimize
        self.alpha = 1e-10
        self.m = None

    def fit(self, y, median, p_eu):
        self.train_y = y.clone()
        K = self.kernel(median, p_eu)
        torch.fill_diagonal(K, 1)
        self.K_trans = K.clone()
        K[torch.diag_indices_from(K)] += self.alpha

        self.L_ = torch.cholesky(K, upper=False)
        self._K_inv = None
        self.alpha_ = torch.cholesky_solve(self.train_y.unsqueeze(1), self.L_, upper=False)
        self.is_fit = True

    def predict(self, return_std=False):
        if not self.is_fit:
            print("GPR Model not fit yet.")
            return

        K_trans = self.K_trans
        y_mean = torch.matmul(K_trans, self.alpha_)
        if return_std == False:
            return y_mean.squeeze()
        else:
            raise ('To cal std')

    def kernel(self, median, p_eu):
        p_eu_nor = p_eu / median
        K = torch.exp(-0.5 * p_eu_nor)
        K = torch.squareform(K)
        return K


class get_Reward_gpu(object):

    def __init__(self, maxlen, inputdata, score_type='BIC', reg_type='LR', alpha=1.0, med_w=1.0, median_flag=False):
        self.maxlen = maxlen
        self.alpha = alpha
        self.med_w = med_w
        self.med_w_flag = median_flag
        self.d = {}
        self.d_RSS = [{} for _ in range(maxlen)]
        self.inputdata = inputdata.float()
        self.n_samples = inputdata.shape[0]
        self.bic_penalty = torch.log(torch.tensor(inputdata.shape[0])) / inputdata.shape[0]

        if score_type not in ('BIC', 'BIC_different_var'):
            raise ValueError('Reward type not supported.')
        if reg_type not in ('LR', 'QR', 'GPR', 'GPR_learnable'):
            raise ValueError('Reg type not supported')
        self.score_type = score_type
        self.reg_type = reg_type

        self.poly = PolynomialFeatures()

        if self.reg_type == 'GPR_learnable':
            self.kernel_learnable = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) \
                                    + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-10, 1e+1))
        elif reg_type == 'LR':
            self.ones = torch.ones((inputdata.shape[0], 1), dtype=torch.float32)
            X = torch.cat((self.inputdata, self.ones), dim=1)
            self.X = X
            self.XtX = torch.matmul(X.t(), X)
        elif reg_type == 'GPR':
            self.gpr = GPR_mine()
            m = inputdata.shape[0]
            self.gpr.m = m
            dist_matrix = []
            for i in range(m):
                for j in range(i + 1, m):
                    dist_matrix.append(torch.square(inputdata[i] - inputdata[j]))
            self.dist_matrix = torch.stack(dist_matrix)

    def cal_rewards(self, graphs, positions=None):
        rewards_batches = []
        for graphi, position in zip(graphs, positions):
            reward_ = self.calculate_reward_single_graph(graphi, position=position)
            rewards_batches.append(reward_.cpu().detach().numpy())
        return torch.from_numpy(np.array(rewards_batches))

    def calculate_yerr(self, X_train, y_train, XtX, Xty):
        if self.reg_type == 'LR':
            return self.calculate_LR(X_train, y_train, XtX, Xty)
        elif self.reg_type == 'QR':
            return self.calculate_QR(X_train, y_train)
        elif self.reg_type == 'GPR':
            return self.calculate_GPR(X_train, y_train, XtX, Xty)
        elif self.reg_type == 'GPR_learnable':
            return self.calculate_GPR_learnable(X_train, y_train)
        else:
            assert False, 'Regressor not supported'

    def calculate_LR(self, X_train, y_train, XtX, Xty):
        theta = torch.linalg.solve(XtX, Xty)
        y_pre = torch.matmul(X_train, theta)
        y_err = y_pre - y_train
        return y_err

    def calculate_QR(self, X_train, y_train):
        X_train = self.poly.fit_transform(X_train)[:, 1:]
        return self.calculate_LR(X_train, y_train)

    def calculate_GPR(self, X_train, y_train, K1, K2):
        p_eu = K1  # TODO our K1 don't sqrt
        med_w = torch.median(p_eu)
        self.gpr.fit(y_train, med_w, p_eu)
        pre = self.gpr.predict()
        return y_train - pre

    def calculate_GPR_learnable(self, X_train, y_train):
        gpr = GPR(kernel=self.kernel_learnable, alpha=0.0).fit(X_train, y_train)
        return y_train.view(-1, 1) - gpr.predict(X_train).view(-1, 1)

    def calculate_reward_single_graph(self, graph_batch, position=None):
        graph_to_int2 = list(position.type(torch.int32))
        graph_batch_to_tuple = tuple(graph_to_int2)
        if graph_batch_to_tuple in self.d:
            graph_score = self.d[graph_batch_to_tuple]
            return graph_score
        RSS_ls = []
        for i in range(self.maxlen):
            RSSi = self.cal_RSSi(i, graph_batch)
            RSS_ls.append(RSSi)
        RSS_ls = torch.tensor(RSS_ls)

        if self.score_type == 'BIC':
            BIC = torch.log(torch.sum(RSS_ls) / self.n_samples + 1e-8) + torch.sum(graph_batch) * self.bic_penalty / self.maxlen
        elif self.score_type == 'BIC_different_var':
            BIC = torch.sum(torch.log(RSS_ls / self.n_samples + 1e-8))  # + np.sum(graph_batch)*self.bic_penalty

        self.d[graph_batch_to_tuple] = -BIC

        return -BIC

    def cal_RSSi(self, i, graph_batch):
        col = graph_batch[:, i]
        str_col = str(col)
        if str_col in self.d_RSS[i]:
            RSSi = self.d_RSS[i][str_col]
            return RSSi

        if torch.sum(col) < 0.1:
            y_err = self.inputdata[:, i]
            y_err = y_err - torch.mean(y_err)
        else:
            cols_TrueFalse = col > 0.5
            if self.reg_type == 'LR':
                cols_TrueFalse = torch.cat((cols_TrueFalse, torch.tensor([True], dtype=torch.bool,device=cols_TrueFalse.device)))
                X_train = self.X[:, cols_TrueFalse]
                y_train = self.X[:, i]

                XtX = self.XtX[:, cols_TrueFalse][cols_TrueFalse, :]
                Xty = self.XtX[:, i][cols_TrueFalse]

                y_err = self.calculate_yerr(X_train, y_train, XtX, Xty)
            # else:
            #     X_train = self.inputdata[:, cols_TrueFalse]
            #     y_train = self.inputdata[:, i]
            #     S = s(0,n_splines=4,)
            #     for i in range(1,X_train.shape[1]):
            #         S+=s(i,n_splines=4,)
            #     gam = LinearGAM().fit(X_train,y_train)
            #     y_err = y_train-gam.predict(X_train)
            elif self.reg_type == 'GPR':
                X_train = self.inputdata[:, cols_TrueFalse]
                y_train = self.inputdata[:, i]
                p_eu = torch.pdist(X_train, p=2)
                if self.med_w_flag:
                    self.med_w = torch.median(p_eu)
                train_y = y_train.clone()
                p_eu_nor = p_eu / self.med_w
                K = torch.exp(-0.5 * p_eu_nor)
                K = torch.squareform(K)

                torch.fill_diagonal(K, 1)
                K_trans = K.clone()
                K[torch.diag_indices_from(K)] += self.alpha
                L_ = torch.cholesky(K, upper=False)
                alpha_ = torch.cholesky_solve(train_y.unsqueeze(1), L_, upper=False)

                y_mean = torch.matmul(K_trans, alpha_)
                y_err = y_train - y_mean.squeeze()

            elif self.reg_type == 'GPR_learnable':
                X_train = self.inputdata[:, cols_TrueFalse]
                y_train = self.inputdata[:, i]
                y_err = self.calculate_yerr(X_train, y_train, X_train, y_train)

        RSSi = torch.sum(torch.square(y_err))
        self.d_RSS[i][str_col] = RSSi

        return RSSi

    def penalized_score(self, score_cyc, lambda1=1, lambda2=1):
        score, cyc = score_cyc
        return score + lambda1 * torch.float(cyc > 1e-5) + lambda2 * cyc

    def update_scores(self, score_cycs):
        ls = []
        for score_cyc in score_cycs:
            ls.append(score_cyc)
        return ls

    def update_all_scores(self):
        score_cycs = list(self.d.items())
        ls = []
        for graph_int, score_l in score_cycs:
            ls.append((graph_int, score_l))
        return sorted(ls, key=lambda x: x[1], reverse=True)
