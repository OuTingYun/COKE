import math
from time import time

import numpy as np
# from sklearn.linear_model import LinearRegression
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import cholesky, cho_solve
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.preprocessing import PolynomialFeatures
# from sklearn.gaussian_process.kernels import RBF, WhiteKernel
# from pygam import LinearGAM,s
import torch

class GPR_mine:
    def __init__(self, optimize=False):
        self.is_fit = False
        self.train_X, self.train_y = None, None
        self.params = {"l": 1, "sigma_f": 1}
        self.optimize = optimize
        self.alpha = 1e-10
        self.m = None

    def fit(self, y, median, p_eu):
        self.train_y = np.asarray(y)
        K = self.kernel(median, p_eu)
        np.fill_diagonal(K, 1)
        self.K_trans = K.copy()
        K[np.diag_indices_from(K)] += self.alpha

        self.L_ = cholesky(K, lower=True)
        self._K_inv = None
        self.alpha_ = cho_solve((self.L_, True), self.train_y)
        self.is_fit = True

    def predict(self, return_std=False):
        if not self.is_fit:
            print("GPR Model not fit yet.")
            return

        K_trans = self.K_trans
        y_mean = K_trans.dot(self.alpha_)
        if return_std == False:
            return y_mean
        else:
            raise ('To cal std')

    def kernel(self, median, p_eu):
        p_eu_nor = p_eu / median
        K = np.exp(-0.5 * p_eu_nor)
        K = squareform(K)
        return K


class Reward(object):

    def __init__(self, config, inputdata):
        self.must_included_edge = config.must_exist_edges_adj
        self.maxlen = config.num_variables
        self.alpha = config.alpha
        self.med_w = config.med_w
        self.med_w_flag = config.med_w_flag
        self.d = {}
        self.d_RSS = [{} for _ in range(self.maxlen)]
        self.inputdata = inputdata.astype(np.float32)
        self.n_samples = config.n_samples
        self.bic_penalty = np.log(self.n_samples)/self.n_samples
        self.score_type = config.score_type
        self.reg_type = config.reg_type
        self.knowledge = config.knowledge
        self.panelty_ratio = config.panelty_ratio

        assert self.score_type in ('BIC', 'BIC_different_var'), 'Reward type not supported.'
        assert self.reg_type in ('LR', 'QR', 'GPR'), 'Reg type not supported.'
    
        self.poly = PolynomialFeatures()
        if self.reg_type=='LR':
            self.ones = np.ones((inputdata.shape[0], 1), dtype=np.float32)
            X = np.hstack((self.inputdata, self.ones))
            self.X = X
            self.XtX = X.T.dot(X)
        elif self.reg_type=='GPR':
            self.gpr = GPR_mine()
            m = inputdata.shape[0]
            self.gpr.m = m
            dist_matrix = []
            for i in range(m):
                for j in range(i + 1, m):
                    dist_matrix.append((inputdata[i] - inputdata[j]) ** 2)
            self.dist_matrix = np.array(dist_matrix)

    def cal_rewards(self, 
                    graphs : any, 
                    positions: any,
                    errors=None):
        '''
        enc: torch.tensor (bs, num_variables, n_samples)
        '''
        if type(graphs)==torch.Tensor:
            if "cuda" in graphs.device.type:
                graphs = graphs.cpu().detach().numpy()
            else:
                graphs = graphs.detach().numpy()
        if type(positions)==torch.Tensor:
            if "cuda" in positions.device.type:
                positions = positions.cpu().detach().numpy()
            else:
                positions = positions.detach().numpy()
        
        assert type(graphs)==np.ndarray, 'graphs should be np.ndarray'
        assert type(positions)==np.ndarray, 'positions should be np.ndarray'

        self.must_exist_edge_errors = 0
        self.must_exist_edges_ratios = []
        rewards_batches = []
        for graphi, position in zip(graphs, positions):
            reward_ = self.calculate_reward_single_graph(graphi, position=position)
            rewards_batches.append(reward_)
        return rewards_batches 

    def calculate_yerr(self, X_train, y_train, XtX, Xty):
        if self.reg_type == 'LR':
            return self.calculate_LR(X_train, y_train, XtX, Xty)
        elif self.reg_type == 'QR':
            return self.calculate_QR(X_train, y_train)
        elif self.reg_type == 'GPR':
            return self.calculate_GPR(X_train, y_train, XtX, Xty)


    def calculate_LR(self, X_train, y_train, XtX, Xty):
        theta = np.linalg.solve(XtX, Xty) #!
        y_pre = X_train.dot(theta) #!
        y_err = y_pre - y_train
        return y_err

    def calculate_QR(self, X_train, y_train):
        X_train = self.poly.fit_transform(X_train)[:,1:]
        return self.calculate_LR(X_train, y_train)

    def calculate_GPR(self, X_train, y_train, K1, K2):
        p_eu = K1   #TODO our K1 don't sqrt
        med_w = np.median(p_eu)
        self.gpr.fit(y_train, med_w, p_eu)
        pre = self.gpr.predict()
        return y_train - pre

    
    def calculate_reward_single_graph(self, graph_batch, position=None):
        graph_to_int2 = list(np.int32(position))
        graph_batch_to_tuple = tuple(graph_to_int2)
        if graph_batch_to_tuple in self.d:
            graph_score = self.d[graph_batch_to_tuple]
            return graph_score
        RSS_ls = []
        for i in range(self.maxlen):
            RSSi = self.cal_RSSi(i, graph_batch)
            RSS_ls.append(RSSi)
        RSS_ls = np.array(RSS_ls)

        if self.score_type == 'BIC':
            BIC = np.log(np.sum(RSS_ls)/self.n_samples+1e-8) + np.sum(graph_batch)*self.bic_penalty/self.maxlen
        elif self.score_type == 'BIC_different_var':
            BIC = np.sum(np.log(np.array(RSS_ls)/self.n_samples+1e-8))# + np.sum(graph_batch)*self.bic_penalty

        reward = -BIC - self.panelized_exist_edges_preds(ratio=self.panelty_ratio)
        self.d[graph_batch_to_tuple] = reward

        return reward
    

    def panelized_exist_edges_preds(self, ratio:float=0.1):
        if self.knowledge:
            return (sum(self.must_exist_edges_ratios)/len(self.must_exist_edges_ratios))*ratio
        else:
            return 0
        

    def get_pred_exist_edge(self,graph_batch):
        if self.must_included_edge is None: 
            self.must_exist_edges_ratios.append(0)
        else:
            num_must_inc = self.must_included_edge.sum()
            pred_strong_edge = np.where(self.must_included_edge, graph_batch, 0)
            self.must_exist_edge_errors += (num_must_inc-pred_strong_edge.sum())
            self.must_exist_edges_ratios.append((num_must_inc-pred_strong_edge.sum()) / num_must_inc)

    def cal_RSSi(self, i, graph_batch):
        self.get_pred_exist_edge(graph_batch)
        col = graph_batch[:,i]
        str_col = str(col)
        if str_col in self.d_RSS[i]:
            RSSi = self.d_RSS[i][str_col]
            return RSSi


        if np.sum(col) < 0.1:   #Root Node
            y_err = self.inputdata[:, i]
            y_err = y_err - np.mean(y_err)
        else:
            cols_TrueFalse = col > 0.5
            if self.reg_type == 'LR':
                cols_TrueFalse = np.append(cols_TrueFalse, True)
                X_train = self.X[:, cols_TrueFalse]
                y_train = self.X[:, i] #! 第 i 個variable (預測的目標variable)

                XtX = self.XtX[:, cols_TrueFalse][cols_TrueFalse,:]
                Xty = self.XtX[:, i][cols_TrueFalse]

                y_err = self.calculate_yerr(X_train, y_train, XtX, Xty)

            elif self.reg_type == 'GPR':
                X_train = self.inputdata[:, cols_TrueFalse]
                y_train = self.inputdata[:, i]
                p_eu = pdist(X_train, 'sqeuclidean')
                if self.med_w_flag:
                    self.med_w = np.median(p_eu)
                train_y = np.asarray(y_train)
                p_eu_nor = p_eu / self.med_w
                K = np.exp(-0.5 * p_eu_nor)
                K = squareform(K)

                np.fill_diagonal(K, 1)
                K_trans = K.copy()
                K[np.diag_indices_from(K)] += self.alpha
                L_ = cholesky(K, lower=True)
                alpha_ = cho_solve((L_, True),train_y)

                y_mean = K_trans.dot(alpha_)
                y_err = y_train - y_mean

        RSSi = np.sum(np.square(y_err))
        self.d_RSS[i][str_col] = RSSi

        return RSSi 

    def penalized_score(self, score_cyc, lambda1=1, lambda2=1):
        score, cyc = score_cyc
        return score + lambda1*np.float(cyc>1e-5) + lambda2*cyc
    
    def update_scores(self, score_cycs):
        ls = []
        for score_cyc in score_cycs:
            ls.append(score_cyc)
        return ls
    
    def update_all_scores(self):
        score_cycs = list(self.d.items())
        ls = []
        for graph_int, score_l in score_cycs:
            ls.append((graph_int,score_l))
        return sorted(ls, key=lambda x: x[1],reverse=True)
