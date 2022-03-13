# from numpy import mean,std,zscore,ones,zeros
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from pylab import *

class TPLS_FaultDection():
    def __init__(self,A,A_r,A_y, cumper=0.9, signifi=0.95):
        self.cumper = cumper
        self.signifi = signifi
        self.model = None
        self.A = A
        self.A_r = A_r
        self.A_y = A_y

    def normalize(self, X, Y):
    # 用训练数据的标准差和均值标准化测试数据
        X_mu = np.mean(X, axis=0).reshape((1, X.shape[1]))
        X_std = np.std(X, axis=0).reshape((1, X.shape[1]))
        Xs = (X-X_mu)/X_std
        mu_array = np.ones((Y.shape[0], 1)) * X_mu
        st_array = np.ones((Y.shape[0], 1)) * X_std
        Ys = (Y - mu_array) / st_array
        return Xs, Ys

    def pca(self,X,k):
        sigma = (X.T @ X) / len(X)
        U, S, V = np.linalg.svd(sigma)
        Z = X @ V[:,:k]
        return Z,V[:,:k]

    def pc_number(self,X):
        U, S, V = np.linalg.svd(X)
        if S.shape[0] == 1:
            i = 1
        else:
            i = 0
            var = 0
            while var < self.cumper*sum(S*S):
                var = var+S[i]*S[i]
                i = i + 1
            return i

    def nipals(self,X, Y,max_iter=1000, epsilon=1e-07):
        t_old = 0
        iters = 0
        u = Y[:,0].reshape(Y.shape[0],1)
        while iters < max_iter:
            W = X.T @ u / (np.linalg.norm(X.T @ u))
            T = X @ W
            Q = Y.T @ T / (T.T @ T)
            u = Y @ Q
            t_diff = T - t_old
            t_old = T
            if np.linalg.norm(t_diff) < epsilon:
                P = X.T @ T / (T.T @ T)
                X = X - T @ (P.T)
                break
            else:
                iters += 1

        for i in range(1,self.A):
            t_old = 0
            iters = 0
            u = Y[:,0].reshape(Y.shape[0],1)
            while iters < max_iter:
                w = X.T @ u / (np.linalg.norm(X.T @ u))
                t = X @ w
                q = Y.T @ t / (t.T @ t)
                u = Y @ q
                t_diff = t - t_old
                t_old = t
                if np.linalg.norm(t_diff) < epsilon:
                    p = X.T @ t / (t.T @ t)
                    X = X - t @ (p.T)
                    t_old = t
                    T = np.hstack((T,t))
                    W = np.hstack((W,w))
                    Q = np.hstack((Q,q))
                    P = np.hstack((P,p))
                    break
                else:
                    iters += 1
        # print(self.A)
        # print(W)
        R = W @ np.linalg.inv((P.T @ W))
                    # 存储训练模型
        return T,W,Q,P,R

    def train(self,X,Y):
        T,W,Q,P,R = self.nipals(X,Y)
        E = X - T @ P.T
        Y_hat = T @ Q.T
        self.A_y=self.pc_number(Y_hat)
        T_y,Q_y = self.pca(Y_hat,self.A_y)

        X_hat = T @ P.T
        P_y = (np.linalg.inv(T_y.T @ T_y) @ T_y.T @ X_hat).T
        X_o_hat = X_hat - T_y @ P_y.T

        T_o,P_o = self.pca(X_o_hat,self.A - self.A_y)
        self.A_r=self.pc_number(E)
        T_r,P_r = self.pca(E,self.A_r)

        #控制限计算
        n=X.shape[0]
        T_y_lim = self.A_y * (n ** 2 - 1) / (n * (n - self.A_y))* scipy.stats.f.ppf(self.signifi, self.A_y, n-self.A_y)
        T_o_lim = (self.A - self.A_y) * (n ** 2 - 1) / (n * (n - (self.A - self.A_y)))* scipy.stats.f.ppf(self.signifi, (self.A - self.A_y), n-(self.A - self.A_y))
        T_r_lim = self.A_r * (n ** 2 - 1) / (n * (n - self.A_r))* scipy.stats.f.ppf(self.signifi, self.A_r, n-self.A_r)
        Qr_normal = []
        for i in range(X.shape[0]):
            Qr_normal.append(E[i,:].T @ E[i,:])
        S1=np.var(Qr_normal);
        mio1=np.mean(Qr_normal);
        V1=2*mio1**2/S1;
        Q_r_lim=S1/(2*mio1)*scipy.stats.chi2.ppf(self.signifi,V1);
        self.model = {
        'P': P,
        'R': R,
        'Q': Q,
        'T_y': T_y,
        'Q_y': Q_y,
        'P_y': P_y,
        'T_o': T_o,
        'P_o': P_o,
        'T_r': T_r,
        'P_r': P_r,
        'T_y_lim': T_y_lim,
        'T_o_lim': T_o_lim,
        'T_r_lim': T_r_lim,
        'Q_r_lim': Q_r_lim
        }
        return self.model

    def test(self,model,X_test):
        T_y_values = []
        T_o_values = []
        T_r_values = []
        Q_r_values = []

        for i in range(X_test.shape[0]):
            x = X_test[i,:]
            t = model['R'].T @ x
            x_res = x - model['P']@ t

            t_y = model['Q_y'].T @model['Q'] @ model['R'].T @ x
            t_o = model['P_o'].T @ (model['P'] - model['P_y'] @ model['Q_y'].T @model['Q']) @ model['R'].T @ x
            t_r = model['P_r'].T @ x_res
            x_res_r = x_res - model['P_r'] @ t_r

            T_y_value = t_y.T @ np.linalg.inv((model['T_y'].T @ model['T_y']) / (model['T_y'].shape[0] - 1))@ t_y
            T_o_value = t_o.T @ np.linalg.inv((model['T_o'].T @ model['T_o']) / (model['T_o'].shape[0] - 1)) @ t_o
            T_r_value = t_r.T @ np.linalg.inv((model['T_r'].T @ model['T_r']) / (model['T_r'].shape[0] - 1)) @ t_r
            Q_r_value = x_res_r.T @ x_res_r

            T_y_values.append(T_y_value)
            T_o_values.append(T_o_value)
            T_r_values.append(T_r_value)
            Q_r_values.append(Q_r_value)
        # return (T_y_values,T_o_values,T_r_values,Q_r_values)
        testresult = {
        'T_y_values': T_y_values,
        'T_o_values': T_o_values,
        'T_r_values': T_r_values,
        'Q_r_values': Q_r_values
        }
        return testresult


    def visualization(self, model, testresult):
        mpl.rcParams['font.sans-serif'] = ['SimHei'] #避免中文乱码
        T_y_lim = model['T_y_lim'] * np.ones((len(testresult['T_y_values']),))
        T_o_lim = model['T_o_lim'] * np.ones((len(testresult['T_y_values']),))
        T_r_lim = model['T_r_lim'] * np.ones((len(testresult['T_y_values']),))
        Q_r_lim = model['Q_r_lim'] * np.ones((len(testresult['T_y_values']),))

        ax1 = plt.subplot(2,2,1)
        ax2 = plt.subplot(2,2,2)
        ax3 = plt.subplot(2,2,3)
        ax4 = plt.subplot(2,2,4)

        ax1.plot(T_y_lim,label='T_y控制限')
        ax1.plot(testresult['T_y_values'],label='T_y')
        # ax1.set_title('SPE统计量')
        ax1.legend(loc="best")
        ax2.plot(T_o_lim,label='T_o控制限')
        ax2.plot(testresult['T_o_values'],label='T_o')
        # ax2.set_title('T2统计量')
        ax2.legend(loc="best")
        ax3.plot(T_r_lim,label='T_r控制限')
        ax3.plot(testresult['T_r_values'],label='T_r')
        # ax2.set_title('T2统计量')
        ax3.legend(loc="best")
        ax4.plot(Q_r_lim,label='Q_r控制限')
        ax4.plot(testresult['Q_r_values'],label='Q_r')
        # ax2.set_title('T2统计量')
        ax4.legend(loc="best")
        plt.show()
