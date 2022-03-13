# 导入相关库
import numpy as np
from scipy.io import loadmat
import scipy.stats
import matplotlib.pyplot as plt

#T,W,Q,P,R
# 归一化数据
def autos(X):
    m = X.shape[0]
    n = X.shape[1]
    X_m = np.zeros((m, n))
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0, ddof=1)
    for i in range(n):
        a = np.ones(m) * mu[i]
        X_m[:, i] = (X[:, i]-a) / sigma[i]
    return X_m, mu, sigma

def autos_test(data,m_train,v_train):
    m = data.shape[0]
    n = data.shape[1]
    data_new = np.zeros((m, n))
    for i in range(n):
        a = np.ones(m) * m_train[i]
        data_new[:, i] = (data[:, i] - a) / v_train[i]
    return data_new

def pc_number(X,percent=0.9):
    U, S, V = np.linalg.svd(X)
    if S.shape[0] == 1:
        pcnumber = 1
    else:
        i = 0
        var = 0
        while var < percent*sum(S*S):
            var = var+S[i]*S[i]
            i = i + 1
        return i

def nipals(X, Y,A,max_iter=1000, epsilon=1e-07):
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
            #print(T.shape)

            break
        else:
            iters += 1
    for i in range(1,A):
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
                #print(t.shape)
                T = np.hstack((T,t))
                W = np.hstack((W,w))
                Q = np.hstack((Q,q))
                P = np.hstack((P,p))
                break
            else:
                iters += 1
    R = W @ np.linalg.inv((P.T @ W))
    return T,W,Q,P,R

def pca(X,k):
    sigma = (X.T @ X) / len(X)
    U, S, V = np.linalg.svd(sigma)
    Z = X @ V[:,:k]
    return Z,V[:,:k]

# tpls建模;目前pls的主元数确定还没有实现
def tpls(X,Y,A,percent,level):
    T,W,Q,P,R = nipals(X,Y,A)
    E = X - T @ P.T
    Y_hat = T @ Q.T
    A_y=pc_number(Y_hat,percent)
    T_y,Q_y = pca(Y_hat,A_y)

    X_hat = T @ P.T
    P_y = (np.linalg.inv(T_y.T @ T_y) @ T_y.T @ X_hat).T
    X_o_hat = X_hat - T_y @ P_y.T

    T_o,P_o = pca(X_o_hat,A - A_y)
    A_r=pc_number(E,percent)
    T_r,P_r = pca(E,A_r)

    #控制限计算
    n=X.shape[0]
    T_y_lim = A_y * (n ** 2 - 1) / (n * (n - A_y))* scipy.stats.f.ppf(level, A_y, n-A_y)
    T_o_lim = (A - A_y) * (n ** 2 - 1) / (n * (n - (A - A_y)))* scipy.stats.f.ppf(level, (A - A_y), n-(A - A_y))
    T_r_lim = A_r * (n ** 2 - 1) / (n * (n - A_r))* scipy.stats.f.ppf(level, A_r, n-A_r)
    Qr_normal = []
    for i in range(X.shape[0]):
        Qr_normal.append(E[i,:].T @ E[i,:])
    S1=np.var(Qr_normal);
    mio1=np.mean(Qr_normal);
    V1=2*mio1**2/S1;
    Qr_lim=S1/(2*mio1)*scipy.stats.chi2.ppf(level,V1);
    return R,P,Q,T_y,Q_y,P_y,T_o,P_o,T_r,P_r,T_y_lim,T_o_lim,T_r_lim,Qr_lim

def tpls_test(X_test,R,P,Q,T_y,Q_y,P_y,T_o,P_o,T_r,P_r):
    T_y_values = []
    T_o_values = []
    T_r_values = []
    Q_r_values = []

    for i in range(X_test.shape[0]):
        x = X_test[i,:]
        t = R.T @ x
        x_res = x - P @ t
        t_y = Q_y.T @ Q @ R.T @ x
        t_o = P_o.T @ (P - P_y @ Q_y.T @ Q) @ R.T @ x
        t_r = P_r.T @ x_res
        x_res_r = x_res - P_r @ t_r

        T_y_value = t_y.T @ np.linalg.inv((T_y.T @ T_y) / (T_y.shape[0] - 1))@ t_y
        T_o_value = t_o.T @ np.linalg.inv((T_o.T @ T_o) / (T_o.shape[0] - 1)) @ t_o
        T_r_value = t_r.T @ np.linalg.inv((T_r.T @ T_r) / (T_r.shape[0] - 1)) @ t_r
        Q_r_value = x_res_r.T @ x_res_r

        T_y_values.append(T_y_value)
        T_o_values.append(T_o_value)
        T_r_values.append(T_r_value)
        Q_r_values.append(Q_r_value)
    return (T_y_values,T_o_values,T_r_values,Q_r_values)

if __name__ == '__main__':
    # load data
    path_train = './data/d00_te.mat'
    path_test= './data/d01te.mat'
    data1 = loadmat(path_train)['d00te']
    X1 = data1[:,:22]
    X2 = data1[:,-11:]
    X_Train= np.hstack((X1,X2))
    Y_Train = data1[:,34:36]
    data2 = loadmat(path_test)['d01te']
    X11 = data2[:,:22]
    X22 = data2[:,-11:]
    X_test = np.hstack((X11,X22))
    Y_test  = data2[:,34:36]

    # noramlization
    ##训练数据标准化
    X,X_mean,X_s = autos(X_Train)
    Y,Y_mean,Y_s = autos(Y_Train)
    ##测试数据标准化
    X_test = autos_test(X_test,X_mean,X_s)
    Y_test = autos_test(Y_test,Y_mean,Y_s)

    [R,P,Q,T_y,Q_y,P_y,T_o,P_o,T_r,P_r,T_y_lim,T_o_lim,T_r_lim,Q_r_lim]=tpls(X,Y,A=6,percent=0.94,level=0.99)#建模
    [T_y_values,T_o_values,T_r_values,Q_r_values]=tpls_test(X_test,R,P,Q,T_y,Q_y,P_y,T_o,P_o,T_r,P_r)#测试

    # 绘图
    plt.figure()
    plt.subplot(2,2,1)
    plt.plot(T_y_values)
    plt.xlabel('Sample number')
    plt.ylabel('$T^2_y$')
    plt.axhline(y=T_y_lim,ls="--",color="r")


    plt.subplot(2,2,2)
    plt.plot(T_o_values)
    plt.xlabel('Sample number')
    plt.ylabel('$T^2_r$')
    plt.axhline(y=T_o_lim,ls="--",color="r")


    plt.subplot(2,2,3)
    plt.plot(T_r_values)
    plt.xlabel('Sample number')
    plt.ylabel('$Q_r$')
    plt.axhline(y=T_r_lim,ls="--",color="r")


    plt.subplot(2,2,4)
    plt.plot(Q_r_values)
    plt.xlabel('Sample number')
    plt.ylabel('$T^2_y$')
    plt.axhline(y=Q_r_lim,ls="--",color="r")
    plt.show()
