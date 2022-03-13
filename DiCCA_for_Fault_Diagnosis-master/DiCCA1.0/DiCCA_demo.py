import numpy as np
from scipy.io import loadmat
import scipy.stats
import matplotlib.pyplot as plt


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

def pc_number(X):
    U, S, V = np.linalg.svd(X)
    if S.shape[0] == 1:
        i = 1
    else:
        i = 0
        var = 0
        while var < 0.85*sum(S*S):
            var = var+S[i]*S[i]
            i = i + 1
    return i

def DiCCA(X, s, a):
    n = X.shape[0]
    m = X.shape[1]
    N = n - s
    alpha = 0.01
    level = 1-alpha
    P = np.zeros((m, a))
    W = np.zeros((m, a))
    T = np.zeros((n, a))
    Beta = np.zeros((s, a))
    w = np.zeros(m)
    w[0]=1
    Xe = X[s:N+s, :]
    if s > 0:
        l = 0
        while l < a:
            iterr = 1000
            temp = np.dot(X, w)
            temp = temp / np.linalg.norm(temp, ord=2)
            while iterr > 0.00001:
                t = np.dot(X, w)
#                 print(t.shape)
                t = t / np.linalg.norm(t, ord=2)
                Ts = np.zeros((N, s));
#                 t = np.array([t]).T
                for i in range(s):
                    Ts[:,i]= t[i:(N+i)]
#                 t = np.array([t]).T
#                 print(np.dot(Ts.T,t[s:(N+s)]))
#                 print((Ts.T)@t[s:(N+s)])
                beta = np.linalg.pinv((Ts.T@Ts))@np.dot(Ts.T,t[s:(N+s)]);
#                 print((Ts.T@Ts)*np.linalg.inv(Ts.T@t[s:(N+s),:]))
                X_hat = np.zeros([N,m]);
                t_hat = np.zeros([N]);
#                 T_hat = np.zeros([N,l]);
#                 print(beta)
                for i in range(s):
                    X_hat= X_hat + beta[i]*X[s-i-1:(N+s-i-1), :]
                for i in range(s):
                    t_hat= t_hat + beta[i]*t[s-i-1:(N+s-i-1)]
#                 print((Xe.T@Xe+X_hat.T@X_hat).shape,(np.dot(Xe.T,t_hat)+(np.dot(X_hat.T,t[s:N+s]))).shape)
                w = np.linalg.pinv(Xe.T@Xe+X_hat.T@X_hat)@(np.dot(Xe.T,t_hat)+(np.dot(X_hat.T,t[s:N+s])))
#                 print(w)
                t = np.dot(X, w)
                t = t / np.linalg.norm(t, ord=2)
                iterr = np.linalg.norm((t-temp), ord=2)
#                 print(iterr)
                temp = t
            p= np.dot(X.T, t)/np.dot(t.T, t)
#             print(t)
#             print(t.shape,p.shape)
            t = np.array([t]).T
            p = np.array([p]).T

            X = X - np.dot(t, p.T)
            t = t* np.linalg.norm(p, ord=2)
            w = w* np.linalg.norm(p, ord=2)
            p = p/ np.linalg.norm(p, ord=2)
            t_hat=t_hat / np.linalg.norm(t_hat, ord=2)

            P[:, l] = p[:, 0]
            W[:, l] = w
            T[:, l] = t[:, 0]
            Beta[:, l]=beta
            l = l+1

        # Dynamic Inner Modeling
        V = T[s:(N+s), :]
        for i in range(a):
            TTs=np.zeros((N, s))
            for j in range(s):
                TTs[:, j] = T[j:(N+j), i]

#             print(Xe.shape,TTs.shape,Beta[:,i].shape,(P[:,i].reshape(1,P.shape[0])).shape,(TTs@Beta[:,i]).shape)
            Xe=Xe-TTs@(Beta[:,i].reshape(-1,1))@(P[:,i].reshape(1,P.shape[0]))
            V[:,i]=V[:,i]-TTs@Beta[:,i]
        av = pc_number(V)
        _, Sv, Pv = np.linalg.svd(V)
        Pv = Pv.T
        Pv = Pv[:, 0:av]
        lambda_v = 1/(N-1)*np.diag(Sv[0:av]**2)
        if av!=a:
            gv = 1/(N-1)*sum(Sv[av:a]**4)/sum(Sv[av:a]**2)
            hv = (sum(Sv[av:a]**2)**2)/sum(Sv[av:a]**4)
            Tv2_lim = av * (N ** 2 - 1) / (N * (N - av))* scipy.stats.f.ppf(level, av, N-av)
            Qv_lim = gv*scipy.stats.chi2.ppf(level, hv)
            PHI_v = np.dot(np.dot(Pv, np.linalg.inv(lambda_v)), Pv.T)/Tv2_lim + (np.identity(len(Pv@Pv.T))-Pv@Pv.T)/Qv_lim;
            SS_v=1/(N-1)*V.T@V
            g_phi_v=np.trace((SS_v@PHI_v)@(SS_v@PHI_v))/(np.trace(SS_v@PHI_v))
            h_phi_v=(np.trace(SS_v@SS_v)**2)/np.trace((SS_v@PHI_v)@(SS_v@PHI_v))
            phi_v_lim = g_phi_v*scipy.stats.chi2.ppf(level, h_phi_v)
        else:
            Tv2_lim = av * (N ** 2 - 1) / (N * (N - av))* scipy.stats.f.ppf(level, av, N-av)
            PHI_v = np.dot(np.dot(Pv, np.linalg.inv(lambda_v)), Pv.T)
            phi_v_lim=Tv2_lim

    a_s = pc_number(Xe)
    _, Ss, Ps = np.linalg.svd(Xe)
    Ps = Ps.T
    Ps = Ps[:,0:a_s]
    Ts = np.dot(Xe, Ps)
    lambda_s = 1 / (N - 1) * np.diag(Ss[0:a_s] ** 2)
    m = Ss.shape[0]
    gs = 1 / (N - 1) * sum(Ss[a_s:m] ** 4) / sum(Ss[a_s:m] ** 2)
    hs = (sum(Ss[a_s:m] ** 2) ** 2) / sum(Ss[a_s:m] ** 4)
    Ts2_lim = a_s * (N ** 2 - 1) / (N * (N - a_s))* scipy.stats.f.ppf(level, a_s, N-a_s)

    Qs_lim = gs*scipy.stats.chi2.ppf(level, hs)
    PHI_s = Ps@np.linalg.pinv(lambda_s)@Ps.T/Ts2_lim + (np.identity(len(Ps@Ps.T))-Ps@Ps.T)/Qs_lim;

    SS_s=1/(N-1)*Xe.T@Xe
    g_phi_s=np.trace((SS_s@PHI_s)@(SS_s@PHI_s))/(np.trace(SS_s@PHI_s))

    h_phi_s=(np.trace(SS_s@PHI_s)**2)/np.trace((SS_s@PHI_s)@(SS_s@PHI_s))

    phi_s_lim = g_phi_s*scipy.stats.chi2.ppf(level, h_phi_s)
    return P,W,Beta,Ps,lambda_s,Ts2_lim,Qs_lim,phi_v_lim,PHI_v


def DiCCA_test(X,P,W,Theta,Ps,s,lambda_s,PHI_v):
    n = X.shape[0]
    N = n - s
    a = P.shape[1]
    Mst = np.dot(np.dot(Ps, np.linalg.inv(lambda_s)), Ps.T)
    Msq = np.eye((Mst.shape[0])) - np.dot(Ps, Ps.T)
    R = np.dot(W, np.linalg.inv(np.dot(P.T, W)))
    if s > 0:
        T = np.dot(X, R)
        TTs = np.zeros((N, s))
        Xe=X[s:(N+s),:]
        V=T[s:(N+s),:]
        for i in range(a):
            TTs=np.zeros((N, s))
            for j in range(s):
                TTs[:, j] = T[j:(N+j), i]
            V[:,i]=V[:,i]-TTs@Beta[:,i]
            Xe=Xe-TTs@(Beta[:,i].reshape(-1,1))@(P[:,i].reshape(1,P.shape[0]))
    phi_v_index = np.zeros(N)
    Ts_index = np.zeros(N)
    Qs_index = np.zeros(N)
    k = s
    while k < s+N:
        if s > 0:
            temp = V[k-s, :]
            temp = np.array([temp])
            v = temp.T
            phi_v_index[k-s] = np.dot(np.dot(v.T, PHI_v), v)
            e = Xe[k-s, :].T
        else:
            e = Xe[k-s, :].T
        Ts_index[k-s] = np.dot(np.dot(e.T, Mst), e)
        Qs_index[k-s] = np.dot(np.dot(e.T, Msq), e)
        k = k+1
    return phi_v_index,Ts_index,Qs_index

if __name__ == '__main__':
    """"load 数据"""
    x_train= loadmat("./data/d00.mat")['d00']
    x_test = loadmat("./data/d05te.mat")['d05te']
    a = 4# latent number
    s = 2# lag number
    X,X_mean,X_s = autos(x_train)
    x_test = autos_test(x_test,X_mean,X_s)
    P,W,Beta,Ps,lambda_s,Ts2_lim,Qs_lim,phi_v_lim, PHI_v= DiCCA(X, s, a);
    phi_v_index,Ts_index,Qs_index = DiCCA_test(x_test, P, W, Beta, Ps, s, lambda_s, PHI_v);
    # 故障监测结果可视化
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(phi_v_index)
    plt.plot(phi_v_lim*np.ones(len(phi_v_index)),'r--')
    plt.subplot(3,1,2)
    plt.plot(Ts_index)
    plt.plot(Ts2_lim*np.ones(len(phi_v_index)),'r--')
    plt.subplot(3,1,3)
    plt.plot(Qs_index)
    plt.plot(Qs_lim*np.ones(len(phi_v_index)),'r--')
    plt.show()
