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

#将测试数据归一化
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
        pcnumber = 1
    else:
        i = 0
        var = 0
        while var < 0.85*sum(S*S):
            var = var+S[i]*S[i]
            i = i + 1
        return i

def DiPLS(X,Y, s, a):
    n = X.shape[0]
    m = X.shape[1]
    N = n - s-1
    #construct augmented matrix
    Ys= Y[s:n, :]
    Xi=[]
    for h in range(s+1):
        Xi.append(X[h:h+N+1,:])
    Zs=Xi[s]
    for z in range(s-1,-1,-1):
        Zs=np.hstack((Zs,Xi[z]))

    P = np.zeros((m, a))
    Q=  np.zeros((Y.shape[1], a))
    W = np.zeros((m, a))
    T = np.zeros((n, a))
    Beta = np.zeros((s+1, a))
    Alpha = np.zeros((s+1, a))
    # Step1
    #       Scale X and Y to zero-mean and unit-variance.******************
    #     Initialize Beta with [1,0,··· ,0]' ,and us as some column of Y s .
    beta = np.zeros([s+1, 1]);
    beta[0]=1;
    us=Ys[:,0];
    us=us.reshape(us.shape[0],1)
    w = np.ones([m,1])
    w = w / np.linalg.norm(w, ord=2)
    if s > 0:
        l = 0
        while l < a:
            iterr = 1000
            temp = np.dot(X, w)
            temp = temp/np.linalg.norm(temp, ord=2)
            while iterr > 0.00001:
                w=np.kron(beta,np.identity(m)).T@Zs.T@us
                w = w/np.linalg.norm(w, ord=2)
                t = np.dot(X, w)
                Ts=t[0:N+1,:]
                for j in range(1,s+1):
                    Ts=np.hstack((t[j:N+j+1,:],Ts))
                q=Ys.T@Zs@np.kron(beta,w)
                us=Ys@q
                beta = Ts.T@us
                beta = beta / np.linalg.norm(beta, ord=2)
                iterr = np.linalg.norm((t-temp), ord=2)
                temp = t
            # Step 3: Inner model building.Build a linear model between
            alpha=np.linalg.inv(Ts.T@Ts)@Ts.T@us;
            us_hat=Ts@alpha;
            # Step 4: Deflation
            p = np.dot(X.T, t)/np.dot(t.T, t)

            q = Ys.T@us_hat@np.linalg.inv(us_hat.T@us_hat);
            Ys = Ys - us_hat@ q.T
            alpha=alpha/np.linalg.norm(p, ord=2)*np.linalg.norm(q, ord=2)
            q=q/np.linalg.norm(q, ord=2)
            t=t* np.linalg.norm(p, ord=2)
            w=w* np.linalg.norm(p, ord=2)
            p=p/np.linalg.norm(p, ord=2)
            P[:, l] = p[:, 0]
            Q[:, l] = q[:, 0]
            W[:, l] = w[:, 0]
            T[:, l] = t[:, 0]
            Beta[:, l] = beta[:, 0]
            Alpha[:, l] = alpha[:, 0]
            l = l+1
            #控制限计算
            #T2
            Lambdax_inv=np.linalg.pinv(T.T@T/(n-1))
            level=0.99
            T2_lim = a * (n ** 2 - 1) / (n * (n - a))* scipy.stats.f.ppf(level, a, n-a)
            #SPE
            residualX = X - T@(P.T)
            SPE1=[]
            for i in range(residualX.shape[0]):
                SPE1.append(residualX[i]@residualX[i].T)
            S1=np.var(SPE1);
            mio1=np.mean(SPE1);
            V1=2*mio1**2/S1;
            Q_lim=S1/(2*mio1)*scipy.stats.chi2.ppf(0.99,V1);
    return {'P':P.tolist(),'W':W.tolist(),'Q':Q.tolist(),'Beta':Beta.tolist(),'Alpha':Alpha.tolist(),'Lambdax_inv':Lambdax_inv.tolist(),'T2_lim':T2_lim,'Q_lim':Q_lim}
#     return P,W,Q,Beta,Alpha,Lambdax_inv,T2_lim,Q_lim

def DiPLS_test(X_test,P,W,lambdax):
    [m,n]=X_test.shape
    R=W@np.linalg.inv(P.T@W)
    T = X_test@R
    residualX=X_test@(np.identity(n)-R@P.T)
    T2=[]
    Q=[]
    for i in range(m):
        Q.append(residualX[i]@(residualX[i].T))
        T2.append(T[i]@lambdax@T[i])
    return {'T2':T2,'Q':Q}

def DiPLS_predict(X_test,P,W,Alpha,Q):
    R=W@np.linalg.inv(P.T@W)
    t = X_test@R
    y_predict=np.zeros([X_test.shape[0],1])
    for i in range(s+1, X_test.shape[0]):
        y_predict[i]=(np.multiply(Alpha,t[i:i-s-1:-1])).sum(axis=0)@Q.T
    y_predict[s]=(np.multiply(Alpha,t[s::-1])).sum(axis=0)@Q.T
    return({'y_predict':y_predict.tolist()})

if __name__ == '__main__':
    """"load 数据"""
    x_train= loadmat("./data/d00.mat")['d00']
    x_test = loadmat("./data/d05te.mat")['d05te']
    # 数据预处理：数据下采样和消除滞后
    # 训练数据
    X=np.c_[x_train[0:-5:5,0:22],x_train[0:-5:5,41:52]]
    Y=x_train[5:-1:5,37]
    X=X.reshape(X.shape[0],-1)
    Y=Y.reshape(Y.shape[0],1)
    # 测试数据
    X_test=np.c_[x_test[0:-5:5,0:22],x_test[0:-5:5,41:52]]
    Y_test=x_test[5:-1:5,37]
    X_test=X_test.reshape(X_test.shape[0],-1)
    Y_test=Y_test.reshape(Y_test.shape[0],1)
    # 数据标准化
    ##训练数据标准化
    X,X_mean,X_s = autos(X)
    Y,Y_mean,Y_s = autos(Y)
    ##测试数据标准化
    X_test = autos_test(X_test,X_mean,X_s)
    Y_test = autos_test(Y_test,Y_mean,Y_s)

    """建模、预测、监控"""
    s=4# lag number
    a=2#  number of latent variables.
    model=DiPLS(X,Y, s, a)#modelling
    predict=DiPLS_predict(X_test,np.array(model['P']),np.array(model['W']),np.array(model['Alpha']),np.array(model['Q']))#predict
    stat=DiPLS_test(X_test,np.array(model['P']),np.array(model['W']),np.array(model['Lambdax_inv']))#monitor

    # 预测可视化
    plt.figure()
    plt.plot(predict['y_predict'])
    plt.plot(Y_test)
    plt.xlabel('Sample number')

    # 诊断可视化
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(stat['T2'])
    plt.axhline(y=model['T2_lim'],ls="--",color="r")
    plt.xlabel('Sample number')
    plt.ylabel('$T^2$')


    plt.subplot(2,1,2)
    plt.plot(stat['Q'])
    plt.axhline(y=model['Q_lim'],ls="--",color="r")
    plt.xlabel('Sample number')
    plt.ylabel('$Q$')
    plt.show()

