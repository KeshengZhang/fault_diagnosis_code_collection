# from numpy import mean,std,zscore,ones,zeros
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from pylab import *

class DiPLS_FaultDection():
    def __init__(self,s,a,signifi=0.95):
        self.s = s
        self.a = a
        self.signifi = signifi
        self.model = None

    def normalize(self, X, Y):
    # 用训练数据的标准差和均值标准化测试数据
        X_mu = np.mean(X, axis=0).reshape((1, X.shape[1]))
        X_std = np.std(X, axis=0).reshape((1, X.shape[1]))
        Xs = (X-X_mu)/X_std
        mu_array = np.ones((Y.shape[0], 1)) * X_mu
        st_array = np.ones((Y.shape[0], 1)) * X_std
        Ys = (Y - mu_array) / st_array
        return Xs, Ys

    def train(self,X,Y):
        s=self.s
        a=self.a
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
                T2_lim = a * (n ** 2 - 1) / (n * (n - a))* scipy.stats.f.ppf(self.signifi, a, n-a)
                #SPE
                residualX = X - T@(P.T)
                SPE1=[]
                for i in range(residualX.shape[0]):
                    SPE1.append(residualX[i]@residualX[i].T)
                S1=np.var(SPE1);
                mio1=np.mean(SPE1);
                V1=2*mio1**2/S1;
                Q_lim=S1/(2*mio1)*scipy.stats.chi2.ppf(self.signifi ,V1);
                model={'P':P,'W':W,'Q':Q,'Beta':Beta,'Alpha':Alpha,'Lambdax_inv':Lambdax_inv,'T2_lim':T2_lim,'Q_lim':Q_lim}
        return model

    def test(self,model,X_test):
        [m,n]=X_test.shape
        R=model['W']@np.linalg.inv(model['P'].T@model['W'])
        T = X_test@R
        residualX=X_test@(np.identity(n)-R@model['P'].T)
        T2=[]
        Q=[]
        for i in range(m):
            Q.append(residualX[i]@(residualX[i].T))
            T2.append(T[i]@model['Lambdax_inv']@T[i])
        testresult={'T2':T2,'Q':Q}
        return testresult

    def predict(self,model,X_test):
        R=model['W']@np.linalg.inv(model['P'].T@model['W'])
        t = X_test@R
        y_predict=np.zeros([X_test.shape[0],1])
        for i in range(self.s+1, X_test.shape[0]):
            y_predict[i]=(np.multiply(model['Alpha'],t[i:i-self.s-1:-1])).sum(axis=0)@model['Q'].T
        y_predict[self.s]=(np.multiply(model['Alpha'],t[self.s::-1])).sum(axis=0)@model['Q'].T
        return(y_predict)

    def visualization(self, model,testresult):
        mpl.rcParams['font.sans-serif'] = ['SimHei'] #避免中文乱码
        T2_lim = model['T2_lim'] * np.ones((len(testresult['T2']),))
        Q_lim = model['Q_lim'] * np.ones((len(testresult['T2']),))

        ax1 = plt.subplot(2,1,1)
        ax2 = plt.subplot(2,1,2)

        ax1.plot(T2_lim,label='T2控制限')
        ax1.plot(testresult['T2'],label='T2')
        # ax1.set_title('SPE统计量')
        ax1.legend(loc="best")
        ax2.plot(Q_lim,label='Q控制限')
        ax2.plot(testresult['Q'],label='Q')
        # ax2.set_title('T2统计量')
        ax2.legend(loc="best")
        plt.show()
