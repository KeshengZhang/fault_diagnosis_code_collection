"""    Concurrent projection on latent space(cpls) for fault detection and fault diagnosis """

# 导入相关库
import numpy as np
from scipy.io import loadmat
import scipy.stats
import matplotlib.pyplot as plt
from pylab import *
import seaborn as sns



class CPLS_FaultDection():  
    def __init__(self,A, level=0.95,cumper=0.9):
        self.level = level #置信限
        self.model = None
        self.A = A    #主元数量
        self.cumper=cumper#累加方差贡献率
    """
    level: float 置信限
    cumper:float 各主元累计贡献率
    A：int PLS主元数

    """

    def normalize(self, X, Y):
        # 用训练数据的标准差和均值对测试数据标准化
        X_mu = np.mean(X, axis=0).reshape((1, X.shape[1]))
        X_std = np.std(X, axis=0).reshape((1, X.shape[1]))
        Xs = (X-X_mu)/X_std
        mu_array = np.ones((Y.shape[0], 1)) * X_mu
        st_array = np.ones((Y.shape[0], 1)) * X_std
        Ys = (Y - mu_array) / st_array
        return Xs, Ys


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
        R = W @ np.linalg.inv((P.T @ W))
        return T,W,Q,P,R

    def pca(self,X,k):
        sigma = (X.T @ X) / len(X)
        U, S, V = np.linalg.svd(sigma)
        Z = X @ U[:,:k] 
        return Z,U[:,:k]
    
    def train(self,X,Y):
        n=X.shape[0]
        T,W,Q,P,R = self.nipals(X,Y)
        Y_hat = T @ Q.T
        Uc, Dc, Vc =np.linalg.svd(Y_hat)#特别注意python中svd的分解方式
        A_c=np.linalg.matrix_rank(Q)
        Uc=Uc[:,:A_c]
        Vc=Vc[:A_c,:].T
        Dc = np.diag(Dc)
        yta_c=Dc@Dc/(n-1)
        Qc=Vc@Dc.T
        Rc=R@Q.T@Vc@np.linalg.pinv(Dc)
        Rcdag=np.linalg.pinv(Rc.T@Rc)@Rc.T
        Yc_hat=Y-Y_hat
        A_y=self.pc_number(Yc_hat)
        T_y,P_y = self.pca(Yc_hat,A_y)
        Y_hat=Yc_hat-T_y@P_y.T
        Xc_hat=X-Uc@Rcdag
        A_x=self.pc_number(Xc_hat)
        T_x,P_x = self.pca(Xc_hat,A_x)
        yta_x=T_x.T@T_x/(T_x.shape[0]-1)
        yta_x[np.where(yta_x<1e-6)] = 0.0#有些数太小
        X_hat = Xc_hat-T_x@ P_x.T

        #控制限计算     
        T_c_lim = A_c* (n ** 2 - 1) / (n * (n - A_c))* scipy.stats.f.ppf(self.level, A_c, n-A_c) 
        T_x_lim = A_x * (n ** 2 - 1) / (n * (n -A_x))* scipy.stats.f.ppf(self.level, A_x, n-A_x) 
        T_y_lim = A_y * (n ** 2 - 1) / (n * (n -A_y))* scipy.stats.f.ppf(self.level, A_y, n-A_y)

        Qx_normal=[]
        for i in range(X_hat.shape[0]):
            Qx_normal.append(X_hat[i,:].T @ X_hat[i,:])
        S1=np.var(Qx_normal); 
        mio1=np.mean(Qx_normal);
        V1=2*mio1**2/S1; 
        Q_x_lim=S1/(2*mio1)*scipy.stats.chi2.ppf(self.level,V1);

        Qy_normal=[]    
        for i in range(Y_hat.shape[0]):
            Qy_normal.append(Y_hat[i,:].T @ Y_hat[i,:])
        S1=np.var(Qy_normal); 
        mio1=np.mean(Qy_normal);
        V1=2*mio1**2/S1; 
        Q_y_lim=S1/(2*mio1)*scipy.stats.chi2.ppf(self.level,V1);
    
        self.model = {
        'R': R,
        'P': P,
        'Q': Q,
        'Rc': Rc,
        'yta_c': yta_c,
        'yta_x': yta_x,
        'Rcdag': Rcdag,
        'Uc_old': Uc,
        'Qc': Qc,
        'T_y_old': T_y,       
        'P_y': P_y,
        'T_x_old': T_x,
        'P_x': P_x,
        'T_c_lim': T_c_lim,
        'T_x_lim': T_x_lim,
        'T_y_lim': T_y_lim,
        'Q_x_lim': Q_x_lim,
         'Q_y_lim': Q_y_lim
        }
    
    def test(self,X_test,Y_test):
        T_c = []
        T_x = []
        T_y = []
        Q_x = []
        Q_y = []


        for i in range(X_test.shape[0]):
            x = X_test[i,:]
            y = Y_test[i,:]
            u_c = self.model['Rc'].T @ x
            xc_hat = x - self.model['Rcdag'].T @ u_c
            yc_hat = y - self.model['Qc'] @ u_c

            t_x = self.model['P_x'].T @ xc_hat
            t_y = self.model['P_y'].T @ yc_hat
            x_hat = xc_hat - self.model['P_x'] @ t_x
            y_hat = yc_hat - self.model['P_y'] @ t_y

            T_c_value = u_c.T @ np.linalg.inv(( self.model['Uc_old'].T @  self.model['Uc_old']) / ( self.model['Uc_old'].shape[0] - 1))@ u_c
            T_x_value = t_x.T @ np.linalg.inv(( self.model['T_x_old'].T @  self.model['T_x_old']) / ( self.model['T_x_old'].shape[0] - 1)) @ t_x
            T_y_value = t_y.T @ np.linalg.inv(( self.model['T_y_old'].T @  self.model['T_y_old']) / ( self.model['T_y_old'].shape[0] - 1)) @ t_y
            Q_x_value = x_hat.T @ x_hat
            Q_y_value = y_hat.T @ y_hat

            T_c.append(T_c_value)
            T_x.append(T_x_value)
            T_y.append(T_y_value)
            Q_x.append(Q_x_value)
            Q_y.append(Q_y_value)
            
            testresult={'T_c': T_c,
                    'T_x': T_x,
                    'T_y': T_y,
                    'Q_x': Q_x,
                    'Q_y': Q_y             
            }
        return testresult

    def visualization(self,testresult):
        mpl.rcParams['font.sans-serif'] = ['SimHei']
        T_c, T_x, T_y, Q_x, Q_y = testresult['T_c'], testresult['T_x'], testresult['T_y'], testresult['Q_x'], testresult['Q_y']
        T_c_lim, T_x_lim, T_y_lim, Q_x_lim, Q_y_lim = self.model['T_c_lim'], self.model['T_x_lim'], self.model['T_y_lim'], self.model['Q_x_lim'], self.model['Q_y_lim']
        plt.figure(figsize=(9.6,6.4),dpi=600)
        ax1 = plt.subplot(2,2,1)
        ax1.axhline(y=T_c_lim,ls="--",color="r",label='$T_c$控制限')
        ax1.plot(T_c,label='$T_c$监测量')
        y = np.array(testresult['T_c'])[((np.where(np.array(testresult['T_c']) > T_c_lim))[0])]
        x = np.where(np.array(testresult['T_c'])>T_c_lim)
        ax1.scatter(x, y, color='red', label='Anomaly',s = 20)
        ax1.legend(loc="best")
        
        ax2 = plt.subplot(2,2,2)
        ax2.axhline(y=T_x_lim,ls="--",color="r",label='$T_x$控制限')
        ax2.plot(T_x,label='$T_x$监测量')
        y = np.array(testresult['T_x'])[((np.where(np.array(testresult['T_x'])>T_x_lim))[0])]
        x = np.where(np.array(testresult['T_x'])>T_x_lim)
        ax2.scatter(x, y, color='red', label='Anomaly',s = 20)
        ax2.legend(loc="best")
        
        ax3 = plt.subplot(2,2,3)
        ax3.axhline(y=T_y_lim,ls="--",color="r",label='$Q_x$控制限')
        ax3.plot(T_y,label='$Q_x$监测量')
        y = np.array(testresult['T_y'])[((np.where(np.array(testresult['T_y'])>T_y_lim))[0])]
        x = np.where(np.array(testresult['T_y'])>T_y_lim)
        ax3.scatter(x, y, color='red', label='Anomaly',s = 20)
        ax3.legend(loc="best")
        plt.show()
        

    def single_sample_con(self,x_test):  #贡献图(reconstruction based contribution plot)
        x_test=x_test.reshape(-1,1)
        m =   x_test.shape[0]
#         Tc_con, Tx_con, Qx_con = [], [], []
        for i in range(3):    
            if i==0:       #Tc
                M1=(self.model['Rc']@(self.model['yta_c']**0.5)@self.model['Rc'].T)
            elif i==1:    #Tx
                M1=(self.model['P_x']@(self.model['yta_x']**0.5)@self.model['P_x'].T)
            else:         #Qx
                M1=np.identity(m)-self.model['P_x']@self.model['P_x'].T 
                
            con=[]
            for j in range(m):
                con.append(np.power(M1[j,:]@x_test,2)[0]);#/M1[j,j]

            if i==0:       #Tc
#                 Tc_con.append(con);
                 Tc_con = con
            elif i==1:    #Tx
#                 Tx_con.append(con);
                Tx_con = con
            else:         #Qx
#                 Qx_con.append(con);
                Qx_con = con
                
        con_result={
        'Tc_con': Tc_con,
        'Tx_con': Tx_con,
        'Qx_con': Qx_con             
        }
        return con_result
    
    def multi_sample_con(self,X_test):#贡献图(reconstruction based contribution plot)
        n=   X_test.shape[0]
        Tc_con = []
        Tx_con = []
        Qx_con = []
#         print(n)
        for i in range(n):
            singlesample_con_result = self.single_sample_con(X_test[i:i+1,:])
            Tc_con.append(singlesample_con_result['Tc_con'])
            Tx_con.append(singlesample_con_result['Tx_con'])
            Qx_con.append(singlesample_con_result['Qx_con'])
            
        multi_sample_con_result={
        'Tc_con': Tc_con,
        'Tx_con': Tx_con,
        'Qx_con': Qx_con             
        }
        return multi_sample_con_result
    
    def con_visualization(self, con_result, fea_names):
        mpl.rcParams['font.sans-serif'] = ['SimHei']
        Tc_con, Tx_con, Qx_con = con_result['Tc_con'], con_result['Tx_con'], con_result['Qx_con']
        plt.figure(figsize=(9.6,6.4),dpi=600)
        ax1 = plt.subplot(3,1,1)
        ax1.bar(x=range(len(Tx_con)),height=Tc_con,width=0.9,label='$T_c$_con变量贡献')
        ax1.legend(loc="best")
        
        ax2= plt.subplot(3,1,2)
        ax2.bar(x=range(len(Tx_con)),height=Tx_con,width=0.9,label='$T_x$_con变量贡献')
        ax2.legend(loc="best")  
        
        ax3= plt.subplot(3,1,3)
        ax3.bar(x=fea_names,height=Qx_con,width=0.9,label='$Q_x$_con变量贡献')
        ax3.tick_params(axis='x', labelsize=10, rotation=-15)    # 设置x轴标签大小
        ax3.legend(loc="best") 
        plt.show()
        
    def con_visualization(self, con_result):
        mpl.rcParams['font.sans-serif'] = ['SimHei']
        Tc_con, Tx_con, Qx_con = con_result['Tc_con'], con_result['Tx_con'], con_result['Qx_con']
        plt.figure(figsize=(9.6,6.4),dpi=600)
        ax1 = plt.subplot(3,1,1)
        ax1.bar(x=range(len(Tx_con)),height=Tc_con,width=0.9,label='$T_c$_con变量贡献')
        ax1.legend(loc="best")
        
        ax2= plt.subplot(3,1,2)
        ax2.bar(x=range(len(Tx_con)),height=Tx_con,width=0.9,label='$T_x$_con变量贡献')
        ax2.legend(loc="best")  
        
        ax3= plt.subplot(3,1,3)
        ax3.bar(x=range(len(Tx_con)),height=Qx_con,width=0.9,label='$Q_x$_con变量贡献')
        ax3.tick_params(axis='x', labelsize=10, rotation=-15)    # 设置x轴标签大小
        ax3.legend(loc="best") 
        plt.show()   
        
    def con_vis_headmap(self, multi_con_result, feaname):
        plt.figure(figsize=(12.8,16),dpi=600)
        ax1 = plt.subplot(3,1,1)
        ax1 = sns.heatmap(np.array(multi_con_result['Tc_con']).T, cmap=sns.color_palette("RdBu_r", 50), yticklabels = feaname)
        ax1.set_xlabel('Samples')
        ax1.set_ylabel('Variables')
        ax1.set_title('$T_c$_con多样本重构')


        ax2 = plt.subplot(3,1,2)
        ax2 = sns.heatmap(np.array(multi_con_result['Tx_con']).T, cmap=sns.color_palette("RdBu_r", 50), yticklabels = feaname)
        ax2.set_xlabel('Samples')
        ax2.set_ylabel('Variables')
        ax1.set_title('$T_x$_con多样本重构')

        ax2 = plt.subplot(3,1,3)
        ax2 = sns.heatmap(np.array(multi_con_result['Qx_con']).T, cmap=sns.color_palette("RdBu_r", 50), yticklabels = feaname)
        ax2.set_xlabel('Samples')
        ax2.set_ylabel('Variables')
        ax1.set_title('$Q_x$_con多样本重构')


    def con_vis_headmap(self, multi_con_result):
        plt.figure(figsize=(12.8,16),dpi=600)
        ax1 = plt.subplot(3,1,1)
        ax1 = sns.heatmap(np.array(multi_con_result['Tc_con']).T, cmap=sns.color_palette("RdBu_r", 50))
        ax1.set_xlabel('Samples')
        ax1.set_ylabel('Variables')
        ax1.set_title('$T_c$_con多样本重构')


        ax2 = plt.subplot(3,1,2)
        ax2 = sns.heatmap(np.array(multi_con_result['Tx_con']).T, cmap=sns.color_palette("RdBu_r", 50))
        ax2.set_xlabel('Samples')
        ax2.set_ylabel('Variables')
        ax1.set_title('$T_x$_con多样本重构')

        ax2 = plt.subplot(3,1,3)
        ax2 = sns.heatmap(np.array(multi_con_result['Qx_con']).T, cmap=sns.color_palette("RdBu_r", 50))
        ax2.set_xlabel('Samples')
        ax2.set_ylabel('Variables')
        ax1.set_title('$Q_x$_con多样本重构')       
        
    def single_sample_recon(self, x_test):#重构贡献图(reconstruction based contribution plot)
        x_test=x_test.reshape(-1,1)
        m = x_test.shape[0]
        for i in range(3):
            if i==1:       #Tc
                M1=self.model['Rc']@(self.model['yta_c'])@self.model['Rc'].T;
            elif i==2:    #Tx
                M1=self.model['P_x']@self.model['yta_x']@self.model['P_x'].T;
            else:         #Qx
                M1=np.identity(m)-self.model['P_x']@self.model['P_x'].T 
                
            Recon = []
            for j in range(m):    
                Recon.append(np.power(M1[j,:]@(x_test),2)[0]/M1[j,j])

            if i==1:       #Tc
                Tc_recon = Recon;
            elif i==2:    #Tx
                Tx_recon = Recon;
            else:         #Qx
                Qx_recon = Recon;
                
        recon_result={
        'Tc_recon': Tc_recon,
        'Tx_recon': Tx_recon,
        'Qx_recon': Qx_recon             
        }
        return recon_result

    def multi_sample_recon(self,X_test):#贡献图(reconstruction based contribution plot)
        n = X_test.shape[0]
        Tc_recon = []
        Tx_recon = []
        Qx_recon = []
        for i in range(n):
            single_sample_recon_result = self.single_sample_recon(X_test[i:i+1,:])
            Tc_recon.append(single_sample_recon_result['Tc_recon'])
            Tx_recon.append(single_sample_recon_result['Tx_recon'])
            Qx_recon.append(single_sample_recon_result['Qx_recon'])
            
        multi_sample_recon_result={
        'Tc_recon': Tc_recon,
        'Tx_recon': Tx_recon,
        'Qx_recon': Qx_recon             
        }
        return multi_sample_recon_result
    
    def recon_vis_headmap(self, multi_sample_recon_result, feaname):
        plt.figure(figsize=(12.8,16),dpi=600)
        ax1 = plt.subplot(3,1,1)
        ax1 = sns.heatmap(np.array(multi_sample_recon_result['Tc_recon']).T, cmap=sns.color_palette("RdBu_r", 50), yticklabels = feaname)
        ax1.set_xlabel('Samples')
        ax1.set_ylabel('Variables')
        ax1.set_title('$T_c$_recon多样本重构')


        ax2 = plt.subplot(3,1,2)
        ax2 = sns.heatmap(np.array(multi_sample_recon_result['Tx_recon']).T, cmap=sns.color_palette("RdBu_r", 50), yticklabels = feaname)
        ax2.set_xlabel('Samples')
        ax2.set_ylabel('Variables')
        ax1.set_title('$T_x$_recon多样本重构')

        ax2 = plt.subplot(3,1,3)
        ax2 = sns.heatmap(np.array(multi_sample_recon_result['Qx_recon']).T, cmap=sns.color_palette("RdBu_r", 50), yticklabels = feaname)
        ax2.set_xlabel('Samples')
        ax2.set_ylabel('Variables')
        ax1.set_title('$Q_x$_recon多样本重构')

    def recon_vis_headmap(self, multi_sample_recon_result):
        plt.figure(figsize=(12.8,16),dpi=600)
        ax1 = plt.subplot(3,1,1)
        ax1 = sns.heatmap(np.array(multi_sample_recon_result['Tc_recon']).T, cmap=sns.color_palette("RdBu_r", 50))
        ax1.set_xlabel('Samples')
        ax1.set_ylabel('Variables')
        ax1.set_title('$T_c$_recon多样本重构')


        ax2 = plt.subplot(3,1,2)
        ax2 = sns.heatmap(np.array(multi_sample_recon_result['Tx_recon']).T, cmap=sns.color_palette("RdBu_r", 50))
        ax2.set_xlabel('Samples')
        ax2.set_ylabel('Variables')
        ax1.set_title('$T_x$_recon多样本重构')

        ax2 = plt.subplot(3,1,3)
        ax2 = sns.heatmap(np.array(multi_sample_recon_result['Qx_recon']).T, cmap=sns.color_palette("RdBu_r", 50))
        ax2.set_xlabel('Samples')
        ax2.set_ylabel('Variables')
        ax1.set_title('$Q_x$_recon多样本重构')    
        
    def recon_visualization(self, recon_result,fea_names):
        mpl.rcParams['font.sans-serif'] = ['SimHei']
        Tc_recon, Tx_recon, Qx_recon = recon_result['Tc_recon'], recon_result['Tx_recon'], recon_result['Qx_recon']
        plt.figure(figsize=(9.6,6.4),dpi=600)
        ax1 = plt.subplot(3,1,1)
        ax1.bar(x=range(len(Tx_recon)),height=Tc_recon,width=0.9,label='$T_c$_recon变量重构贡献')
        ax1.legend(loc="best")
        
        ax2= plt.subplot(3,1,2)
        ax2.bar(x=range(len(Tx_recon)),height=Tx_recon,width=0.9,label='$T_x$_recon变量重构贡献')
        ax2.legend(loc="best")  
        
        ax3= plt.subplot(3,1,3)
        ax3.bar(x=fea_names,height=Qx_recon,width=0.9,label='$Q_x$_recon变量重构贡献')
        ax3.legend(loc="best") 
        ax3.tick_params(axis='x', labelsize=10, rotation=-15)    # 设置x轴标签大小
        plt.show()
        
    def recon_visualization(self, recon_result):
        mpl.rcParams['font.sans-serif'] = ['SimHei']
        Tc_recon, Tx_recon, Qx_recon = recon_result['Tc_recon'], recon_result['Tx_recon'], recon_result['Qx_recon']
        plt.figure(figsize=(9.6,6.4),dpi=600)
        ax1 = plt.subplot(3,1,1)
        ax1.bar(x=range(len(Tx_recon)),height=Tc_recon,width=0.9,label='$T_c$_recon变量重构贡献')
        ax1.legend(loc="best")
        
        ax2= plt.subplot(3,1,2)
        ax2.bar(x=range(len(Tx_recon)),height=Tx_recon,width=0.9,label='$T_x$_recon变量重构贡献')
        ax2.legend(loc="best")  
        
        ax3= plt.subplot(3,1,3)
        ax3.bar(x=range(len(Tx_recon)),height=Qx_recon,width=0.9,label='$Q_x$_recon变量重构贡献')
        ax3.legend(loc="best") 
        ax3.tick_params(axis='x', labelsize=10, rotation=-15)    # 设置x轴标签大小
        plt.show()