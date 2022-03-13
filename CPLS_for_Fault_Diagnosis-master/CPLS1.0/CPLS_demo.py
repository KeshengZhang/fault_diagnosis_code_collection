from CPLS_FaultDection import CPLS_FaultDection
from scipy.io import loadmat
import numpy as np
path_train = './data/d00.mat'
path_test= './data/d05_te.mat'

data1 = loadmat(path_train)['d00']
X1 = data1[:,:22]
X2 = data1[:,-11:]
X_train= np.hstack((X1,X2))
Y_train = data1[:,34:36]
data2 = loadmat(path_test)['d05_te']
X11 = data2[:,:22]
X22 = data2[:,-11:]
X_test = np.hstack((X11,X22))
Y_test  = data2[:,34:36]



#初始化
model = CPLS_FaultDection(A=5,level=0.99)# 主要调节参数A:pls主元数

#数据标准化（若是标准化过后的数据则无需这一步）
[X_Train,X_test] = model.normalize(X_train,X_test)
[Y_Train,Y_test] = model.normalize(Y_train,Y_test)


#训练模型
model.train(X_Train,Y_Train)
"""
{
        'R': R,
        'P': P,
        'Q': Q,
        'Rc': Rc,
        'yta_c': yta_c,
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
"""

#测试模型
testresult = model.test(X_test,Y_test)

#过程监测可视化
model.visualization(testresult)

    
fault_no = 200
### 单样本贡献图可视化
con_result = model.single_sample_con(X_test[fault_no])
model.con_visualization(con_result)

### 单样本重构贡献图可视化
recon_result=model.single_sample_recon(X_test[fault_no])
model.recon_visualization(recon_result)
    
### 多样本贡献图热力图可视化
multi_con_result=model.multi_sample_con(X_test[160:200,:])
model.con_vis_headmap(multi_con_result)

### 多样本重构贡献图热力图可视化
multi_recon_result=model.multi_sample_recon(X_test[160:200,:])
model.recon_vis_headmap(multi_recon_result)
