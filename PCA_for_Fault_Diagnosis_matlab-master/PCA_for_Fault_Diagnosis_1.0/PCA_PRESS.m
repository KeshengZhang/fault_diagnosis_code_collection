function []=PCA_PRESS(xx_train,xx_test)
Xtrain =xx_train;
Xtest =xx_test;
%标准化处理：
X_mean = mean(Xtrain);  %按列求Xtrain平均值                           
X_std = std(Xtrain);    %求标准差                      
[X_row,X_col] = size(Xtrain); %求Xtrain行、列数               
[m,n]=size(Xtest);
Xtrain=(Xtrain-repmat(X_mean,X_row,1))./repmat(X_std,X_row,1);

%求协方差矩阵
sigmaXtrain = cov(Xtrain);
%对协方差矩阵进行特征分解，lamda为特征值构成的对角阵，T的列为单位特征向量，且与lamda中的特征值一一对应：
[T,lamda] = eig(sigmaXtrain);                            
% disp('特征根（由小到大）');
% disp(lamda);
% disp('特征向量：');
% disp(T);                                            

%取对角元素(结果为一列向量)，即lamda值，并上下反转使其从大到小排列，主元个数初值为1，若累计贡献率小于90%则增加主元个数
D = flipud(diag(lamda));                            
num_pc = 1;
PRESS=zeros(52,1);
for num_pc = 1:52
    %取与num_pc相对应的特征向量P
    P = T(:,1:num_pc);
    Xtest_estimate=Xtest*P*P';
    Xcanca=Xtest- Xtest_estimate;
    PRESS(num_pc)=sum(norm(Xcanca,2))/(m*n);
end 
figure
plot(PRESS)
