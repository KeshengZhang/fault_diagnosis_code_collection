function []=PCA_PRESS(xx_train,xx_test)
Xtrain =xx_train;
Xtest =xx_test;
%��׼������
X_mean = mean(Xtrain);  %������Xtrainƽ��ֵ                           
X_std = std(Xtrain);    %���׼��                      
[X_row,X_col] = size(Xtrain); %��Xtrain�С�����               
[m,n]=size(Xtest);
Xtrain=(Xtrain-repmat(X_mean,X_row,1))./repmat(X_std,X_row,1);

%��Э�������
sigmaXtrain = cov(Xtrain);
%��Э���������������ֽ⣬lamdaΪ����ֵ���ɵĶԽ���T����Ϊ��λ��������������lamda�е�����ֵһһ��Ӧ��
[T,lamda] = eig(sigmaXtrain);                            
% disp('����������С����');
% disp(lamda);
% disp('����������');
% disp(T);                                            

%ȡ�Խ�Ԫ��(���Ϊһ������)����lamdaֵ�������·�תʹ��Ӵ�С���У���Ԫ������ֵΪ1�����ۼƹ�����С��90%��������Ԫ����
D = flipud(diag(lamda));                            
num_pc = 1;
PRESS=zeros(52,1);
for num_pc = 1:52
    %ȡ��num_pc���Ӧ����������P
    P = T(:,1:num_pc);
    Xtest_estimate=Xtest*P*P';
    Xcanca=Xtest- Xtest_estimate;
    PRESS(num_pc)=sum(norm(Xcanca,2))/(m*n);
end 
figure
plot(PRESS)
