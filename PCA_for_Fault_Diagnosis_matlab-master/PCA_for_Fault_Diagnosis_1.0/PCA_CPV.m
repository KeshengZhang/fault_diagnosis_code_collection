function []=PCA_CPV(xx_train)
Xtrain =xx_train;
%��׼������
X_mean = mean(Xtrain);  %������Xtrainƽ��ֵ                           
X_std = std(Xtrain);    %���׼��                      
[X_row,~] = size(Xtrain); %��Xtrain�С�����     
Xtrain=(Xtrain-repmat(X_mean,X_row,1))./repmat(X_std,X_row,1);

%��Э�������
sigmaXtrain = cov(Xtrain);
%��Э���������������ֽ⣬lamdaΪ����ֵ���ɵĶԽ���T����Ϊ��λ��������������lamda�е�����ֵһһ��Ӧ��
[~,lamda] = eig(sigmaXtrain);                            
% disp('����������С����');
% disp(lamda);
% disp('����������');
% disp(T);                                            

%ȡ�Խ�Ԫ��(���Ϊһ������)����lamdaֵ�������·�תʹ��Ӵ�С���У���Ԫ������ֵΪ1�����ۼƹ�����С��90%��������Ԫ����
D = flipud(diag(lamda));
cont=size(50,1);
for i=1:50
    cont(i)=sum(D(1:i))/sum(D);
    
end                                                 
figure
plot(cont)
xlabel('��Ԫ��')
ylabel('�ۼƷ������')
hold on
line([1,50],[0.9,0.9],'LineStyle','--','Color','r')
