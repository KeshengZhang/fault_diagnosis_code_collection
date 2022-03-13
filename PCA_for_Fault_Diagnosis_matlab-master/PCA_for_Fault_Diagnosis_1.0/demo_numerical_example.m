clear
clc
close all
%% ����ѵ������
num_sample=100;
a=10*randn(num_sample,1);
x1=a+randn(num_sample,1);
x2=1*sin(a)+randn(num_sample,1);
x3=5*cos(5*a)+randn(num_sample,1);
x4=0.8*x2+0.1*x3+randn(num_sample,1);
Xtrain=[x1,x2,x3,x4];
%% ������������
a=10*randn(num_sample,1);
x1=a+randn(num_sample,1);
x2=1*sin(a)+randn(num_sample,1);
x3=5*cos(5*a)+randn(num_sample,1);
x4=0.8*x2+0.1*x3+randn(num_sample,1);
Xtest=[x1,x2,x3,x4];
Xtest(51:100,2)=Xtest(51:100,2)+15*ones(50,1);

%��׼������
X_mean = mean(Xtrain);  %������Xtrainƽ��ֵ                           
X_std = std(Xtrain);    %���׼�� 
[X_row,X_col] = size(Xtrain); %��Xtrain�С�����               
Xtrain=(Xtrain-repmat(X_mean,X_row,1))./repmat(X_std,X_row,1);

%��Э�������
sigmaXtrain = cov(Xtrain);
%��Э���������������ֽ⣬lamdaΪ����ֵ���ɵĶԽ���T����Ϊ��λ��������������lamda�е�����ֵһһ��Ӧ��
[T,lamda] = eig(sigmaXtrain);                            
disp('����������С����');
disp(lamda);
% disp('����������');
% disp(T);                                            

%ȡ�Խ�Ԫ��(���Ϊһ������)����lamdaֵ�������·�תʹ��Ӵ�С���У���Ԫ������ֵΪ1�����ۼƹ�����С��90%��������Ԫ����
D = flipud(diag(lamda));                                                                  
num_pc=Scree_Test(D,0.85);

%ȡ��lamda���Ӧ����������
P = T(:,X_col-num_pc+1:X_col);                            
TT=Xtrain*T;
TT1=Xtrain*P;
%�����Ŷ�Ϊ99%��95%ʱ��T2ͳ�ƿ�����                       
T2UCL1=num_pc*(X_row-1)*(X_row+1)*finv(0.95,num_pc,X_row - num_pc)/(X_row*(X_row - num_pc));

%���Ŷ�Ϊ95%��Qͳ�ƿ�����
for i = 1:3
    theta(i) = sum((D(num_pc+1:X_col)).^i);
end
h0 = 1 - 2*theta(1)*theta(3)/(3*theta(2)^2);
ca = norminv(0.95,0,1);
QUCL = theta(1)*(h0*ca*sqrt(2*theta(2))/theta(1) + 1 + theta(2)*h0*(h0 - 1)/theta(1)^2)^(1/h0);                           

%% ���߼�⣺
%��׼������
n = size(Xtest,1);
Xtest=(Xtest-repmat(X_mean,n,1))./repmat(X_std,n,1);

%��T2ͳ������Qͳ����
[r,y] = size(P*P');
I = eye(r,y);

T2 = zeros(n,1);
Q = zeros(n,1);
for i = 1:n
    T2(i)=Xtest(i,:)*P*pinv(lamda(X_col-num_pc+1:X_col,X_col-num_pc+1:X_col))*P'*Xtest(i,:)';  
    Q(i) = Xtest(i,:)*(I - P*P')*(I - P*P')'*Xtest(i,:)';                                                                                    
end

%% �������ݼ�ؽ�����ӻ�
figure
set(gcf,'color','w')
subplot(2,1,1);
plot(1:n,T2,'k');                                    
title('��Ԫ����ͳ�����仯ͼ');
xlabel('������');
ylabel('T^2');
hold on;
line([0,n],[T2UCL1,T2UCL1],'LineStyle','--','Color','r');

subplot(2,1,2);
plot(1:n,Q,'k');
xlabel('������');
ylabel('SPE');
hold on;
line([0,n],[QUCL,QUCL],'LineStyle','--','Color','r');

%% ��Ԫ��ͼ
pcaComponent_plot(Xtest,P,D,2)

fault_sample_number=3;
%% ����ͼ
%1.ȷ�����ʧ��״̬�ĵ÷�
S = Xtest(fault_sample_number,:)*P(:,1:num_pc);
r = [ ];
for i = 1:num_pc
    if S(i)^2/lamda(i) > T2UCL1/num_pc
        r = cat(2,r,i);
    end
end

%2.����ÿ���������������ʧ�ص÷ֵĹ���
cont = zeros(length(r),X_col);
for i = length(r)
    for j = 1:X_col
        cont(i,j) = abs(S(i)/D(i)*P(j,i)*Xtest(fault_sample_number,j));
    end
end

%3.����ÿ���������ܹ���
CONTJ = zeros(X_col,1);
for j = 1:X_col
    CONTJ(j) = sum(cont(:,j));
end

%4.����ÿ��������Q�Ĺ���
e = Xtest(fault_sample_number,:)*(I - P*P');
contq = e.^2;

%5. ���ƹ���ͼ
figure;
contribution_Title=sprintf('������#%d�Ĺ���ͼ',fault_sample_number);
set(gcf,'color','w')
subplot(2,1,1);
bar(CONTJ,'k');
xlabel('������');
ylabel('T^2������ %');
title(contribution_Title)
subplot(2,1,2);
bar(contq,'k');
xlabel('������');
ylabel('Q������ %');
hold on 
for i = 1:length(CONTJ)
    h=bar(i,CONTJ(i));
    if CONTJ(i) < 16
        set(h,'FaceColor','b');
    elseif CONTJ(i) < 80
        set(h,'FaceColor','y');
    else
        set(h,'FaceColor','b');
    end
end
hold off
%��Ԫ����ͼ
Contribution_plot(Xtest(fault_sample_number,:),P,D,2)

%���������
alpha=0.9;
S=lamda(X_col-num_pc+1:X_col,X_col-num_pc+1:X_col);
FAI=P*pinv(S)*P'/T2UCL1+(eye(X_col)-P*P')/QUCL;
S=cov(Xtrain);
g=trace((S*FAI)^2)/trace(S*FAI);
h=(trace(S*FAI))^2/trace((S*FAI)^2);
kesi =g*chi2inv(alpha,h);
 %% �ۺ�ָ��
figure;
set(gcf,'color','w')
fai=(Q/QUCL)+(T2/T2UCL1);
plot(fai)
hold on;
line([0,n],[kesi,kesi],'LineStyle','--','Color','r');
title('�ۺ�ָ��');
xlabel('������')

%% ���ӻ�
% ԭʼ�÷־�����ӻ�
figure;
set(gcf,'color','w')
subplot(2,3,1)
plot(TT(:,1),TT(:,2),'*');
xlabel('PC1');
ylabel('PC2');
title('SCORE PLOT FOR PC1 AND PC2');
subplot(2,3,2)
plot(TT(:,1),TT(:,3),'*');
xlabel('PC1');
ylabel('PC3');
title('SCORE PLOT FOR PC1 AND PC3');
subplot(2,3,3)
plot(TT(:,1),TT(:,4),'*');
xlabel('PC1');
ylabel('PC4');
title('SCORE PLOT FOR PC1 AND PC4');
subplot(2,3,4)
plot(TT(:,2),TT(:,3),'*');
xlabel('PC2');
ylabel('PC3');
title('SCORE PLOT FOR PC2 AND PC3');
subplot(2,3,5)
plot(TT(:,2),TT(:,4),'*');
xlabel('PC2');
ylabel('PC4');
title('SCORE PLOT FOR PC2 AND PC4');
subplot(2,3,6)
plot(TT(:,3),TT(:,4),'*');
xlabel('PC3');
ylabel('PC4');
title('SCORE PLOT FOR PC3 AND PC4');

%% ��Ԫ�÷ֿ��ӻ�
figure;
set(gcf,'color','w')
subplot(2,2,1)
plot(TT1(:,1),TT1(:,2),'*');
xlabel('PC1');
ylabel('PC2');
title('SCORE PLOT FOR PC1 AND PC2');
subplot(2,2,2)
plot(TT1(:,1),TT1(:,3),'*');
xlabel('PC1');
ylabel('PC3');
title('SCORE PLOT FOR PC1 AND PC3');
subplot(2,2,3)
plot(TT1(:,2),TT1(:,3),'*');
xlabel('PC2');
ylabel('PC3');
title('SCORE PLOT FOR PC2 AND PC3');
subplot(2,2,4)
plot3(TT1(:,1),TT1(:,2),TT1(:,3),'*');
xlabel('PC1');
ylabel('PC2');
zlabel('PC3');
grid on;
title('�÷־���');