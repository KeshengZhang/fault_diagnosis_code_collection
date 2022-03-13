clear
clc
global Cp
Cp = 1; 
sigma = 23;
Npos = 160;

load ('.\data\d00.dat')
load ('.\data\d05_te.dat')
Y = d05_te;
X=d00';
[n4,m4]=size(X);
s = std(X);
mm = mean(X);
X = (X-ones(n4,1)*mm)*diag((s.^(-1)));%���Ļ�
sample_number = size(Y,1);
Y = (Y-ones(sample_number,1)*mm)*diag((s.^(-1)));
TrainMatrix = X;
TestMatrix = Y;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%�˾���%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[Rows,Columns] = size(TrainMatrix); 
KM = zeros(Rows,Rows);
for i=1:Rows
    for j=1:Rows
        s = TrainMatrix(i,:) - TrainMatrix(j,:);
        KM(i,j) = exp(-(norm(s)^2)./(sigma^2));
    end
end
[alpha R] = svdd_solve(KM);
SV = TrainMatrix(alpha>0,:);
out = predict(KM,TestMatrix,SV,alpha,sigma);
SupportVector = find(alpha>0);
plot(out);
hold on
plot(R.*ones(1,size(out,2)),'r')



excessive = find((out - R)>0);
excessive = find(excessive<161);
excessive = size(excessive,2);

weak = find((out - R)<0);
weak = find(weak>160);
weak = size(weak,2); 

fprintf(2,'©�������ٷ�֮ %g \n',weak./8);
fprintf(2,'�鱨�����ٷ�֮ %g \n',excessive./1.6);