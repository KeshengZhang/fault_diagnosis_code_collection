function [num_pc]=Scree_Test(D,threohold)
n=size(D,1);%特征根数目
for i=1:n
    cont(i)=sum(D(1:i))/sum(D);
end
for i=1:n
 if (cont(i)>threohold)
        num_pc=i;
        break
 end
end
figure
plot(cont)
xlabel('主元数')
ylabel('累积方差贡献率')
hold on
line([0,n],[threohold,threohold],'LineStyle','--','Color','r')
figure
plot(D)
xlabel('主元数')
ylabel('崖底碎石图')
end