function [num_pc]=Scree_Test(D,threohold)
n=size(D,1);%��������Ŀ
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
xlabel('��Ԫ��')
ylabel('�ۻ��������')
hold on
line([0,n],[threohold,threohold],'LineStyle','--','Color','r')
figure
plot(D)
xlabel('��Ԫ��')
ylabel('�µ���ʯͼ')
end