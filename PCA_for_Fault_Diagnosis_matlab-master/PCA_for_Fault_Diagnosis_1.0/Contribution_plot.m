%����ͼ
function Contribution_plot(x,P,D,pca_ncomponents)
    t=P'*x';
    [m,~]=size(P);
    con=zeros(pca_ncomponents,m);
    for i=1:pca_ncomponents
        figure
        con(i,:)=(P(:,i)*t(i)/D(i))';
        bar(con(i,:))
        xlabel('������')
        str = sprintf('��Ԫ%d�Ĺ���ͼ',i);
        ylabel(str)
    end
end