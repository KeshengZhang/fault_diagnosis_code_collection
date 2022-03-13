%贡献图
function Contribution_plot(x,P,D,pca_ncomponents)
    t=P'*x';
    [m,~]=size(P);
    con=zeros(pca_ncomponents,m);
    for i=1:pca_ncomponents
        figure
        con(i,:)=(P(:,i)*t(i)/D(i))';
        bar(con(i,:))
        xlabel('变量号')
        str = sprintf('主元%d的贡献图',i);
        ylabel(str)
    end
end