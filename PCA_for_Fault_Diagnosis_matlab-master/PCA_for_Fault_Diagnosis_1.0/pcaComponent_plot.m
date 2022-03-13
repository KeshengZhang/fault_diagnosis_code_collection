%贡献图
function pcaComponent_plot(X,P,D,plot_components)
% 选取n_components主元数
T=X*P(:,1:plot_components);
for i=1:plot_components
 figure
 plot(T(:,i)/D(i))
 xlabel('样本号')
 str = sprintf('主元%d',i);
%  ylabel('主元%d',i)
 ylabel(str)

end