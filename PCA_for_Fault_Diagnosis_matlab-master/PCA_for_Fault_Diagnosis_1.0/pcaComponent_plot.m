%����ͼ
function pcaComponent_plot(X,P,D,plot_components)
% ѡȡn_components��Ԫ��
T=X*P(:,1:plot_components);
for i=1:plot_components
 figure
 plot(T(:,i)/D(i))
 xlabel('������')
 str = sprintf('��Ԫ%d',i);
%  ylabel('��Ԫ%d',i)
 ylabel(str)

end