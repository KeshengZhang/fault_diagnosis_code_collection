function [Cont_mrbc]=recon(x_test,PHI,phi_lim)
    %% Iteration for faultion directions at all times 
    for time=1:size(x_test,1)  
            n_test=size(x_test,1);
            I=eye(size(x_test,2));
            mrbc=[];xita=[]; % faultion direction_home
            l=0; maxrbc=0;
            x_testf=x_test(time,:);% faultion sample
        while maxrbc<(x_testf*PHI*x_testf'-phi_lim) % 寻找故障样本
            temp=xita;
            for ii=1:(size(x_testf,2)-l)
                f=I(:,ii);
                xita=[temp f];% 故障方向矩阵
                %重构贡献值             
                mrbc(ii)=x_testf*PHI*xita*pinv(xita'*PHI*xita)*xita'*PHI*x_testf';
            end
            %寻找最大的方向并记录下         
            for jj=1:(size(x_testf,2)-1)
                if mrbc(jj)>maxrbc
                   maxrbc=mrbc(jj);
                   maxdir=jj;
                end
            end
            xita=[temp I(:,maxdir)];
            % 换下一个变量       
            I=[I(:,1:maxdir-1) I(:,maxdir+1:size(x_testf,2)-l)];
            l=l+1;
        end
        %% Contribution Of Variables 
        Cont_mrbc(1:size(x_test,2),time)=zeros(size(x_test,2),1);
        I_mrbc=eye(l);
        for rbc_dir=1:l
            order = find(xita(:,rbc_dir)~=0); % true varible location
            %order变量的位置和time样本位置         
            Cont_mrbc(order,time)=(I_mrbc(:,rbc_dir)'*(( pinv(xita'*PHI*xita ))^0.5)*xita'*PHI*x_testf')^2;% 此处Xe要从1重新迭代跑一下各个时刻的贡献，结果存为一矩阵，然后 imagesc
        end
    end
    Cont_mrbc=abs(Cont_mrbc);%取绝对值，可以不用
    %scale
    Cont_mrbc=Cont_mrbc./max(max(Cont_mrbc));
    figure
    sam=[1,n_test];
    dir=[1,size(x_test,2)];
    % clims = [0 max(Cont_mrbc(:))];
    clims=[0,1];
    imagesc(sam,dir,Cont_mrbc,clims)
    colorbar
    ylabel('variable index');xlabel('samples number');
    title('MRBC for φ')
end