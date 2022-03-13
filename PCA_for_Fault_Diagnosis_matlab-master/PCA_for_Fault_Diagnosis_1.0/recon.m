function [Cont_mrbc]=recon(x_test,PHI,phi_lim)
    %% Iteration for faultion directions at all times 
    for time=1:size(x_test,1)  
            n_test=size(x_test,1);
            I=eye(size(x_test,2));
            mrbc=[];xita=[]; % faultion direction_home
            l=0; maxrbc=0;
            x_testf=x_test(time,:);% faultion sample
        while maxrbc<(x_testf*PHI*x_testf'-phi_lim) % Ѱ�ҹ�������
            temp=xita;
            for ii=1:(size(x_testf,2)-l)
                f=I(:,ii);
                xita=[temp f];% ���Ϸ������
                %�ع�����ֵ             
                mrbc(ii)=x_testf*PHI*xita*pinv(xita'*PHI*xita)*xita'*PHI*x_testf';
            end
            %Ѱ�����ķ��򲢼�¼��         
            for jj=1:(size(x_testf,2)-1)
                if mrbc(jj)>maxrbc
                   maxrbc=mrbc(jj);
                   maxdir=jj;
                end
            end
            xita=[temp I(:,maxdir)];
            % ����һ������       
            I=[I(:,1:maxdir-1) I(:,maxdir+1:size(x_testf,2)-l)];
            l=l+1;
        end
        %% Contribution Of Variables 
        Cont_mrbc(1:size(x_test,2),time)=zeros(size(x_test,2),1);
        I_mrbc=eye(l);
        for rbc_dir=1:l
            order = find(xita(:,rbc_dir)~=0); % true varible location
            %order������λ�ú�time����λ��         
            Cont_mrbc(order,time)=(I_mrbc(:,rbc_dir)'*(( pinv(xita'*PHI*xita ))^0.5)*xita'*PHI*x_testf')^2;% �˴�XeҪ��1���µ�����һ�¸���ʱ�̵Ĺ��ף������Ϊһ����Ȼ�� imagesc
        end
    end
    Cont_mrbc=abs(Cont_mrbc);%ȡ����ֵ�����Բ���
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
    title('MRBC for ��')
end