function [alpha R] = svdd_solve(in_KM)
global Cp;
L = size(in_KM,1);
alpha= zeros(1,L);
alpha(3) = 1;
G = zeros(1,L);

for i=1:L
    if alpha(i)>0
        G(:) = G(:) + alpha(i) * in_KM(:,i);
    end
end

while(1)
    [i j b_exit] = svdd_select_sets(alpha,G,L);
    if b_exit
        break;
    end
    
    old_alpha_i = alpha(i);
    old_alpha_j = alpha(j);
    
    delta = (G(i)-G(j))/max(in_KM(i,i) + in_KM(j,j)-2 * in_KM(i,j),0);
    sum = alpha(i) + alpha(j);
    
    alpha(j) = alpha(j) + delta;
    alpha(i) = alpha(i) - delta;
    
    
    if (alpha(i)<0)
        alpha(i) = 0;
        alpha(j) = sum;
    end
    if (alpha(j)<0)
        alpha(j) = 0;
        alpha(i) = sum;
    end
       %Cp为默认的惩罚参数1
    if (alpha(i)>Cp)
        alpha(i) = Cp;
        alpha(j) = sum - Cp;
    end
    if (alpha(j)>Cp)
        alpha(j) = Cp;
        alpha(i) = sum - Cp;
    end
        
    delta_alpha_i = alpha(i) - old_alpha_i;
    delta_alpha_j = alpha(j) - old_alpha_j;
    
    G(:) = G(:) + in_KM(:,i) * delta_alpha_i + in_KM(:,j) * delta_alpha_j;
end %结束while循环




%待修改一下程序部分
R=0;
for i=1:L
    if (alpha(i)>0)
        for j=1:L
            if (alpha(j)>0)
                R = R + alpha(i) * alpha(j) * in_KM(i,j);
            end
        end
    end
end

all = 0;
times = 0;
for i=1:L
    if ((alpha(i)>0) &&(alpha(i)<Cp))
        tt = 0;
        for j=1:L
            tt = tt - 2 * alpha(j) * in_KM(i,j);
        end
        tt = tt + R + 1;
        all = all + sqrt(tt);
        times = times + 1;
    end
end
R = all / times;
end