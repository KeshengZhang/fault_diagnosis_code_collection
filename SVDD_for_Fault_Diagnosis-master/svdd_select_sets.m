function [s,t,b_exit] = svdd_select_sets(alpha,G,L)
global Cp;
Gmax1 = -Inf;
Gmax1_idx = -1;
Gmax2 = -Inf;
Gmax2_idx = -1;

eps = 1e-5;
for i=1:L
    if (alpha(i)<Cp)
        if (-G(i)>Gmax1+1e-15)
            Gmax1 = -G(i);
            Gmax1_idx = i;
        end
    end
    if (alpha(i)>0)
        if (G(i)>(Gmax2+1e-15))
            Gmax2 = G(i);
            Gmax2_idx = i;
        end
    end
end

    s = Gmax1_idx;
    t = Gmax2_idx;



if((Gmax1+Gmax2) < (0.5*eps))
    b_exit = 1;
else
    b_exit = 0;
end

end
