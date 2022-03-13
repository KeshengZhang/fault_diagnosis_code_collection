function [KM] = KernelMatrix(TrainMatrix,Sigma)
[Rows,Columns] = size(TrainMatrix); 
KM = zeros(Rows,Rows);
for i=1:Rows
    for j=1:Rows
        s = TrainMatrix(i,:) - TrainMatrix(j,:);
        t = norm(s);
        KM(i,j) = exp(-(t^2)/(Sigma^2));
    end
end
    
    