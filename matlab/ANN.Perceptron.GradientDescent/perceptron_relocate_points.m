function [ x, label] = perceptron_relocate_points( x, label )
[rowCount, colCount] = size(label);

for row = 1: rowCount
if(label(row) == -1)    % relocate
    label(row) =1;
    x(row,:) = -x(row,:);    
end
end
end

