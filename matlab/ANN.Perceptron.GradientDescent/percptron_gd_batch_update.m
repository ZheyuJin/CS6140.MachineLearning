%percptron_gd_batch_update perceptron GD logic. returns w as weight vector.
function nw = percptron_gd_batch_update(learningRate, w,x,label,iterLimit)
obj_values = ones(1,iterLimit);
mistake_counts = ones(1,iterLimit);
[wRows, wCols] = size(w);

for iter = 1 : iterLimit
    output = x*w;
    [mistakeSet, mcount] = getMistake(output);
    
    %print log.
    disp(sprintf('iter %d; \t total mistakes %d',iter,mcount));
    obj_values(iter) = obj_mistake_sum(output,mistakeSet);
    mistake_counts(iter) =mcount;
    
    [mistakeRows,cols] = size(mistakeSet);
    
    deltaVec = zeros(wRows,1);
    for mRow =1 : mistakeRows
        if(mistakeSet(mRow) ==1)%mistake?!
            deltaVec = deltaVec + x(mRow,:)' * learningRate;            
        end
    end
    w = w + deltaVec;
    % end of one iteration.
end

nw = w; % update into nw
%lets plot.
plot(1:iterLimit,  mistake_counts);

xlabel('iteration number'); ylabel(' mistake count');
best = mistake_counts(end);
title(sprintf('learning rate = %f best mistake count%f',learningRate,best));
end

%calculate mistake set.
% 1 means mistake, 0 means correct.
function [mset, mcount] = getMistake(output)
[labelRow, labelCol] = size(output);
mset = ones(1,labelRow)'; % all initialized as mistake now.

for row = 1:labelRow
    if  output(row) >= 0 % on positive side of hyperplane.
        mset(row) = 0;% correct
    end       
end

% 1 means mistake, 0 means correct.
mcount = sum(mset);
end

% objective function mistake sum; perceprton
% looking for this value to go down.
function value = obj_mistake_sum(output, mistakeSet)
value =0;
[rowsTotal, cols] = size(mistakeSet);

for row = 1:rowsTotal
    if mistakeSet(row) == 1
        value = value - output(row); % sum of negation in mistake set.
    end
end
end
