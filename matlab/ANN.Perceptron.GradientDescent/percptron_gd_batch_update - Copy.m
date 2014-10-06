%percptron_gd_batch_update perceptron GD logic. returns w as weight vector.
function nw = percptron_gd_batch_update(learningRate, w,x,label,iterLimit)
obj_values = ones(1,iterLimit);

for iter = 1 : iterLimit        
    output = x*w;
    [mistakeSet, mcount] = getMistake(label,output);
        
    %print log.
    disp(sprintf('iter %d; \t total mistakes %d\n',iter,mcount));
    obj_values(iter) = obj_mistake_sum(output,mistakeSet);   
    
    [mistakeRows,cols] = size(mistakeSet);        
    for mRow =1 : mistakeRows
        if(mistakeSet(mRow) ==1)%mistake?!
            delta = x(mRow)' * learningRate;
            w(mRow) = w(mRow) + delta;
        end
    end
        
    % end of one iteration.
end

nw = w; % update into nw
%lets plot.
plot(1:iterLimit, obj_values);

xlabel('iteration number'); ylabel('sum of mistake values');
bestOBJ = obj_values(end);
title(sprintf('learning rate = %f best OBJ value %f',learningRate,bestOBJ));
end

%calculate mistake set.
% 1 means mistake, 0 means correct.
function mset, mcount = getMistake(label,output)
[labelRow, labelCol] = size(label);
mset = ones(1,labelRow);
for row = 1:labelRow
    if output(row) >= 0 && label == 1
        mset(row) = 0;% correct
    end
    
    if output(row) < 0 && label == -1
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


%{
% get the gradient at point w.
function ret = batch_derivative_min_mistake_sum(w,x,label,w_index)
theCol = x(:,w_index); %Xj
out =   label - vec_sigmoid(x*w);
vec = theCol .* out;
ret = sum(vec)  ;
end
}%