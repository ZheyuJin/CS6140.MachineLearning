% linear regression GD logic. returns w as weight vector.
function nw = logistic_gd_batch_update(learningRate, w,x,label,iterLimit)
obj_values = ones(1,iterLimit);

for iter = 1 : iterLimit
    nw = w;
    
    [rows,cols] = size(nw);
    
    for w_row =1 : rows
        delta = batch_derivative_mean_sum_sqerr(w,x,label,w_row) * learningRate;
        nw(w_row) = w(w_row) - delta;
    end
    
    obj_values(iter) = obj_mean_sum_sq_err(nw,x,label);
    
    w= nw;
    % end of one iteration.
end

%lets plot.
plot(1:iterLimit, obj_values);

xlabel('iteration number'); ylabel('mean square error OBJ function value');
bestMSE = obj_values(end);
title(sprintf('learning rate = %f bestTrainMSE %f',learningRate,bestMSE));
end

% get the gradient at point w.
function ret = batch_derivative_mean_sum_sqerr(w,x,label,w_index)
theCol = x(:,w_index);
out = vec_sigmoid(x*w) - label;
vec = theCol .* out;
ret = mean(vec)  ;
end

% objective function for logistic regression.
% What I found is that,,, using this objctive funtion will get you no
% where.. it converges tooooo slow so that I gave up using this and turned
% into max likelihood function.
function theMean = obj_mean_sum_sq_err(w,x,lables)
theMean=mean((vec_sigmoid(x * w) -lables) .^2);
end