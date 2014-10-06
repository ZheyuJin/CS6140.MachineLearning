% linear regression GD logic. returns w as weight vector.
function nw = logistic_ga_batch_update(learningRate, w,x,label,iterLimit)
obj_values = ones(1,iterLimit);

for iter = 1 : iterLimit
    nw = w;
    
    [rows,cols] = size(nw);
    
    for w_row =1 : rows
        delta = batch_derivative_max_likelihood(w,x,label,w_row) * learningRate;
        nw(w_row) = w(w_row) + delta;
    end
    
    obj_values(iter) = obj_log_of_max_likelihood(nw,x,label);
    
    w= nw;
    % end of one iteration.
end

%lets plot.
plot(1:iterLimit, obj_values);

xlabel('iteration number'); ylabel('max likelihood OBJ function value');
bestLikeliHood = obj_values(end);
title(sprintf('learning rate = %f logOfBestLikeliHood %f',learningRate,bestLikeliHood));
end

% get the gradient at point w.
function ret = batch_derivative_max_likelihood(w,x,label,w_index)
theCol = x(:,w_index); %Xj
out =   label - vec_sigmoid(x*w);
vec = theCol .* out; 
ret = sum(vec)  ;
end

% objective function for logistic regression.
% looking for this value to go up.
function value = obj_log_of_max_likelihood(w,x,lables)
h_vec=vec_sigmoid(x * w); % hypo vector for x
value=sum(lables .* log(h_vec) + (1 - lables) .* log(1-h_vec));
end