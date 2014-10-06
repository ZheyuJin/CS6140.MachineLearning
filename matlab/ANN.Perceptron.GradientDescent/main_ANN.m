%target matrix
T = eye(8);
%input matrix
D = T;
D=[ones(8,1),D];
[dRowCount,dColCount]= size(D);

% weight matrix for hidden and output layer
hidden_w = (rand(9,3) -0.5);
output_w = (rand(4,8)-0.5);
[hid_w_rows,hid_w_cols] = size(hidden_w);
[out_w_rows,out_w_cols] = size(output_w);

% derivative vector for hidden and output layer
hidden_deriv = ones(hid_w_cols,1)*100;
output_deriv = ones(out_w_cols,1)*100;

totalOutput = ones(8,8);% record final output.

iterLimit = 50000;
objValueVector = zeros(1,iterLimit);
outputVector =zeros(8,iterLimit);
learningRate = 0.3;
for iter = 1: iterLimit
    % for every datapoint; stochastic
    for drow = 1 :dRowCount
        hidden_out = ones(3,1) * 100;
        output_out = ones(8,1) * 100;
        
        target = T(drow,:)'; % colum vector
        
        %-----------------Forward Prop----------------
        % FP into hidden layer
        input_out = D(drow,:)';
        for col = 1:hid_w_cols
            hidden_out(col) = sigmoid(sum(input_out' * hidden_w(:,col)));
        end
        hidden_out = [1;hidden_out];% add bias term.
        
        % FP into output layer
        for col = 1: out_w_cols
            output_out(col) = sigmoid(sum(hidden_out' * output_w(:, col)));
        end
        %got output of ANN----> output_out
        
        %-----------------Back Prop----------------
        
        %output_deriv calc
        for h = 1: out_w_cols
            ok = output_out(h);
            tk = target(h);
            output_deriv(h) = ok*(1-ok)*(tk-ok);
        end
        
        %hidden_deriv calc
        for h = 1: hid_w_cols
            oh = hidden_out(h);
            %h+1 is due to free term
            sigma= output_w(h+1,:) *output_deriv;
            hidden_deriv(h) = oh*(1-oh)*sigma;
        end
        
        %update all weight vector in the ANN
        
        hidden_w = update_weights(learningRate,hidden_w, hidden_deriv, input_out);
        output_w = update_weights(learningRate,output_w, output_deriv, hidden_out);
                
        totalOutput(:, drow) = output_out;                
    end
    
    % calc obj func
    err = sum_sqerr(totalOutput,T);    
    objValueVector(iter) = err;
    
    if err < 0.01
        break;
    end
    % FP into hidden layer
end

%1:iterLimit,objValueVector,
%plot(1:iterLimit,outputVector);
plot(1:iter,objValueVector);

title(sprintf('rate = %f',learningRate));
hidden_w
totalOutput
