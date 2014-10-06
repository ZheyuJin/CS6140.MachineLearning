%{
result and lables has the same length.
    reuslt : output vector of the regressor.
    lables : label vector 
    thresh: the threashold.
%}
function [truePos, trueNeg, falsePos,falseNeg] = spam_acc_err_count(result,  labels, thresh)
truePos =0;
trueNeg =0;

falsePos =0;
falseNeg =0;

for i=1:length(result)
    if labels(i) == 1
        if result(i) >= thresh %if largeer than 0.5; its 1
            truePos= truePos + 1;
        else %else its 0
            falseNeg = falseNeg  + 1;
        end
    else 
        if result(i) < thresh
            trueNeg = trueNeg  + 1;
        else
            falsePos= falsePos + 1;
        end
    end
end

end