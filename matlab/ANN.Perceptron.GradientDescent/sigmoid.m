function [var] = sigmoid(z)
%% sigmoid function. 
var = 1/(1+exp(-z));
end