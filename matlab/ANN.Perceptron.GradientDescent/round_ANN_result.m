function [ m] = round_ANN_result(m)
[rows, cols] = size(m);
for i = 1: rows*cols
    m(i) = round(m(i));
end
end

function r = round(value)
if value < 0.5
    r = 0;
else
    r =1;
end
end