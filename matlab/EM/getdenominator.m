function [ sum ] = getdenominator( xi, PI,MIU,SIGMAS )
%GETDENOMINATOR Summary of this function goes here
%   Detailed explanation goes here
MC = length (PI);
sum =0;
for j = 1:MC
    tmp = PI(j) * mvnpdf(xi, MIU(j,:),SIGMAS(:,:,j));
    sum = sum +  tmp;
end


end

