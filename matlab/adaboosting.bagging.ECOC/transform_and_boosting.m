function [ output_args ] = transform_and_boosting( filename, posclass )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here


% parsing data
[A,delimiterOut]=importdata(filename);
A = strrep(A, delimiterOut, '');%delete delimeter.
A = char(A);

% remove lines with ?
tmp = A == '?';
rows = (sum(tmp') ==0)';
A= A(rows,:); %keep a portion.

%make number
label = A(:,end);
label = label == posclass ; % map into 0 and 1
A = A+0; % turn into number.
A(:,end) = label; % reform the matrix


% call boosting func

% runs with getOptStump function. 
adaboost( A, 15, @getOptStump);

% runs with getOptStump function. 
adaboost( A, 300, @getRandStump);

end

