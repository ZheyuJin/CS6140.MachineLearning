%% for vote.
% parsing data
filename = 'data\\vote\\vote.data';
posclass = 'd';
% text = fileread(filename);
[A,delimiterOut]=importdata(filename);
A=char(A);


% remove lines with ?
tmp = A == '?';
rows = (sum(tmp') ==0)';
A= A(rows,:); %keep a portion.

%make number
label = A(:,end);
label = label == posclass; % map into 0 and 1
A = A+0; % turn into number.
A(:,end) = label; % reform the matrix


% call boosting func

% runs with getOptStump function. 
adaboost( A, 100, @getOptStump);

% runs with getOptStump function. 
adaboost( A, 300, @getRandStump);

