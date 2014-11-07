
%% for crx.


%read conf
conf = 'data\\crx\\crx.config';
fid = fopen(conf);
str = fgets(fid);
[dataCount str]= strtok(str);
dataCount = str2double(dataCount);
[feat_discrete str]= strtok(str);
[feat_acucmul str]= strtok(str);
featCount = str2double(feat_acucmul)  + str2double(feat_discrete);% 1 for label.


% parsing data
filename = 'data\\crx\\crx.data';
posclass = '+';
% text = fileread(filename);
[A,delimiterOut]=importdata(filename);
%A = strrep(A, delimiterOut, '');%delete delimeter.
A = char(A);

D = zeros(dataCount,featCount+1); % data matrix
keeprows = zeros(dataCount,1);

actual_datacount =0; % used to ignroe ? containining lines
for i = 1: dataCount
    remain = A(i,:); % pick a line
    
    if(~isempty(strfind(remain, '?'))) % if ? is in the line, ignore.
        keeprows (i) =0;
        continue;
    end
   
    keeprows (i) =1;
    %parse the line; get out features.
    for j = 1: featCount +1;
        [field remain]= strtok(remain,delimiterOut);
        field = strrep(field,' ','');% delete space.
        
        num = str2double(field);
        
        if(~isreal(num)) % to kill complex numbers.
            num = NaN;
        end        
        
        if (isnan(num)) % a char.
            D(i,j) = field +0; % make char into number;
        else
            D(i,j) = num;
        end
        
    end
end
D =D(logical(keeprows), :);
%fix lables.
D(:,end) = D(:,end) =='+';

D = D(randperm(size(D,1)),:); % shuffle. yes.

% call boosting func
% runs with getOptStump function.
adaboost( D, 30, @getOptStump);

% runs with getOptStump function.
adaboost( D, 100, @getRandStump);