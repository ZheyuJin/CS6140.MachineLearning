function [ Data ] = read_matrix( path,DataPointsCount,FeaturesCount )
%read_matrix read matrix from given path, last colum is label.
%   Detailed explanation goes here
fid = fopen(path);

Data = zeros(DataPointsCount,FeaturesCount+1);

lineNum=0;
while true
    % read and check.
    line = fgets (fid);
    if ~ischar(line)
        break;
    end
    lineNum = lineNum +1;
    C = strsplit(line,' ');
    A = char(C);
    
    rows = size(A,1);
    for r = 1: rows-1
        if r ==1 % read label
            label = sscanf(A(r,:),'%d');
            Data(lineNum,FeaturesCount +1) = label;
        else % read data col and value.
            
            [val] = sscanf(A(r,:),'%d:%f');
            if isempty(val)
                break;
            end
            Data(lineNum,round(val(1)) +1) = val(2); % matrix has 0 based colum index.
        end
    end
end

fclose(fid);

end

