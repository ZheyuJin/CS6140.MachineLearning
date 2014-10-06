% mapps all data into [0,1]
function matrix = normalize(data)
[rowcount, colcount] = size(data);
matrix = data;

for col = 1 : colcount-1
    minn = min(data(:,col));
    matrix(:,col) = data(:,col) - minn;
    maxx = max(matrix(:,col));
    matrix(:,col) = matrix(:,col) / maxx;
end

end