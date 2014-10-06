% percepron gd
D = load('e:\\study\\ml\\hw2\\perceptronData.txt');
% normalize
D = normalize(D);
[totalRows,totalCols] = size(D);

label = D(:,end); % label
freeTerms = ones(totalRows,1); % free terms
D =D(:, 1:end-1);
D=[freeTerms,D]; % add free terms.
[datarows, featureCols] = size(D);

%relocate y=-1 to y=1 by x= -x;
[D,label] = perceptron_relocate_points(D,label);

%initialize w vector
w = (rand(featureCols,1) - 0.5); %% (-0.5, 0.5)
w = percptron_gd_batch_update(0.03, w,D,label,100);
w