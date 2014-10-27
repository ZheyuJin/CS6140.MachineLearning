

D =load('2gaussian.txt');
[DPC,FC] = size(D);
MC = 2; % gaussian model count; you have to tell me this.

%% random initialiize for params.
PI = ones(MC,1) / MC;
oldPI = PI;
Z = zeros(DPC,MC); % expectation.
for i = 1: DPC
    if mod(i, 7) ==1
        Z(i,1) = 1;
    else
        Z(i,2) = 1;
    end
end

MIU = rand(MC,FC);
%SIGMAS = randi([1 5],FC,FC,MC);
%make sigmas symmatric
for i = 1:MC
    SIGMAS(:,:,i) = eye(FC,FC) + 2*i;
end


%% iteration.
maxiter = 100;
isConverge= false;
llh = zeros(maxiter,1);
tol = 1e-10;

for iter =1:maxiter
    
    
    %% Estep
    
    for i =1:DPC % data point count
        for m=1: MC % model count
            %             % not using logarithm. %
            %             p = ((2*pi) ^ (-FC/2));
            %             sigma = SIGMAS(:,:,m);
            %             sigmaDet = det(sigma) ^ -0.5;
            %             tmp = (D(i,:) - MIU(m,:))' ;
            %             expo = exp(-0.5* tmp' * inv(sigma) * tmp);
            %             p = p*sigmaDet * expo;
            p = mvnpdf(D(i,:), MIU(m,:),SIGMAS(:,:,m));
            denom= getdenominator(D(i,:), PI,MIU,SIGMAS );
            Z(i,m) =  p * PI(m) / denom;
        end
    end
    
    %% M step. upadata param for each gaussian
    for m = 1: MC
        sigmaZim = sum(Z);
        %% PI
        PI(m) = sigmaZim(m) / DPC;
        
        %% MIU
        up =0;
        for i = 1:DPC
            up = up+ D(i,:)*Z(i,m);
        end
        MIU(m,:) = up ./ sigmaZim(m);
        %% SIGMA
        up =0;
        for i = 1:DPC
            tmp= D(i,:)-MIU(m,:);
            up= up+ Z(i,m) *   tmp' * tmp ;
        end
        
        SIGMAS(:,:,m) =up ./ sigmaZim(m);
    end
    
    SIGMAS
    MIU
    PI
    
    %% check convergensce
    errsum(iter) = sum((oldPI -PI) .^ 2)
    if errsum < 0.0001
        break;        
    end
   oldPI =PI;
end

iter
plot(1:iter, errsum);


