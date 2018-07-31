%
% One possible way to initialize and optimize hyperparameters:
%
% Uses omgpEinc to perform the update of qZ and omgpbound to update
% hyperparameters.
%

function  [F, qZ, loghyperinit, mu, C, pi0, convergence, loghyper] = omgp(covfunc, M, X, Y, Xs, loghyperinit)

if ~issorted(X) || size(X,2)~=1
    warning('Your *inputs* are multidimensional or not sorted. Optimization may have trouble converging! (Multidimensional outputs are OK).')
end

% --- Initialization
[N,oD]=size(Y);
maxiter = 100;

% Assume zero-mean functions, substract mean 
meany=mean(Y);   
stdy = std(Y);
Y=(Y-ones(N,1)*meany)./(ones(N,1)*stdy+1e-6);

% --- Independent or shared hyperparameters
if length(covfunc) == M 
    display('Using a different set of hyperparameters for each component');
else
    display('Using shared hyperparameters for all components');
end

% --- Initial hyperparameters setting
lengthscale=log(mean((max(X)-min(X))'/2/5));
lengthscale(lengthscale<-1e2)=-1e2;
covpower=0.5*log(1);
noisepower=0.5*log(1/8)*ones(M,1);

if nargin < 6
    loghyper = [];
    for m = 1:length(covfunc)
        cm = covfunc{m};
        if strcmp(cm{1},'covNoise')
            loghyper = [loghyper;covpower];
        elseif strcmp(cm{1},'covLINone')
            loghyper = [loghyper;lengthscale;covpower];
        elseif strcmp(cm{1},'covSEiso')
            loghyper = [loghyper;lengthscale;covpower];
        elseif strcmp(cm{1},'covConst')
            loghyper = [loghyper;noisepower;covpower];
        else
            error('Covariance type not supported')
        end
    end
else
    loghyper = loghyperinit(1:end-M);
    noisepower = loghyperinit(end-M+1:end);
end

% Add responsibilities
qZ = rand(N, M) + 10;qZ= qZ./repmat(sum(qZ,2),1,M);
logqZ = log(qZ);logqZ=logqZ-logqZ(:,1)*ones(1,M);logqZ = logqZ(:,2:end);
loghyper = [loghyper; zeros(M-1,1); noisepower;logqZ(:)];

% --- Iterate EM updates
F_old = inf;convergence=[];
for iter= 1:maxiter
    [loghyper, conv1] = omgpEinc(loghyper, covfunc, M, X, Y);
    fprintf(1,'Bound after E-step is %.4f\n',conv1(end))
    [loghyper, conv2] = minimize(loghyper, 'omgpbound', 10, 'learnhyp', covfunc, M, X, Y);
    convergence = [conv1;conv2;];F=convergence(end);
    if abs(F-F_old)<abs(F_old)*1e-6
        break
    end
    F_old=F;
end
if iter == maxiter display('Maximum number of iterations exceeded');end

% Final pass, also updating pi0
[loghyper, conv] = minimize(loghyper, 'omgpbound', 20, 'learnall', covfunc, M, X, Y);
F = conv(end);
loghyperinit = [loghyper(1:end-N*(M-1)-2*M+1);  loghyper(end-N*(M-1)-M+1:end-N*(M-1))];

% Compute qZ and pi0
logqZ= [zeros(N,1) reshape(loghyper(end-N*(M-1)+1:end),N,M-1)];
qZ = exp(logqZ - max(logqZ,[],2)*ones(1,M));qZ = qZ./(sum(qZ,2)*ones(1,M));

logpZ = [0; loghyper(end-N*(M-1)-2*M+2:end-N*(M-1)-M)]'; 
logpZ = logpZ - max(logpZ);logpZ = logpZ-log(sum(exp(logpZ)));pi0 = exp(logpZ);

% --- Make predictions
if nargin > 4 % There is also test data
    
    Ntst = size(Xs,1);            
    [mu, C] = omgpbound(loghyper, 'learnall', covfunc, M, X, Y, Xs);
    mu = repmat(meany, [Ntst 1 M])+mu.*repmat(stdy+1e-6,[Ntst 1 M]);
    C = C.*repmat((stdy+1e-6).^2,[Ntst 1 M]);
end