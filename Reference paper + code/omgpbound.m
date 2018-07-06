% Computes the negative of the Marginalized Variational Bound (F) and its 
% derivatives wrt loghyper (dF).
%
% Parameters:
% loghyper: K hyperparameters, pZ, sn2 (M trajectories), logqZ
% learn: 'learnqZ', 'learnhyp', 'learnall'
% covfunc: Array of covariance functions. If it is a single one, it is shared.
% M: Number of trajectories
% X, Y, Xs: Inputs, outputs, test inputs
%
% (c) Miguel Lazaro-Gredilla 2010

function  [F, dF] = omgpbound(loghyper, learn, covfunc, M, X, Y, Xs)

% --- Initialize

%[N, D] = size(X); 
[N, oD] = size(Y);

logqZ = [zeros(N,1) reshape(loghyper(end-N*(M-1)+1:end),N,M-1)];
logqZ = logqZ - max(logqZ,[],2)*ones(1,M);logqZ = logqZ-log(sum(exp(logqZ),2))*ones(1,M);qZ=exp(logqZ);
sn2 = ones(N,1)*exp(2*loghyper(end-N*(M-1)-M+1:end-N*(M-1)))';
logpZ = [0; loghyper(end-N*(M-1)-2*M+2:end-N*(M-1)-M)]'; 
logpZ = logpZ - max(logpZ);
logpZ = logpZ-log(sum(exp(logpZ)));
logpZ = ones(N,1)*logpZ;
sqB = sqrt(qZ./sn2);
dlogqZ = zeros(N,M);

% --- Contribution of independent (modified) GPs
if nargin == 6; F = 0; dF = zeros(size(loghyper)); 
elseif nargin==7 F = zeros(size(Xs,1),oD,M); dF = zeros(size(Xs,1),oD,M); end
hypstart=1; 
cm = covfunc{1}; numhyp = eval(feval(cm{:})); K = feval(cm{:}, loghyper(1:numhyp), X);
for m = 1:M
    if length(covfunc) > 1
        cm = covfunc{m};numhyp = eval(feval(cm{:}));
        K = feval(cm{:}, loghyper(hypstart:hypstart+numhyp-1), X);
    else
        hypstart = 1;
    end
    R = chol(eye(N) + K.*(sqB(:,m)*sqB(:,m)'));
    sqBY =  (sqB(:,m)*ones(1,oD)).*Y;
    v = R'\sqBY;
    if nargin == 6
        F = F + 0.5*sum(sum(v.^2)) + oD*sum(log(diag(R)));
        if nargout == 2 % Compute derivatives
            U = R'\diag(sqB(:,m));alpha = U'*v;
            diagW =sum((Y - K*alpha).^2,2) + oD*(diag(K) - sum((U*K).^2,1)'); % diag( (I+KB)\y * ((I+KB)\y)' ) + diag(K - K*U'*U*K)
            if ~strcmp(learn, 'learnqZ')
                W = oD*(U'*U)-alpha*alpha';                % precompute for convenience
                for i = 1:numhyp
                    if strcmp('learnall', learn) && m==1
                        disp('');
                    end
                    dF(i+hypstart-1) = dF(i+hypstart-1) + sum(sum(W.*feval(cm{:}, loghyper(hypstart:hypstart+numhyp-1), X, i)))/2;
                end
                dF(end-N*(M-1)-M+m) = diagW'*qZ(:,m)*exp(-2*loghyper(end-N*(M-1)-M+m))*-2/2; % diagW * dB/dsn2
                
            end
            if ~strcmp(learn, 'learnhyp')
                dlogqZ(:,m) = diagW./sn2(:,m)/2; 
            end
        end
    elseif nargin == 7 % Compute predictions
        U = R'\diag(sqB(:,m));alpha = U'*v;
        [Kss, Kfs] = feval(cm{:}, loghyper(hypstart:hypstart+numhyp-1), X, Xs);
        F(:,:, m) =  Kfs'*alpha;
        if nargout == 2
            dF(:,:,m) = (sn2(1,m) + Kss - sum((U*Kfs).^2,1)')*ones(1,oD);
        end
    end
    hypstart = hypstart + numhyp;
end
if hypstart+2*M+N*(M-1)-2 ~= length(loghyper) error('Incorrect number of parameters');end

if nargin ==6
    KLZ = sum(sum(qZ.*(logqZ-logpZ),2)); % KL Divergence from the posterior to the prior on Z
    F = F + oD/2*sum(sum(qZ.*log(2*pi*sn2))) + KLZ;
    
    if nargout == 2
        if ~strcmp(learn, 'learnhyp')
            dKLZlogpz =  sum(-qZ+exp(logpZ))';
            dF(end-N*(M-1)-2*M+2:end-N*(M-1)-M) = dKLZlogpz(2:end);  %Derivative wrt pZ
            dlogqZ = dlogqZ + logqZ-logpZ+ oD/2*log(2*pi*sn2); % Derivative wrt qZ
            dlogqZ = qZ.*(dlogqZ-sum(qZ.*dlogqZ,2)*ones(1,M)) ; % Derivative wrt actual hyperparam "beta" defnining qZ
            dlogqZ = dlogqZ(:,2:end);
            dF(end-N*(M-1)+1:end) = dlogqZ(:);
        end
        if ~strcmp(learn, 'learnqZ')
            dF(end-N*(M-1)-M+1:end-N*(M-1)) = dF(end-N*(M-1)-M+1:end-N*(M-1)) + oD/2*sum(qZ*2,1)'; %Derivative wrt sn2
        end
    end
end