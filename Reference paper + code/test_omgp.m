%
% Tests functions omgp_gen and omgp
%

clear all 
close all

% Number of time instants per GP, dimensions, and GPs

n = 20;
D = 2;
M = 3;


% Tunable hyperparameters
timescale = 20;
sigvar = 1;
noisevar = 0.002;

% Data generation and plotting
close all
loghyper = [log(timescale); 0.5*log(sigvar); 0.5*log(noisevar)];
x = xlsread('/Users/Gabriele/Desktop/Poli/OMGP_python/inputs/x.xlsx')
Y = xlsread('/Users/Gabriele/Desktop/Poli/OMGP_python/inputs/Y.xlsx')
%[x, Y] = omgp_gen(loghyper, n, D, M);

x_train = x(1:2:end);
Y_train = Y(1:2:end,:);
x_test = x(2:2:end);
Y_test = Y(2:2:end,:);

figure
plot3(x_train, Y_train(:,1), Y_train(:,2), 'kx')
title(sprintf('%d trajectories to be separated (drag to see)',M))
pause


% OMGP tracking and plotting
covfunc = {};
for m=1:1       % Same type of covariance function for every GP in the model
    covfunc = {covfunc{:} {'covSEiso'}};
end

[F, qZ, loghyperinit, mu, C, pi0] = omgp(covfunc, M, x_train, Y_train, x_test);

pi0

[NMSE, NLPD] = quality(Y_test, mu, C, pi0);

[nada, label] = max(qZ,[],2);
figure

colors = get(gcf,'DefaultAxesColorOrder');
for c = 1:M
    plot3(x_train(label==c),Y_train(label==c,1),Y_train(label==c,2),'color',colors(mod(c+1,7)+1,:),'marker','x','linestyle','none');
    hold on
    plot3(x_test,mu(:,1,c),mu(:,2,c),'color',colors(mod(c+1,7)+1,:));
end
xlabel('X Axis')
ylabel('Y Axis')
zlabel('Z Axis')
view([5,3,2])
grid on
