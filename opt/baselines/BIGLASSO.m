clc;clear all;close all

% This code is for obtaining the graph learing performance using the method
% A. Kalaitzis, J. Lafferty, N. D. Lawrence, and S. Zhou, “The bigraphicallasso,” 
% inProc.  of  the  30th  Int.  Conf.  on  Machine  Learning,  vol.  28,no. 3, 
% Atlanta, Georgia, USA, June 2013.


% Since the precision matrix estimates from BiGLasso are not valid Laplacian matrices,
% we project them onto a set of all the valid graph Laplacian matrices

% To run this file add all the folders and subfolders of the BigLASSO folder 
% to the current working directory


addpath ./misc/
load('Graphs.mat')
p = size(Lp,1);
q = size(Lq,1);
Lp = (p/trace(Lp))*Lp;
Lq = (q/trace(Lq))*Lq;
L = KronSum(Lp, Lq);
L = L/trace(L)*size(L,1);
W = diag(diag(L)) - L;
G = graph(W); % Product graph

[TrueIdx_Gp, binsizes] = conncomp(Gp);
[TrueIdx_Gq, binsizes] = conncomp(Gq);
[TrueIdx_G, binsizes] = conncomp(G);
clear binsizes W Wp Wq

%%
NumSignals = [10 50 100 250 500 1000 2000 5000];
% NumSignals = [5000];
nIter = 10;  % Number of iterations for one data sample

Psi0{1} = Lp; % Defining the inputs as defined in the BIGLasso code
Psi0{2} = Lq;


lambda = [0.01 0.01];
a = 60;
tol = 1e-10;
maxiter = 200;
ps(1) = size(Lp,1);
ps(2) = size(Lq,1);
tic
type = 'L1';

for i = 1:length(NumSignals)
N = NumSignals(i);
for j = 1:nIter
    [X T1 T2] = DataGen(Lp, Lq, N);
    S{1} = T1/N;
    S{2} = T2/N;
    clear T1 T2 X;

    [ Psi, Psis ] = teralasso( S, ps, type, a, tol ,lambda, maxiter);
    L_r = KronSum(Psi{1}, Psi{2});

    [~, ~,fp(i,j), ~, ~] = graph_learning_perf_eval(Lp, Psi{1});
    [~, ~,fq(i,j), ~, ~] = graph_learning_perf_eval(Lq, Psi{2});
    [~, ~,f(i,j), ~, ~] = graph_learning_perf_eval(L, L_r);
    
    % Results with the covariance matrix output
    [Vp, ~] = eigs(full(Psi{1}), k1,'smallestabs');
    [Vq, ~] = eigs(full(Psi{2}), k2,'smallestabs');
    [V, ~] = eigs(full(L_r), k1*k2,'smallestabs');
    
    [Result_Lp, est_idx_Lp] = perf_kmeans(Vp, k1, TrueIdx_Gp);
    [Result_Lq, est_idx_Lq] = perf_kmeans(Vq, k2, TrueIdx_Gq);
    [Result_L, est_idx_Lq] = perf_kmeans(V, k1*k2, TrueIdx_G);
    clear Vp Vq V
    Lp_pu(i,j) = Result_Lp(1);  Lq_pu(i,j) = Result_Lq(1);
    Lp_nmi(i,j) = Result_Lp(2); Lq_nmi(i,j) = Result_Lq(2);
    Lp_ri(i,j) = Result_Lp(3);  Lq_ri(i,j) = Result_Lq(3);
    
    L_pu(i,j) = Result_L(1);
    L_nmi(i,j) = Result_L(2);
    L_ri(i,j) = Result_L(3);
    
    
    
    % Results with the Laplacian matrix (After projection)
    Psi_p{1} = LapProject(Psi{1});
    Psi_p{2} = LapProject(Psi{2});
    Wp = diag(diag(Psi_p{1})) - Psi_p{1};


    Psi_p{1}(abs(Psi_p{1})<0.001) = 0;
    Psi_p{2}(abs(Psi_p{2})<0.001) = 0;
    L_proj = KronSum(Psi_p{1}, Psi_p{2});
    
   
    [Vp, ~] = eigs(full(Psi_p{1}), k1,'smallestabs');
    [Vq, ~] = eigs(full(Psi_p{2}), k2,'smallestabs');
    [V, ~] = eigs(full(L_proj), k1*k2,'smallestabs');
    
    [Result_PLp, est_idx_PLp] = perf_kmeans(Vp, k1, TrueIdx_Gp);
    [Result_PLq, est_idx_PLq] = perf_kmeans(Vq, k2, TrueIdx_Gq);
    [Result_PL, est_idx_PLq] = perf_kmeans(V, k1*k2, TrueIdx_G);
    clear Vp Vq V
    PLp_pu(i,j) = Result_PLp(1);  PLq_pu(i,j) = Result_PLq(1);
    PLp_nmi(i,j) = Result_PLp(2); PLq_nmi(i,j) = Result_PLq(2);
    PLp_ri(i,j) = Result_PLp(3);  PLq_ri(i,j) = Result_PLq(3);
    
    PL_pu(i,j) = Result_PL(1);
    PL_nmi(i,j) = Result_PL(2);
    PL_ri(i,j) = Result_PL(3);    

    [~, ~,Projfp(i,j), ~, ~] = graph_learning_perf_eval(Lp,Psi_p{1});
    [~, ~,Projfq(i,j), ~, ~] = graph_learning_perf_eval(Lq,Psi_p{2});
    [~, ~,Projf(i,j), ~, ~] = graph_learning_perf_eval(L, L_proj);

end
end

rmpath ./misc/