clc;clear all; close all;
%%%% This code is for graphs with community structure.
load('Graphs150.mat')
ps(1) = size(Ln,1);
ps(2) = size(Lp,1);
Len = [10 50 100 250 500 1000 2000 5000];
% Len = 500;
max_iter = 5;

Psi0{1} = Ln;
Psi0{2} = Lp;
%% param for BigLasso algorithm
lambda = [0.01 0.01];
a = 60;
tol = 1e-10;
maxiter = 200;
tic
type = 'L1';
%% Algorithm
L = KronSum(Psi0{1}, Psi0{2});
for i = 1:length(Len)
    N = Len(i);
    for j = 1:max_iter
        [X T1 T2] = DataGen(Ln, Lp, N);
        S{1} = T1/N;
        S{2} = T2/N;
        clear T1 T2 X;
        [ Psi,Psis ] = teralasso( S, ps,type,a,tol ,lambda,maxiter);
        L_r = KronSum(Psi{1}, Psi{2});
        [precision_1(i,j),recall_1(i,j),Fmeasure_1(i,j),NMI_1(i,j),num_of_edges_1(i,j)] = graph_learning_perf_eval(Ln,Psi{1});
        [precision_2(i,j),recall_2(i,j),Fmeasure_2(i,j),NMI_2(i,j),num_of_edges_2(i,j)] = graph_learning_perf_eval(Lp,Psi{2});
        [precision(i,j),recall(i,j),Fmeasure(i,j),NMI(i,j),num_of_edges(i,j)] = graph_learning_perf_eval(L,L_r);
        
        
        Psi_p{1} = LapProject(Psi{1});
        Psi_p{2} = LapProject(Psi{2});
        Psi_p{1}(abs(Psi_p{1})<0.3 ) = 0;
        Psi_p{2}(abs(Psi_p{2})<0.3 ) = 0;
        [precision_1p(i,j),recall_1p(i,j),Fmeasure_1p(i,j),NMI_1p(i,j),num_of_edges_1p(i,j)] = graph_learning_perf_eval(Ln,Psi_p{1});
        [precision_2p(i,j),recall_2p(i,j),Fmeasure_2p(i,j),NMI_2p(i,j),num_of_edges_2p(i,j)] = graph_learning_perf_eval(Lp,Psi_p{2});
%         [precisionp(i,j),recallp(i,j),Fmeasurep(i,j),NMIp(i,j),num_of_edgesp(i,j)] = graph_learning_perf_eval(L,L_r);

        en(i,j) = norm(Ln-Psi_p{1},'fro');
        ep(i,j) = norm(Lp-Psi_p{2},'fro');
    end
end

[X T1 T2] = DataGen(Ln, Lp, N);
Psi0{1} = Ln;
Psi0{2} = Lp;
lambda = [0.01 0.01];
S{1} = T1/N;
S{2} = T2/N;
clear T1 T2 X;

a = 60;

tol = 1e-10;

maxiter = 200;
tic
type = 'L1';
[ Psi,Psis ] = teralasso( S,ps,type,a,tol ,lambda,maxiter);

for k= 1: 2
    figure();
    subplot(211);spy(Psi0{k});
    subplot(212);spy(Psi{k});
end