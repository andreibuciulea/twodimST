clc;clear all;close all


addpath ./misc/
load('Graphs.mat')
p = size(Lp,1);
q = size(Lq,1);
Lp = (p/trace(Lp))*Lp;
Lq = (q/trace(Lq))*Lq;
L = KronSum(Lp, Lq);
W = diag(diag(L)) - L;
G = graph(W); % Product graph

[TrueIdx_Gp, binsizes] = conncomp(Gp);
[TrueIdx_Gq, binsizes] = conncomp(Gq);
[TrueIdx_G, binsizes] = conncomp(G);
clear binsizes W Wp Wq
%%
NumSignals  = [5000]; % Maximum number of samples
% NumSignals = [10 50 100 250 500 1000 2000 5000];
nIter = 5; % number of times we want to repeat the process
param.max_iter = 20000; % max number of iterations for waterfilling solver.
max_iter = 100; % max_iter for the proposed algorithm

param.b1 = 0.5;
param.g1 = 0.7;
param.k1 = Kp;

param.b2 = 0.3;
param.g2 = 0.7;
param.k2 = Kq;

for i = 1:length(NumSignals)
for j = 1:nIter
N = NumSignals(i); % Number of graph signals
[X, Sp, Sq] = DataGen(Lp, Lq, N);
Sp = Sp/N;
Sq = Sq/N;

[Lp_i, Lq_i, Vp, Vq, error, objec] = RPGL_Laplacian(Lp, Lq, Sp, Sq, param, max_iter);
L_r = KronSum(Lp_i, Lq_i);

[~, ~, f_p(i,j), ~, ~] = graph_learning_perf_eval(Lp,Lp_i);
[~, ~, f_q(i,j), ~, ~] = graph_learning_perf_eval(Lq,Lq_i);
[~, ~, f(i,j), ~, ~] = graph_learning_perf_eval(L,L_r);

% To evaluate the clustering performance
[V, ~] = eigs(full(L_r) + 1e-4*eye(size(L_r)), Kp*Kq, 'smallestabs');

[Result_Lp, est_idx_Lp] = perf_kmeans(Vp, Kp, TrueIdx_Gp);
[Result_Lq, est_idx_Lq] = perf_kmeans(Vq, Kq, TrueIdx_Gq);
[Result_L, est_idx_L] = perf_kmeans(V, Kp*Kq, TrueIdx_G);

Lp_pu(i,j) = Result_Lp(1);  Lq_pu(i,j) = Result_Lq(1); 
Lp_nmi(i,j) = Result_Lp(2); Lq_nmi(i,j) = Result_Lq(2);
Lp_ri(i,j) = Result_Lp(3);  Lq_ri(i,j) = Result_Lq(3);

L_pu(i,j) = Result_L(1);
L_nmi(i,j) = Result_L(2);
L_ri(i,j) = Result_L(3);
clear Result_Lp Result_Lq est_idx_Lp est_idx_Lq

end
end


figure(1);clf;subplot(221);spy(Lp);title('Ground Lp');
subplot(222);spy(Lq); title('Ground Lq')
subplot(223);spy(Lp_i);title('Recon Lp')
subplot(224);spy(Lq_i);title('Recon Lq')

Wp_r = diag(diag(Lp_i)) - Lp_i;
Wq_r = diag(diag(Lq_i)) - Lq_i;
Gp_r = graph(Wp_r);
Gq_r = graph(Wq_r);


figure(2);clf;subplot(221);plot(Gp);title('Grnd Lp')
subplot(222);plot(Gq);title('Grnd Lq')
subplot(223);plot(Gp_r);title('Recon Lp');
subplot(224);plot(Gq_r);title('Recon Lq');

% figure(4);clf;plot(smooth(sum(f_p,2)/nIter)); hold on
% plot(smooth(sum(Lp_pu,2)/nIter)); hold on
% plot(smooth(sum(Lp_nmi,2)/nIter)); hold on
% plot(smooth(sum(Lp_ri,2)/nIter)); hold on
% legend('Fmeasure','purity', 'NMI', 'RI')
% title('Lp')
% 
% figure(5);clf;plot(smooth(sum(f_q,2)/nIter)); hold on
% plot(smooth(sum(Lq_pu,2)/nIter)); hold on
% plot(smooth(sum(Lq_nmi,2)/nIter)); hold on
% plot(smooth(sum(Lq_ri,2)/nIter)); hold on
% legend('Fmeasure','purity', 'NMI', 'RI')
% title('Lq')
% 
% figure(6);clf;plot(smooth(sum(f,2)/nIter)); hold on
% plot(smooth(sum(L_pu,2)/nIter)); hold on
% plot(smooth(sum(L_nmi,2)/nIter)); hold on
% plot(smooth(sum(L_ri,2)/nIter)); hold on
% legend('Fmeasure','purity', 'NMI', 'RI')
% title('L')
% clear Wp Wq Wp_r Wq_r Gp Gq Gp_r Gq_r X

rmpath ./misc/

%%
function [Lp_i, Lq_i, Vp, Vq, error, objec] = RPGL_Laplacian(Lp, Lq, Sp, Sq, param, max_iter)
p = size(Sp,1);
q = size(Sq,1);

Gp_i = gsp_sensor(p);
Gq_i = gsp_sensor(q);
[Vp, ~] = eigs(full(Gp_i.L), param.k1,'smallestabs');
[Vq, ~] = eigs(full(Gq_i.L), param.k2,'smallestabs');
clear Gp_i Gq_i

Dp = duplication_matrix(p);
Dq = duplication_matrix(q);

P0 = blkdiag(2*param.b1*Dp'*Dp, 2*param.b2*Dq'*Dq);
q1p = [vec(Sp)'*Dp]';
q1q = [vec(Sq)'*Dq]';
% q2p = [param.g1*vec(Vp*Vp')'*Dp]';
% q2q = [param.g2*vec(Vq*Vq')'*Dq]';
% q0 = [q1p + q2p; q1q + q2q];

% Constraints
Cp = [vec(eye(p))'*Dp; kron(ones(p,1)',eye(p))*Dp];
dp = [p; zeros(p,1)];
Cq = [vec(eye(q))'*Dq; kron(ones(q,1)',eye(q))*Dq];
dq = [q; zeros(q,1)];
C = blkdiag(Cp, Cq);
d = [dp; dq];

clear Cp Cq dp dq

q1p = [vec(Sp)'*Dp]';
q1q = [vec(Sq)'*Dq]';
Ep = [];
Eq = [];
for k = 1:max_iter
    
    q2p = [param.g1*vec(Vp*Vp')'*Dp]';
    q2q = [param.g2*vec(Vq*Vq')'*Dq]';
    q0 = [q1p + q2p; q1q + q2q];
    
%     [l_wf err] = waterfill_solver(P0, q0, C, d, param.max_iter);
    [l_wf,  err] = PGL(P0, q0, C, d, 1e-6, 0.0051);
    v1 = size(Dp,2);
    v2 = size(Dq,2);
    
    
    Lp_i = full(reshape(Dp*l_wf(1:v1),p,p)); Lp_i(abs(Lp_i)<1e-4)=0;
    Lq_i = full(reshape(Dq*l_wf(v1+1:end),q,q)); Lq_i(abs(Lq_i)<1e-4)=0;
    
    [Vp, ~] = eigs(full(Lp_i), param.k1,'smallestabs');
    [Vq, ~] = eigs(full(Lq_i), param.k2,'smallestabs');
    
    Ep = [Ep eig(full(Lp_i))];
    Eq = [Eq eig(full(Lq_i))];
    
    % convergence
    if k>1
        error(k) = norm(Lp_prev- Lp_i,'fro') + norm(Lq_prev-Lq_i,'fro');
        objec(k) = trace(Sp*Lp_i) + trace(Sq*Lq_i) ...
            + param.b1*norm(Lp_i,'fro')^2 + param.b2*norm(Lq_i,'fro')^2 ...
            + param.g1*trace(Vp'*Lp_i*Vp) + param.g2*trace(Vq'*Lq_i*Vq);
    end
    if k>1 & error(k)<1e-4
        fprintf('Converged at %d iteration \n',k)
        break;
    end
    
    fprintf('trace(Vp^T*Lp_true*Vp) is %0.3f and trace(Vq^T*Lq_true*Vq) is %0.3f \n', trace(Vp'*Lp*Vp), trace(Vq'*Lq*Vq))
    fprintf('rank(Lp) is %d and rank of (Lq) is %d \n', rank(full(Lp_i),10^-3), rank(full(Lq_i),10^-3))
    
    Lp_prev = Lp_i;
    Lq_prev = Lq_i;
end
end


function [l err] = PGL(P, q, C, d, tol, rho)

p = diag(P);
%mu = rand(size(C,1),1);
mu = zeros(size(C,1),1);
l_aux = p.^(-1).*(C'*mu - q);
l = max(0, l_aux);

%%
    k=1;
    res = norm(C*l - d);
    while k < 20000
        if (res > tol) %this threshold is tunable, 10^-4 is good enough
            mu = mu - rho*(C*l - d); % gradient-descent 
            l_aux = p.^(-1).*(C'*mu - q);
            l = max(0, l_aux);
        end
        k = k+1;
        res = norm(C*l - d);
        err(k) = res;
    end
end

