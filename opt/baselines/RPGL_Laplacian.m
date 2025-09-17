function [Lp_i, Lq_i, Vp, Vq, error, objec] = RPGL_Laplacian(Lp, Lq, Sp, Sq, param, max_iter)
p = size(Sp,1);
q = size(Sq,1);

Gp_i = generate_connected_ER(p,0.2);
Gq_i = generate_connected_ER(q,0.2);
Lpi = diag(sum(Gp_i))-Gp_i;
Lqi = diag(sum(Gq_i))-Gq_i;
[Vp, ~] = eigs(full(Lpi), param.k1,'smallestabs');
[Vq, ~] = eigs(full(Lqi), param.k2,'smallestabs');
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