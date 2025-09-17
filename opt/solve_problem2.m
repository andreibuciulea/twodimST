function [Ss, St] = solve_problem2(C, alpha, max_iters, lr, lambda)
% min_Ss,St ||Ss||_1 + alpha * ||St||_1  s.t.  C(Ss⊗St) = (Ss⊗St)C
n = sqrt(size(C,1));
Ss = rand(n); St = rand(n);
Ss = project_to_S(Ss); St = project_to_S(St);

for iter = 1:max_iters
    K = kron(Ss, St);
    commutator = C*K - K*C;
    comm_grad = lambda * (C'*commutator - commutator*C');

    grad_Ss = alpha * ones(n) + reshape(sum(reshape(comm_grad, n, n, n, n), [3 4]), n, n);
    grad_St = ones(n) + reshape(sum(reshape(comm_grad, n, n, n, n), [1 2]), n, n);

    Ss = Ss - lr * grad_Ss;
    St = St - lr * grad_St;

    Ss = project_to_S(Ss);
    St = project_to_S(St);
end
end
