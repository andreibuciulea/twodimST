function [Ss, St] = solve_problem3(Cs, Ct, max_iters, lr, lambda)
% min ||Ss||_1 + ||St||_1 s.t. Cs Ss = Ss Cs  and Ct St = St Ct
n = size(Cs,1);
Ss = rand(n); St = rand(n);
Ss = project_to_S(Ss); St = project_to_S(St);

for iter = 1:max_iters
    comm_s = Cs*Ss - Ss*Cs;
    comm_t = Ct*St - St*Ct;

    grad_Ss = lambda * (Cs'*comm_s - comm_s*Cs') + ones(n);
    grad_St = lambda * (Ct'*comm_t - comm_t*Ct') + ones(n);

    Ss = Ss - lr * grad_Ss;
    St = St - lr * grad_St;

    Ss = project_to_S(Ss);
    St = project_to_S(St);
end
end
