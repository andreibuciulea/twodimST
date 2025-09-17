function [S_est, time] = estimate_graph(model, C, Cs, Ct, S_true, Ss_true, St_true, max_iters, prms)
% estimate_graph: Function to solve different graph estimation problems
%
% Inputs:
%   model     : string, one of "pr1", "pr2", "pr3", "pr1-cvx", "pr2-cvx", "pr3-cvx"
%   C         : data matrix for the problem
%   Cs, Ct    : spatial and temporal covariance matrices (or other dimension-specific matrices)
%   S_true    : true graph matrix for error computation (optional)
%   Ss_true, St_true : true dimension-specific graphs (optional)
%   max_iters : maximum number of iterations for iterative methods
%   prms      : additional parameters (alpha, learning rate, lambda, etc.)
%
% Outputs:
%   S_est     : estimated graph (matrix or structure containing Ss, St)
%   time      : computation time in seconds

verbose = prms.verbose;

switch model
    case 'pr1-cvx'
        fprintf('--- Solving Problem 1 (CVX) ---\n');
        tic;
        S_est = solve_problem1_cvx(C, verbose);
        time = toc;

    case 'pr1-aug'
        %fprintf('--- Solving Problem 1 (Augmented) ---\n');
        tic;
        S_est = solve_problem1_aug(C, S_true, max_iters, prms);
        time = toc;

    case 'pr2-cvx'
        fprintf('--- Solving Problem 2 (CVX) ---\n');
        tic;
        [Ss, St] = solve_problem2_cvx(C, Cs, Ct, prms.alpha, verbose);
        time = toc;
        S_est = kron(Ss, St);

    case 'pr2-aug'
        fprintf('--- Solving Problem 2 (Augmented) ---\n');
        tic;
        S_est = solve_problem2_aug(C, prms.alpha, max_iters);
        time = toc;

    case 'pr3-cvx'
        fprintf('--- Solving Problem 3 (CVX) ---\n');
        tic;
        [Ss, St] = solve_problem3_cvx(Cs, Ct);
        time = toc;
        S_est = kron(Ss, St);

    case 'pr3-aug'
        fprintf('--- Solving Problem 3 (Augmented) ---\n');
        tic;
        [Ss, St] = solve_problem3_aug(Ct, Cs, St_true, Ss_true, max_iters);
        time = toc;
        S_est = kron(Ss, St);

    case 'pr1'
        fprintf('--- Solving Problem 1 (Gradient) ---\n');
        tic;
        S_est = solve_problem1(C, max_iters, prms.lr, prms.lambda);
        time = toc;

    case 'pr2'
        fprintf('--- Solving Problem 2 (Gradient) ---\n');
        tic;
        [Ss, St] = solve_problem2(C, prms.alpha, max_iters, prms.lr, prms.lambda);
        time = toc;
        S_est = kron(Ss, St);

    case 'pr3'
        fprintf('--- Solving Problem 3 (Gradient) ---\n');
        tic;
        [Ss, St] = solve_problem3(Cs, Ct, max_iters, prms.lr, prms.lambda);
        time = toc;
        S_est = kron(Ss, St);

    case 'pr4-cvx'
        fprintf('--- Solving Problem 4 (CVX) ---\n');
        tic;
        [Ss, St] = solve_problem4_cvx(Cs, Ct, verbose);
        time = toc;
        S_est = kron(Ss, St);

    case 'DNNLasso'
        lams = size(Cs,1)*1e-3;
        lamt = size(Ct,1)*1e-3;
        OPTIONS.tol = 5e-3;
        fprintf('--- Solving DNNLasso ---\n');
        tic;
        [~, Gamma, Omega, ~, ~, ~, ~, ~, ~] = DNNLasso(Ct, Cs, lamt, lams, OPTIONS);
        time = toc;
        St = abs(Gamma - diag(diag(Gamma)));
        St = St / max(max(St));
        Ss = abs(Omega - diag(diag(Omega)));
        Ss = Ss / max(max(Ss));
        S_est = kron(Ss, St);

    case 'TERRALasso'
        lambda = [1e-2 1e-2];
        a = 1e2;
        tol = 1e-10;
        maxiter = 200;
        type = 'L1';
        ps(1) = size(Cs,1);
        ps(2) = size(Ct,1);
        Sa{1} = Cs; % spatial covariance
        Sa{2} = Ct; % temporal covariance
        fprintf('--- Solving TERRALasso ---\n');
        tic;
        [Psi, ~] = teralasso(Sa, ps, type, a, tol, lambda, maxiter);
        time = toc;
        L_s = Psi{1}; L_t = Psi{2};
        if ~isreal(L_s) || ~isreal(L_t)
            L_s = abs(L_s);
            L_t = abs(L_t);
        end
        L_s = L_s - diag(diag(L_s));
        L_t = L_t - diag(diag(L_t));
        L_r = kron(L_s, L_t);
        S_est = L_r / max(max(L_r));

    otherwise
        error('Unrecognized model: %s', model);
end

end

