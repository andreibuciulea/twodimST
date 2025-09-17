function S = solve_problem1_cvx(C, verbose)
% SOLVE_PROBLEM1_CVX  Solve a convex relaxation for edge learning under graph-stationarity
%
%   S = solve_problem1_cvx(C, verbose)
%
% Problem solved (convex form):
%   minimize_S  ||S||_1
%   subject to  ||C*S - S*C||_F <= epsilon
%               S is symmetric
%               S >= 0 (elementwise)
%               diag(S) == 0
%               sum(S,2) >= 1   (each row sum >= 1)
%
% Inputs:
%   C       - (n x n) covariance-like matrix used in the commutativity constraint
%   verbose - logical flag to print progress messages (true/false)
%
% Output:
%   S       - (n x n) estimated adjacency-like matrix (if solver fails,
%             returns last successful solution or zeros(n))
%
% Notes:
%   - The algorithm repeatedly tries to solve the CVX problem with an
%     increasing tolerance `epsilon` until a feasible solution is found or
%     epsilon exceeds epsilon_max.
%   - The commutativity constraint is enforced in the Frobenius norm:
%         norm(C*S - S*C, 'fro') <= epsilon
%     which is a natural relaxation of exact commutation.
%   - The problem encourages sparsity via the l1 norm.
%   - Adjust epsilon_min, epsilon_max, and growth_factor to control robustness.

    if nargin < 2
        verbose = false;
    end

    n = size(C, 1);

    % Parameters controlling epsilon schedule
    epsilon_min   = 1e-8;   % initial (small) tolerance
    epsilon_max   = 1e3;    % maximum allowed tolerance
    growth_factor  = 10;    % factor by which epsilon is increased on failure

    % Initialize outputs / safe fallback
    S = zeros(n);

    epsilon = epsilon_min;
    mu = 1e8;
    solved = false;

    while epsilon <= epsilon_max
        if verbose
            fprintf('Attempting to solve with epsilon = %.1e\n', epsilon);
        end
        %for perfect covariance is better using epsilon
        %for sampled covariance is better using mu
        cvx_begin quiet
            variable S_var(n,n) symmetric
            minimize( norm(S_var, 1) + mu*norm(C*S_var - S_var*C, 'fro'))
            subject to
                %norm(C*S_var - S_var*C, 'fro') <= epsilon
                diag(S_var) == 0;
                S_var >= 0;
                sum(S_var, 2) >= 1;
        cvx_end

        % Accept either exact 'Solved' or 'Inaccurate/Solved' statuses
        if exist('cvx_status', 'var') && (strcmp(cvx_status, 'Solved') || strcmp(cvx_status, 'Inaccurate/Solved'))
            S = (S_var + S_var')/2;    % symmetrize to avoid tiny asymmetric numerics
            solved = true;
            if verbose
                fprintf('Solved with epsilon = %.1e (cvx_status: %s)\n', epsilon, cvx_status);
            end
            break;
        else
            if verbose
                if exist('cvx_status', 'var')
                    fprintf('Not solved with epsilon = %.1e (cvx_status: %s). Increasing epsilon...\n', epsilon, cvx_status);
                else
                    fprintf('CVX did not report status. Increasing epsilon from %.1e...\n', epsilon);
                end
            end
            epsilon = epsilon * growth_factor;
        end
    end

    if ~solved
        warning('Failed to solve the convex problem within the epsilon range (pr1). Returning fallback S (zeros or last feasible if any).');
    end
end

