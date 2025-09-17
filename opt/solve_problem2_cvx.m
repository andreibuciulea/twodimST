function [Ss, St] = solve_problem2_cvx(C, Cs, Ct, alpha, verbose)
% SOLVE_PROBLEM2_CVX  Alternating convex optimization for separable spatial/temporal graphs
%
%   [Ss, St] = solve_problem2_cvx(C, Cs, Ct, alpha, verbose)
%
% Problem (convex alternating form):
%   minimize_{Ss,St}   ||Ss||_1 + alpha * ||St||_1
%   subject to          || (Ss ⊗ St) * C - C * (Ss ⊗ St) ||_F <= epsilon
%                       Ss, St symmetric, elementwise nonnegative, zero diagonal,
%                       and each row sum >= 1.
%
% Method:
%   Alternating minimization: fix St and solve for Ss (CVX), then fix Ss and solve for St.
%   Each subproblem uses an increasing epsilon schedule if infeasible, up to epsilon_max.
%   The outer loop stops early if Ss and St change little between iterations.
%
% Inputs:
%   C       - (N x N) matrix where N = s * t, corresponds to spatio-temporal cov.
%   alpha   - scalar weight for the temporal sparsity term (||St||_1).
%   verbose - logical flag to print progress (true/false).
%
% Outputs:
%   Ss, St  - estimated spatial and temporal factors (both n x n where n = sqrt(N)).
%
% Notes:
%   - The function assumes Ss and St share the same dimension n where n^2 = size(C,1).
%   - Adjust epsilon_min/max/growth_factor, tol_break and patience to tune robustness.

    if nargin < 3
        verbose = false;
    end

    % infer factor size (assumes square factors and N = n^2)
    s = size(Cs,1);
    t = size(Ct,1);

    % Initialize Ss, St (safe defaults)
    Ss = eye(s);
    St = eye(t);

    % Epsilon schedule for feasibility relaxation
    epsilon_min   = 1e-8;
    epsilon_max   = 1e3;
    growth_factor  = 10;
    mu = 1e8;

    % Outer loop stopping by small variation
    tol_break = 1e-4;    % relative tolerance for change
    patience  = 2;       % consecutive iterations below tol to stop

    prev_Ss = Ss;
    prev_St = St;
    no_change_counter = 0;

    max_iters = 10;
    for k = 1:max_iters
        if verbose
            fprintf('=== Outer iter %d ===\n', k);
        end

        % === Optimize Ss given St ===
        epsilon = epsilon_min;
        solved = false;
        while epsilon <= epsilon_max
            if verbose
                fprintf('  Solving Ss with epsilon = %.1e ...\n', epsilon);
            end
            %for perfect covariance is better using epsilon
            %for sampled covariance is better using mu
            cvx_begin quiet
                variable Ss_try(s,s) symmetric
                S_kron = kron(Ss_try, St);
                minimize(1*sum(Ss_try(:)) + mu*norm(C*S_kron - S_kron*C, 'fro'))% St contribution is constant here
                subject to
                    %norm(C*S_kron - S_kron*C, 'fro') <= epsilon
                    diag(Ss_try) == 0
                    Ss_try >= 0
                    sum(Ss_try, 2) >= 1
            cvx_end

            if exist('cvx_status','var') && (strcmp(cvx_status,'Solved') || strcmp(cvx_status,'Inaccurate/Solved'))
                Ss = (Ss_try + Ss_try')/2;  % symmetrize small numeric asymmetry
                Ss = Ss/max(max(Ss));
                solved = true;
                if verbose
                    fprintf('   Ss solved (epsilon = %.1e, status: %s)\n', epsilon, cvx_status);
                end
                break;
            else
                if verbose
                    if exist('cvx_status','var')
                        fprintf('   Ss NOT solved (status: %s). Increasing epsilon.\n', cvx_status);
                    else
                        fprintf('   CVX did not report status for Ss. Increasing epsilon.\n');
                    end
                end
                epsilon = epsilon * growth_factor;
            end
        end
        if ~solved
            warning('Could not solve Ss (pr2) in outer iter %d. Returning last feasible values.', k);
            break;
        end
        
        % === Optimize St given Ss ===
        epsilon = epsilon_min;
        solved = false;
        while epsilon <= epsilon_max
            if verbose
                fprintf('  Solving St with epsilon = %.1e ...\n', epsilon);
            end
            cvx_begin quiet
                variable St_try(t,t) symmetric
                S_kron = kron(Ss, St_try);
                minimize(1* sum(St_try(:)) + mu*norm(C*S_kron - S_kron*C, 'fro'))   % Ss contribution is constant here
                subject to
                    %norm(C*S_kron - S_kron*C, 'fro') <= epsilon
                    diag(St_try) == 0
                    St_try >= 0
                    sum(St_try, 2) >= 1
            cvx_end

            if exist('cvx_status','var') && (strcmp(cvx_status,'Solved') || strcmp(cvx_status,'Inaccurate/Solved'))
                St = (St_try + St_try')/2;
                St = St/max(max(St));
                solved = true;
                if verbose
                    fprintf('   St solved (epsilon = %.1e, status: %s)\n', epsilon, cvx_status);
                end
                break;
            else
                if verbose
                    if exist('cvx_status','var')
                        fprintf('   St NOT solved (status: %s). Increasing epsilon.\n', cvx_status);
                    else
                        fprintf('   CVX did not report status for St. Increasing epsilon.\n');
                    end
                end
                epsilon = epsilon * growth_factor;
            end
        end
        if ~solved
            warning('Could not solve St (pr2) in outer iter %d. Returning last feasible values.', k);
            break;
        end

        % === Stopping criterion: relative change on Ss and St ===
        prev_norm_Ss = norm(prev_Ss, 'fro');
        prev_norm_St = norm(prev_St, 'fro');

        if prev_norm_Ss > 0
            rel_change_Ss = norm(Ss - prev_Ss, 'fro') / prev_norm_Ss;
        else
            rel_change_Ss = norm(Ss - prev_Ss, 'fro');
        end

        if prev_norm_St > 0
            rel_change_St = norm(St - prev_St, 'fro') / prev_norm_St;
        else
            rel_change_St = norm(St - prev_St, 'fro');
        end

        max_rel_change = max(rel_change_Ss, rel_change_St);

        if verbose
            fprintf('  rel_change_Ss = %.3e, rel_change_St = %.3e, max = %.3e\n', ...
                    rel_change_Ss, rel_change_St, max_rel_change);
        end

        if max_rel_change < tol_break
            no_change_counter = no_change_counter + 1;
        else
            no_change_counter = 0;
        end

        if no_change_counter >= patience
            if verbose
                fprintf('Stopping at outer iter %d: small change (max_rel_change = %.3e) for %d iterations.\n', ...
                        k, max_rel_change, patience);
            end
            break;
        end

        % update previous for next iteration
        prev_Ss = Ss;
        prev_St = St;
    end

    if verbose
        fprintf('Finished alternating optimization. Returning Ss and St.\n');
    end
end

