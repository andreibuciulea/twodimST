function [Ss, St] = solve_problem3_cvx(Cs, Ct, verbose)
% SOLVE_PROBLEM3_CVX  Joint convex estimation of spatial and temporal factors
%
%   [Ss, St] = solve_problem3_cvx(Cs, Ct, verbose)
%
% Problem solved:
%   minimize_{Ss,St}   ||Ss||_1 + ||St||_1
%   subject to          || Cs*Ss - Ss*Cs ||_F <= epsilon
%                       || Ct*St - St*Ct ||_F <= epsilon
%                       Ss, St symmetric, nonnegative, zero diagonal,
%                       and each row sum >= 1.
%
% Method:
%   Single CVX problem over Ss and St. If infeasible, epsilon is increased
%   multiplicatively (growth_factor) until a solution is found or epsilon_max
%   is reached.
%
% Inputs:
%   Cs, Ct  - (n x n) covariance-like matrices for the spatial and temporal domains
%   verbose - optional logical flag (default false) to enable progress printing
%
% Outputs:
%   Ss, St  - (n x n) estimated factors (if solver fails, returns identity matrices)

    if nargin < 3
        verbose = false;
    end

    % Dimensions and safe initializations
    s = size(Cs,1);
    t = size(Ct,1);

    % Safe fallback initial guesses
    Ss = eye(s);
    St = eye(t);

    % Epsilon schedule parameters
    epsilon_min   = 1e-8;
    epsilon_max   = 1e3;
    growth_factor  = 10;

    epsilon = epsilon_min;
    solved = false;
    mu = 1e8;

    while epsilon <= epsilon_max
        if verbose
            fprintf('Attempting joint CVX solve with epsilon = %.1e\n', epsilon);
        end
        cvx_begin quiet
            variable Ss_var(s,s) symmetric
            minimize( sum(Ss_var(:)) + mu*norm(Cs*Ss_var - Ss_var*Cs, 'fro'))
            subject to
                %norm(Cs*Ss_var - Ss_var*Cs, 'fro') <= epsilon
                diag(Ss_var) == 0;
                Ss_var >= 0;
                sum(Ss_var,2) >= 1;
        cvx_end

        cvx_begin quiet
            variable St_var(t,t) symmetric
            minimize( sum(St_var(:)) + mu*norm(Ct*St_var - St_var*Ct, 'fro'))
            subject to
                %norm(Ct*St_var - St_var*Ct, 'fro') <= epsilon
                diag(St_var) == 0;
                St_var >= 0;
                sum(St_var,2) >= 1;
        cvx_end

        if exist('cvx_status','var') && (strcmp(cvx_status,'Solved') || strcmp(cvx_status,'Inaccurate/Solved'))
            % Accept solution and symmetrize to remove tiny numerical asymmetries
            Ss = (Ss_var + Ss_var')/2;
            St = (St_var + St_var')/2;
            Ss = Ss/max(max(Ss));
            St = St/max(max(St));
            solved = true;
            if verbose
                fprintf('Solved joint CVX problem with epsilon = %.1e (status: %s)\n', epsilon, cvx_status);
            end
            break;
        else
            if verbose
                if exist('cvx_status','var')
                    fprintf('Not solved (status: %s). Increasing epsilon from %.1e.\n', cvx_status, epsilon);
                else
                    fprintf('CVX did not report status. Increasing epsilon from %.1e.\n', epsilon);
                end
            end
            epsilon = epsilon * growth_factor;
        end
    end

    if ~solved
        warning('Failed to solve joint CVX problem (pr3) within epsilon range. Returning fallback Ss and St (identities or last feasible if any).');
    end
end
