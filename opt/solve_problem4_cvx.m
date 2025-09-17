function [Ss, St] = solve_problem4_cvx(Cs, Ct, verbose)
% Solves the following alternating optimization problem:
%
%   minimize    ||Ss||_1 + ||St||_1
%   subject to  ‖Cs*Ss - Ss*Cs‖_F ≤ epsilon
%               ‖Ct*St - St*Ct‖_F ≤ epsilon
%               Kronecker-based consistency constraints
%               diag(Ss) = 0,  Ss >= 0,  sum(Ss,2) >= 1
%               diag(St) = 0,  St >= 0,  sum(St,2) >= 1
%
% The solver uses an adaptive epsilon strategy (progressively relaxed 
% if infeasible) and alternates between optimizing Ss and St.
% Convergence is monitored by relative changes in Ss, St, 
% and their Kronecker product.

s = size(Cs, 1);
t = size(Ct, 1);
Ss = eye(s);
St = eye(t);
%Ss = generate_connected_ER(s,0.2);
%St = generate_connected_ER(t,0.2);

% Epsilon control parameters
epsilon_min = 1e-8;
epsilon_max = 1e3;
growth_factor = 10;
max_iters = 10;

% Stopping criteria based on relative changes
tol_break = 1e-4;    % tolerance for relative variation
patience  = 1;       % number of consecutive iterations below tol

% Initialization of previous solutions
prev_Ss = Ss;
prev_St = St;
no_change_counter = 0;

% (Optional) Kronecker monitoring
prev_kron = kron(Cs*prev_Ss, Ct*prev_St);
prev_kron_norm = norm(prev_kron, 'fro');
CsCt = kron(Cs,Ct);

for k = 1:max_iters
    % === Optimize Ss given Cs ===
    epsilon = epsilon_min;
    solved = false;
    CtSt = Ct*St;
    StCt = St*Ct;
    while epsilon <= epsilon_max
        cvx_begin quiet
            variable Ss_try(s,s) symmetric
            minimize( sum(sum(Ss_try)) )
            subject to
                norm(Cs*Ss_try - Ss_try*Cs, 'fro') <= epsilon
                %norm(CsCt*kron(Ss_try,St) - kron(Ss_try,St)*CsCt,'fro') <= epsilon
                norm(kron(CtSt,Cs*Ss_try) - kron(StCt,Ss_try*Cs),'fro') <= epsilon
                diag(Ss_try) == 0
                Ss_try >= 0
                sum(Ss_try,2) >= 1
        cvx_end
        if strcmp(cvx_status, 'Solved') || strcmp(cvx_status, 'Inaccurate/Solved')
            Ss = Ss_try;
            Ss = Ss/max(max(Ss));
            solved = true;
            break;
        else
            epsilon = epsilon * growth_factor;
        end
    end
    if ~solved
        warning('Could not solve Ss (pr4) in iteration %d.', k);
        break;
    end

    % === Optimize St given Ct ===
    epsilon = epsilon_min;
    solved = false;
    CsSs = Cs*Ss;
    SsCs = Ss*Cs;
    while epsilon <= epsilon_max
        cvx_begin quiet
            variable St_try(t,t) symmetric
            minimize( sum(sum(St_try)) )
            subject to
                norm(Ct*St_try - St_try*Ct, 'fro') <= epsilon
                %norm(CsCt*kron(Ss,St_try) - kron(Ss,St_try)*CsCt,'fro') <= epsilon
                norm(kron(Ct*St_try,CsSs) - kron(St_try*Ct,SsCs),'fro') <= epsilon
                diag(St_try) == 0
                St_try >= 0
                sum(St_try,2) >= 1
        cvx_end
        if strcmp(cvx_status, 'Solved') || strcmp(cvx_status, 'Inaccurate/Solved')
            St = St_try;
            St = St/max(max(St));
            solved = true;
            break;
        else
            epsilon = epsilon * growth_factor;
        end
    end
    if ~solved
        warning('Could not solve St (pr4) in iteration %d.', k);
        break;
    end

    % === Convergence check based on relative variations ===
    % Relative change in Ss
    prev_norm_Ss = norm(prev_Ss,'fro');
    if prev_norm_Ss > 0
        rel_change_Ss = norm(Ss - prev_Ss, 'fro') / prev_norm_Ss;
    else
        rel_change_Ss = norm(Ss - prev_Ss, 'fro');
    end

    % Relative change in St
    prev_norm_St = norm(prev_St,'fro');
    if prev_norm_St > 0
        rel_change_St = norm(St - prev_St, 'fro') / prev_norm_St;
    else
        rel_change_St = norm(St - prev_St, 'fro');
    end

    % Relative change in Kronecker product
    curr_kron = kron(Cs*Ss, Ct*St);
    curr_kron_norm = norm(curr_kron, 'fro');
    if prev_kron_norm > 0
        rel_change_kron = norm(curr_kron - prev_kron, 'fro') / prev_kron_norm;
    else
        rel_change_kron = norm(curr_kron - prev_kron, 'fro');
    end

    max_rel_change = max([rel_change_Ss, rel_change_St, rel_change_kron]);

    % Optional debugging output
    if verbose
        fprintf('Iter %d: rel_change_Ss=%.3e, rel_change_St=%.3e, rel_change_kron=%.3e, max=%.3e\n', ...
            k, rel_change_Ss, rel_change_St, rel_change_kron, max_rel_change);
    end

    % Check convergence
    if max_rel_change < tol_break
        no_change_counter = no_change_counter + 1;
    else
        no_change_counter = 0;
    end

    if no_change_counter >= patience
        if verbose
            fprintf('Terminated at iter %d: small variation (max_rel_change = %.3e) sustained for %d iterations.\n', ...
                k, max_rel_change, patience);
        end
        break;
    end

    % Update previous values
    prev_Ss = Ss;
    prev_St = St;
    prev_kron = curr_kron;
    prev_kron_norm = curr_kron_norm;
end

end


