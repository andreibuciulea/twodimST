function [Ss, St] = solve_problem2_aug(C, alpha, max_iters)
% solve_problem2_aug: Alternating optimization for separable graph learning
%
% Solves:
%   min ||Ss||_1 + alpha * ||St||_1
%   s.t. ||(Ss ⊗ St) * C - C * (Ss ⊗ St)||_F <= epsilon
% with adaptive increase of epsilon if the problem is infeasible.
%
% Inputs:
%   C         : data matrix
%   alpha     : weight for the temporal graph sparsity
%   max_iters : maximum number of alternating iterations
%
% Outputs:
%   Ss : estimated graph along dimension S (spatial)
%   St : estimated graph along dimension T (temporal)

n = round(sqrt(size(C,1)));       % assume square matrices for Ss and St
Ss = ones(n) - eye(n);            % initial spatial graph
St = ones(n) - eye(n);            % initial temporal graph

epsilon_min = 1e-3;               % minimum tolerance for Frobenius norm constraint
epsilon_max = 1e3;                % maximum tolerance
growth_factor = 10;                % factor to increase epsilon if infeasible
max_iters = min(max_iters, 20);   % safety cap on iterations

for k = 1:max_iters
    % === Optimize Ss given St ===
    epsilon = epsilon_min;
    solved = false;
    while epsilon <= epsilon_max
        cvx_begin quiet
            variable Ss_try(n,n) symmetric
            minimize( sum(sum(Ss_try)) )
            subject to
                norm(kron(Ss_try, St)*C - C*kron(Ss_try, St), 'fro') <= epsilon
                diag(Ss_try) == 0
                Ss_try >= 0
                sum(Ss_try, 2) >= 1
        cvx_end
        if strcmp(cvx_status, 'Solved') || strcmp(cvx_status, 'Inaccurate/Solved')
            Ss = Ss_try;
            solved = true;
            break;
        else
            epsilon = epsilon * growth_factor;
        end
    end
    if ~solved
        warning('Failed to solve Ss in iteration %d.', k);
        break;
    end
    Ss = Ss / max(max(Ss)); % normalize

    % === Optimize St given Ss ===
    epsilon = epsilon_min;
    solved = false;
    while epsilon <= epsilon_max
        cvx_begin quiet
            variable St_try(n,n) symmetric
            minimize( alpha * sum(sum(St_try)) )
            subject to
                norm(kron(Ss, St_try)*C - C*kron(Ss, St_try), 'fro') <= epsilon
                diag(St_try) == 0
                St_try >= 0
                sum(St_try, 2) >= 1
        cvx_end
        if strcmp(cvx_status, 'Solved') || strcmp(cvx_status, 'Inaccurate/Solved')
            St = St_try;
            solved = true;
            break;
        else
            epsilon = epsilon * growth_factor;
        end
    end
    if ~solved
        warning('Failed to solve St in iteration %d.', k);
        break;
    end
    St = St / max(max(St)); % normalize

    % === Optional: visualize the graphs after each iteration ===
    figure()
    subplot(1,2,1)
    imagesc(Ss)
    colorbar()
    title('Spatial Graph Ss')
    subplot(1,2,2)
    imagesc(St)
    colorbar()
    title('Temporal Graph St')
end

end

