function [Ss, St] = solve_problem3_aug(Ct, Cs, St_true, Ss_true, max_iters)
% solve_problem3_aug: Alternating graph estimation using augmented approach
%
% Solves:
%   min ||Ss||_1 + alpha*||St||_1
%   s.t. ||(Ss ⊗ St) * C - C * (Ss ⊗ St)||_F <= epsilon
% with adaptive increase of epsilon if the problem is infeasible.
%
% Inputs:
%   Ct        : data matrix for the temporal dimension
%   Cs        : data matrix for the spatial dimension
%   St_true   : ground truth temporal graph (optional, for initialization)
%   Ss_true   : ground truth spatial graph (optional, for initialization)
%   max_iters : maximum number of iterations for the estimator
%
% Outputs:
%   Ss : estimated spatial graph
%   St : estimated temporal graph

% Initialize graphs as fully connected minus self-loops
St_init = ones(size(Ct)) - eye(size(Ct));
Ss_init = ones(size(Cs)) - eye(size(Cs));

% Estimate the temporal graph St
St = estimate_S_from_C(Ct, St_init, St_true, max_iters);

% Estimate the spatial graph Ss
Ss = estimate_S_from_C(Cs, Ss_init, Ss_true, max_iters);

end

