function S = solve_problem1(C, max_iters, lr, lambda)
% solve_problem1: Projected gradient descent for
%   min_S ||S||_1   subject to   C*S = S*C  and  S ∈ S_set
%
% Inputs:
%   - C        : matrix C for the problem
%   - max_iters: maximum number of iterations
%   - lr       : learning rate
%   - lambda   : penalty parameter for the commutation term
%
% Output:
%   - S        : final projected solution

mu = 1e-6;              % small regularization term
N = size(C, 1);         % number of nodes
S = rand(N);             % random initialization
S = project_to_S(S);     % initial projection onto feasible set

S = zeros(N);            % reset S (optional, could also keep initialization)

for iter = 1:max_iters
    % Compute the commutator [C, S] = C*S - S*C
    commutator = C*S - S*C;
    
    % Gradient of the objective with respect to S
    grad = lambda * (C'*commutator - commutator*C') + mu*ones(N);
    
    % Update S using gradient descent
    S = S - lr * grad;
    
    % Project S back onto the feasible set
    S = project_to_S(S);
end

end

