function S = estimate_S_from_C(C, S_init, S_true, max_iters)   
    N = size(C,1);
    On = ones(N);
    In = eye(N);
    P = zeros(N);
    Z = P;
    beta = 1;%commutativity
    eta = 1e-3;%frobenius
    rho = 1e-3;%sparsity
    delta = 1e-3;
    del = 1e-4;
    %%%%% For the first subproblem we update S, P, and Z
    C_inv = abs(inv(C));
    S = (C_inv+C_inv')/2;
    S = S/max(max(S));

    la_max = max(abs(eig(C)));
    L1 = (1*beta*la_max^2);
    nSt = norm(S_true,'fro');
    results = zeros(4,max_iters);
    for r = 1:max_iters
        w_S = On./(abs(S) + del).*(On-In);
        %gradient of S
        gS = C*C*S + S*C*C - 2*C*S*C + P*C - C*P + 1/beta*(C*Z - Z*C);
        %update S before projection
        Sp = S - 1/L1*(rho*w_S.*S + eta*S + beta*gS); 
        S = max(0,abs(Sp-diag(diag(Sp))));
        S = (S+S')/2;
        S = S/max(max(S));
        %update for P
        Pp = C*S-S*C+1/beta*Z;
        P = lag_mult_projection(Pp,delta);
        %update for Z
        Z = Z + beta*(C*S-S*C-P);

        results(r,1) = norm(C*S-S*C,'fro')^2;
        results(r,2) = (norm(S-S_true,'fro')/nSt)^2;
        
        if mod(r,100)==0
            figure(4)
            semilogy(results(:,2))
            grid on
            legend('S err')
        end
    end
end
