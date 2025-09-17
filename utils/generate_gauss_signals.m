function out = generate_gauss_signals(Ss,St,L,r,verbose)
    
    S = kron(Ss,St);
    s = size(Ss,1);
    t = size(St,1);
    hs = rand(L,1);
    ht = rand(L,1);
    Hs = zeros(s);
    Ht = zeros(t);
    for ii = 1:L
        Hs = Hs + hs(ii)*Ss^(ii-1);
        Ht = Ht + ht(ii)*St^(ii-1);
    end

    eigvals = eig(Ss);
    Cs_inv = (0.01-min(eigvals))*eye(s,s) + (0.9+0.1*rand(1,1))*Ss;
    Cs = inv(Cs_inv);

    eigvals = eig(St);
    Ct_inv = (0.01-min(eigvals))*eye(t,t) + (0.9+0.1*rand(1,1))*St;
    Ct = inv(Ct_inv);

    eigvals = eig(S);
    C_inv = (0.01-min(eigvals))*eye(s*t) + (0.9+0.1*rand(1,1))*S;
    C = inv(C_inv);

    X0 = zeros(s,t,r);
    Xv = zeros(s*t,r);

    Xs = sqrtm(Cs)*randn(s,r);
    Cs_sampled = Xs*Xs'/r;

    Xt = sqrtm(Ct)*randn(t,r);
    Ct_sampled = Xt*Xt'/r;
    
    %sampled covariance
    Xv = sqrtm(C)*randn(s*t,r);
    C_sampled = Xv*Xv'/(r);
    
    comm = norm(C*S-S*C,"fro")^2;
    comms = norm(C_sampled*S-S*C_sampled,"fro")^2;
    if verbose
        disp(['Commutativity of C: ' num2str(comm) ' | Commutativity of Cs: ' num2str(comms)]);
    end

    out.C = C/max(max(C));
    out.Ct = Ct/max(max(Ct));
    out.Cs = Cs/max(max(Cs));
    out.C_sampled = C_sampled/max(max(C_sampled));
    out.Ct_sampled = Ct_sampled/max(max(Ct_sampled));
    out.Cs_sampled = Cs_sampled/max(max(Cs_sampled));
end