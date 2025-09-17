function out = generate_st_signals(Ss,St,L,r,verbose)
    
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

    Hs2 = Hs^2;
    Ht2 = Ht^2;

    Cs_sampled = zeros(s);
    Ct_sampled = zeros(t);
    Cs = trace(Ht2)*Hs2;
    Ct = trace(Hs2)*Ht2;
    C = kron(Hs2,Ht2);

    X0 = zeros(s,t,r);
    Xv = zeros(s*t,r);
    for m = 1:r
        W = randn(s,t);
        HWH = Hs*W*Ht;
        X0(:,:,m) = HWH;
        Xv(:,m) = vec(HWH');
        Cs_sampled = Cs_sampled + HWH*HWH'/r;
        Ct_sampled = Ct_sampled + HWH'*HWH/r;
    end
    
    %sampled covariance
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