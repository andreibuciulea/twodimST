function out = generate_sptemp_signals(S,L,s,t,r)

    option = 1;
   
    N = size(S,1);
    h = rand(L,1);
    H = zeros(N);
    for ii = 1:L
        H = H + h(ii)*S^(ii-1);
    end
    C = H^2;
    if option == 1
        [V,D] = eig(S); 
        dc = log(eig(C));
        dc = dc + abs(min(dc)+0.1);
        C = V*diag(dc)*V';
    end


    %spatio temporal signals
    X_iid=randn(N,r);
    X_0 = H*X_iid;
    %sampled covariance
    C_sampled = X_0*X_0'/(r-1);
    
    comm = norm(C*S-S*C,"fro")^2;
    comms = norm(C_sampled*S-S*C_sampled,"fro")^2;
    disp(['Commutativity of C: ' num2str(comm)]);
    disp(['Commutativity of Cs: ' num2str(comms)]);
    figure()
    subplot(121)
    imagesc(C)
    colorbar()
    subplot(122)
    imagesc(C_sampled)
    colorbar()

    %spatial and temporal sampled covariance
    X_0st = reshape(X_0,[s,t,r]);
    Cs = zeros(s);
    Ct = zeros(t);
    for nt = 1:t
        X0s = squeeze(X_0st(:,nt,:));
        Cs = Cs+X0s*X0s';
    end
    Cs = Cs/(r-1);
    for ns = 1:s
        X0t = squeeze(X_0st(ns,:,:));
        Ct = Cs+X0t*X0t';
    end
    Ct = Ct/(r-1);

    out.C = C;
    out.C_sampled = C_sampled;
    out.Ct = Ct;
    out.Cs = Cs;
end