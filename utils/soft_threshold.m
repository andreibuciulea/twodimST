function S = soft_threshold(Sp, tau)
% Prox de ||·||_1:  Soft(A, tau)
    S = max(tau,Sp-diag(diag(Sp)));
    S = (S+S')/2;
    S = S/max(max(S));
end