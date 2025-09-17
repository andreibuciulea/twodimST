function D = duplication_matrix(p)
    % Returns the duplication matrix D_p of size p^2 x (p*(p+1)/2)
    % such that vec(S) = D * vech(S) for any symmetric S in R^{p x p}.
    
    k = 0;
    D = zeros(p^2, p*(p+1)/2);
    for j = 1:p
        for i = j:p
            k = k + 1;
            E = zeros(p);
            E(i,j) = 1;
            E(j,i) = 1 * (i ~= j); % avoid double count on diagonal
            D(:,k) = reshape(E,[],1);
        end
    end
end
