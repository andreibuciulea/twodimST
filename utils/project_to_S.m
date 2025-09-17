function S_proj = project_to_S(S)

S = 0.5 * (S + S');         
S = max(S, 0);              
S(logical(eye(size(S)))) = 0; 


row_sums = sum(S, 2);
for i = 1:size(S,1)
    if row_sums(i) < 1
        deficit = 1 - row_sums(i);
        non_zero_idx = S(i,:) > 0;
        if any(non_zero_idx)
            S(i,non_zero_idx) = S(i,non_zero_idx) + deficit / sum(non_zero_idx);
        else
            S(i,:) = 1 / size(S,2);
            S(i,i) = 0;
        end
    end
end

S_proj = S;
end
