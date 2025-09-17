function D = diag3(T, dim1, dim2)
    % Extracts the diagonal elements of a tensor T along dimensions dim1 and dim2
    idx = 1:size(T, dim1);                  % Create an index vector along dim1
    subs = repmat({':'}, 1, ndims(T));      % Create a cell array of ':' for all dimensions
    subs{dim1} = idx;                       % Set indexing for dim1
    subs{dim2} = idx;                       % Set indexing for dim2
    D = T(subs{:});                         % Extract the diagonal elements
end

