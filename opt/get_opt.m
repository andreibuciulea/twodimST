function v = get_opt(opts, name, default)
    if isfield(opts,name) && ~isempty(opts.(name)), v = opts.(name);
    else, v = default; end
end