% function [Tr1G, Tr2G] = partial_traces(G, ns, nt)
%     T = reshape(G, [ns, nt, ns, nt]); % (i,a,j,b)
% 
%     % Tr1: suma sobre i=j, queda nt x nt
%     Tr1G = squeeze(sum(diag3(T,1,3),1));
%     Tr1G = squeeze(sum(Tr1G,2));
% 
%     % Tr2: suma sobre a=b, queda ns x ns
%     Tr2G = squeeze(sum(diag3(T,2,4),2));
%     Tr2G = sum(Tr2G,3);
% end

function [Tr1G, Tr2G] = partial_traces(G, ns, nt)
% TR1: nt x nt  ;  TR2: ns x ns
    T = reshape(G,[ns, nt, ns, nt]); % (i,a,j,b)
    % Tr1(a,b) = sum_i T(i,a,i,b)  -> suma diagonal en dims 1 y 3
    Tr1G = zeros(nt,nt);
    for a=1:nt
        for b=1:nt
            s = 0;
            for i=1:ns
                s = s + T(i,a,i,b);
            end
            Tr1G(a,b) = s;
        end
    end
    % Tr2(i,j) = sum_a T(i,a,j,a)  -> suma diagonal en dims 2 y 4
    Tr2G = zeros(ns,ns);
    for i=1:ns
        for j=1:ns
            s = 0;
            for a=1:nt
                s = s + T(i,a,j,a);
            end
            Tr2G(i,j) = s;
        end
    end
end

