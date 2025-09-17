function [S,Ss,St] = generate_sptemp_graph(s,t,ps,pt)
    Ss = generate_connected_ER(s,ps);
    St = generate_connected_ER(t,pt);
    S = kron(Ss,St);
end