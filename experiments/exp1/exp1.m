clear; clc;
addpath(genpath("../opt"))
addpath(genpath("../utils"))

% Parameters
alpha = 1;              % Regularization weight (problem 2)
max_iters = 1e3;        % Maximum iterations
lr = 1;                 % Learning rate
lambda = 1;             % Commutativity regularizer
nG = 100;                 % Number of graphs per size (can increase)
sig_type = 'ST';
verbose = false;
L = 3;
r = 1e6;                % Number of realizations
ps = 0.3;               % Spatial connection probability
pt = 0.3;               % Temporal connection probability

% Models
Models = ["pr1-cvx","pr2-cvx","pr3-cvx"];
nM = numel(Models);

% Graph sizes to test
graph_sizes = [5 5; 5 6; 6 6; 6 7; 7 7; 7 8; 8 8; 8 9; 9 9; 9 10; 10 10]; % each row = [s t]
% Preallocate results
num_sizes = size(graph_sizes,1);
all_errors = zeros(num_sizes, nG, nM);
all_fscore = zeros(num_sizes, nG, nM);
all_times = zeros(num_sizes, nG, nM);
all_comm = zeros(num_sizes, nG, nM);
tic 
for sz = 1:num_sizes
    s = graph_sizes(sz,1);
    t = graph_sizes(sz,2);
    N = s*t;
    fprintf('Running experiments for graph size s=%d, t=%d (N=%d)\n', s, t, N);
    
    parfor g = 1:nG
        % Generate spatiotemporal graph and signals
        [S,Ss,St] = generate_sptemp_graph(s,t,ps,pt);
        figure()
        subplot(131);imagesc(Ss);
        subplot(132);imagesc(St);
        subplot(133);imagesc(S);
        out = generate_st_signals(Ss,St,L,r,verbose);
        C_sampled = out.C_sampled;
        C = out.C;
        Cs = out.Cs;
        Ct = out.Ct;
        nS = norm(S,'fro')^2;

        errors_g = zeros(1,nM);
        fscore_g = zeros(1,nM);
        times_g = zeros(1,nM);
        comm_g = zeros(1,nM);

        for m = 1:nM
            params = struct('alpha',alpha,'lr',lr,'lambda',lambda,'verbose',verbose);
            [S_est, Mtime] = estimate_graph(Models(m),C,Cs,Ct,S,Ss,St,max_iters,params);
            
            errors_g(m) = norm(S_est-S,'fro')^2/nS;
            fscore_g(m) = fscore(S,S_est);
            times_g(m) = Mtime;
            comm_g(m) = norm(C*S_est-S_est*C,'fro')^2;
        end

        all_errors(sz,g,:) = errors_g;
        all_fscore(sz,g,:) = fscore_g;
        all_times(sz,g,:) = times_g;
        all_comm(sz,g,:) = comm_g;
    end

end
toc
save('exp1_results_v0.mat');

%% ===================== ANÁLISIS / PLOTEO: MEDIA y MEDIANA + COMM =====================
load("exp1_results_v0.mat")
num_sizes = size(graph_sizes,1);
N_vect = graph_sizes(:,1) .* graph_sizes(:,2);  % número de nodos para cada tamaño
nM = numel(Models);

% Agregados (media y mediana sobre repeticiones g)
mean_err  = squeeze(nanmean(all_errors, 2));   % [num_sizes x nM]
med_err   = squeeze(nanmedian(all_errors, 2));

mean_f1   = squeeze(nanmean(all_fscore, 2));   % [num_sizes x nM]
med_f1    = squeeze(nanmedian(all_fscore, 2));

mean_time = squeeze(nanmean(all_times, 2));
med_time  = squeeze(nanmedian(all_times, 2));

mean_comm = squeeze(nanmean(all_comm, 2));
med_comm  = squeeze(nanmedian(all_comm, 2));

% Estética
line_styles = {'-','--',':','-.'};
markers = {'^','s','o','d','*','v','>','<'};
colors = lines(nM);
lw = 2; ms = 8; fs_axes = 16; fs_lbl = 18; fs_leg = 12;

outdir = 'figs_sizes';
if ~exist(outdir,'dir'), mkdir(outdir); end

% Ordenar por N (por si graph_sizes no está ordenado)
[ N_sorted, idx_sort ] = sort(N_vect);
mean_err_s  = mean_err(idx_sort, :);
med_err_s   = med_err(idx_sort, :);
mean_f1_s   = mean_f1(idx_sort, :);
med_f1_s    = med_f1(idx_sort, :);
mean_time_s = mean_time(idx_sort, :);
med_time_s  = med_time(idx_sort, :);
mean_comm_s = mean_comm(idx_sort, :);
med_comm_s  = med_comm(idx_sort, :);

% Crear figura 2x4: Error | F1 | Time | Comm
figure('Name','Metrics vs N (mean & median)','Position',[100 100 1800 700]);

% --- Error (mean) ---
subplot(2,4,1); 
for m = 1:nM
    semilogy(N_sorted, mean_err_s(:,m), 'LineWidth', lw, ...
        'LineStyle', line_styles{1+mod(m-1,numel(line_styles))}, ...
        'Marker', markers{1+mod(m-1,numel(markers))}, 'MarkerSize', ms, ...
        'Color', colors(m,:));
    hold on
end
grid on; box on;
xlabel('Number of nodes $N$','Interpreter','latex','FontSize',fs_lbl);
ylabel('Frobenius error (mean)','Interpreter','latex','FontSize',fs_lbl);
title('Mean Error','Interpreter','latex','FontSize',fs_lbl);
set(gca,'FontSize',fs_axes);

% --- F1 (mean) ---
subplot(2,4,2); 
for m = 1:nM
    semilogy(N_sorted, mean_f1_s(:,m), 'LineWidth', lw, ...
        'LineStyle', line_styles{1+mod(m-1,numel(line_styles))}, ...
        'Marker', markers{1+mod(m-1,numel(markers))}, 'MarkerSize', ms, ...
        'Color', colors(m,:));
    hold on
end
grid on; box on; ylim([0 1]); yticks(0:0.1:1);
xlabel('Number of nodes $N$','Interpreter','latex','FontSize',fs_lbl);
ylabel('F-score (mean)','Interpreter','latex','FontSize',fs_lbl);
title('Mean F1','Interpreter','latex','FontSize',fs_lbl);
set(gca,'FontSize',fs_axes);

% --- Time (mean) ---
subplot(2,4,3); 
for m = 1:nM
    semilogy(N_sorted, mean_time_s(:,m), 'LineWidth', lw, ...
        'LineStyle', line_styles{1+mod(m-1,numel(line_styles))}, ...
        'Marker', markers{1+mod(m-1,numel(markers))}, 'MarkerSize', ms, ...
        'Color', colors(m,:));
    hold on
end
grid on; box on;
xlabel('Number of nodes $N$','Interpreter','latex','FontSize',fs_lbl);
ylabel('Runtime (s, mean)','Interpreter','latex','FontSize',fs_lbl);
title('Mean Time','Interpreter','latex','FontSize',fs_lbl);
set(gca,'FontSize',fs_axes);

% --- Comm (mean) ---
subplot(2,4,4); 
for m = 1:nM
    semilogy(N_sorted, mean_comm_s(:,m), 'LineWidth', lw, ...
        'LineStyle', line_styles{1+mod(m-1,numel(line_styles))}, ...
        'Marker', markers{1+mod(m-1,numel(markers))}, 'MarkerSize', ms, ...
        'Color', colors(m,:));
    hold on
end
grid on; box on;
xlabel('Number of nodes $N$','Interpreter','latex','FontSize',fs_lbl);
ylabel('Comm (mean, $\|CS-SC\|_F^2$)','Interpreter','latex','FontSize',fs_lbl);
title('Mean Comm','Interpreter','latex','FontSize',fs_lbl);
set(gca,'FontSize',fs_axes);

% --- Error (median) ---
subplot(2,4,5); 
for m = 1:nM
    semilogy(N_sorted, med_err_s(:,m), 'LineWidth', lw, ...
        'LineStyle', line_styles{1+mod(m-1,numel(line_styles))}, ...
        'Marker', markers{1+mod(m-1,numel(markers))}, 'MarkerSize', ms, ...
        'Color', colors(m,:));
    hold on
end
grid on; box on;
xlabel('Number of nodes $N$','Interpreter','latex','FontSize',fs_lbl);
ylabel('Frobenius error (median)','Interpreter','latex','FontSize',fs_lbl);
title('Median Error','Interpreter','latex','FontSize',fs_lbl);
set(gca,'FontSize',fs_axes);

% --- F1 (median) ---
subplot(2,4,6); 
for m = 1:nM
    plot(N_sorted, med_f1_s(:,m), 'LineWidth', lw, ...
        'LineStyle', line_styles{1+mod(m-1,numel(line_styles))}, ...
        'Marker', markers{1+mod(m-1,numel(markers))}, 'MarkerSize', ms, ...
        'Color', colors(m,:));
    hold on
end
grid on; box on; ylim([0 1]); yticks(0:0.1:1);
xlabel('Number of nodes $N$','Interpreter','latex','FontSize',fs_lbl);
ylabel('F-score (median)','Interpreter','latex','FontSize',fs_lbl);
title('Median F1','Interpreter','latex','FontSize',fs_lbl);
set(gca,'FontSize',fs_axes);

% --- Time (median) ---
subplot(2,4,7); 
for m = 1:nM
    semilogy(N_sorted, med_time_s(:,m), 'LineWidth', lw, ...
        'LineStyle', line_styles{1+mod(m-1,numel(line_styles))}, ...
        'Marker', markers{1+mod(m-1,numel(markers))}, 'MarkerSize', ms, ...
        'Color', colors(m,:));
    hold on
end
grid on; box on;
xlabel('Number of nodes $N$','Interpreter','latex','FontSize',fs_lbl);
ylabel('Runtime (s, median)','Interpreter','latex','FontSize',fs_lbl);
title('Median Time','Interpreter','latex','FontSize',fs_lbl);
set(gca,'FontSize',fs_axes);

% --- Comm (median) ---
subplot(2,4,8); 
for m = 1:nM
    semilogy(N_sorted, med_comm_s(:,m), 'LineWidth', lw, ...
        'LineStyle', line_styles{1+mod(m-1,numel(line_styles))}, ...
        'Marker', markers{1+mod(m-1,numel(markers))}, 'MarkerSize', ms, ...
        'Color', colors(m,:));
    hold on
end
grid on; box on;
xlabel('Number of nodes $N$','Interpreter','latex','FontSize',fs_lbl);
ylabel('Comm (median, $\|CS-SC\|_F^2$)','Interpreter','latex','FontSize',fs_lbl);
title('Median Comm','Interpreter','latex','FontSize',fs_lbl);
set(gca,'FontSize',fs_axes);

% Leyenda (fuera, única)
hL = legend(Models,'Interpreter','latex','FontSize',fs_leg,'Orientation','horizontal');
hL.Position = [0.25 0.01 0.5 0.03];
set(hL,'Color','none');

sgtitle(sprintf('Metrics vs N (N = s \\times t) — mean (fila 1) / median (fila 2)'), 'Interpreter','latex','FontSize',16);

%
figure()

% --- Left y-axis (F-score, solid lines) ---
yyaxis left
for m = 1:nM
    semilogy(N_sorted, mean_f1_s(:,m), 'LineWidth', lw, ...
        'LineStyle', '-', ...  % solid line
        'Marker', markers{1+mod(m-1,numel(markers))}, 'MarkerSize', ms, ...
        'Color', colors(m,:));
    hold on
end
ylabel('F-score','Interpreter','latex','FontSize',fs_lbl);
hold off;
% --- Right y-axis (Runtime, dotted lines) ---
yyaxis right
for m = 1:nM
    semilogy(N_sorted, mean_time_s(:,m), 'LineWidth', lw, ...
        'LineStyle', ':', ...  % dotted line
        'Marker', markers{1+mod(m-1,numel(markers))}, 'MarkerSize', ms, ...
        'Color', colors(m,:));
    hold on
end
ylabel('Runtime (s)','Interpreter','latex','FontSize',fs_lbl);

% --- Common settings ---
xlabel('Number of nodes $N$','Interpreter','latex','FontSize',fs_lbl);
%title('Mean F1 and Runtime','Interpreter','latex','FontSize',fs_lbl);
grid on; box on;
set(gca,'FontSize',fs_axes);
lg = legend({'ST','K-ST','SepK-ST'}, 'Location', 'best','Interpreter','latex','FontSize',12);
set(lg,'color','none');
xlim([25 100])
xticks([25, 50, 75, 100])
xticklabels({'25','50','75','100'})
