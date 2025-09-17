%% ============================================================
%  EXPERIMENT: Vary number of realizations (r) with a fixed graph
%  Measures: Frobenius error, F-score, runtime (s) 
%  Models: "pr1-cvx", "pr2-cvx", "pr3-cvx", "pr4-cvx"
% ============================================================
clear; clc;
addpath(genpath("../../opt"))
addpath(genpath("../../utils"))

%% -------- General parameters --------
alpha     = 1;        % regularization weight (if applicable)
max_iters = 1e3;      % maximum iterations for iterative methods
lr        = 1;        % learning rate (if applicable)
lambda    = 1;        % commutativity regularizer (if applicable)
nG        = 100;      % number of repetitions per r (signal re-sampling)
sig_type  = 'ST';     % signal type
verbose   = false;    % verbosity flag
L         = 3;        % filter degree (if applicable)
ps        = 0.3;      % spatial connection probability (for random graph generation)
pt        = 0.3;      % temporal connection probability

%% -------- Models to compare --------
Models = ["pr1-cvx","pr2-cvx","pr3-cvx"];
nM = numel(Models);

%% -------- Fixed graph size --------
s = 8;  % number of spatial nodes
t = 8;  % number of temporal instants
N = s*t;

%% -------- Range of realizations to evaluate --------
r_values = round(logspace(2, 6, 9));  % 10^2 ... 10^6 (9 points)
num_r = numel(r_values);

%% -------- Preallocate results matrices --------
all_errors  = zeros(nG, num_r, nM);
all_fscore  = zeros(nG, num_r, nM);
all_times   = zeros(nG, num_r, nM);
all_comm    = zeros(nG, num_r, nM);

fprintf('Fixed graph with s=%d, t=%d (N=%d). Varying number of realizations r...\n', s, t, N);
t1 = tic;

%% -------- Main parallel loop over repetitions --------
parfor g = 1:nG
    % Generate fixed spatio-temporal graph
    [S, Ss, St] = generate_sptemp_graph(s, t, ps, pt);
    nS = norm(S, 'fro')^2;
    
    for ir = 1:num_r
        r = r_values(ir);
        fprintf('  -> r = %d\n', r);
        
        % Generate signals for current r (same graph, different noise)
        out  = generate_st_signals(Ss, St, L, r, verbose);
        C    = out.C_sampled;   % full covariance
        Cs   = out.Cs_sampled;  % spatial covariance
        Ct   = out.Ct_sampled;  % temporal covariance
        
        for m = 1:nM
            params = struct('alpha', alpha, 'lr', lr, 'lambda', lambda, 'verbose', verbose);
            [S_est, Mtime] = estimate_graph(Models(m), C, Cs, Ct, S, Ss, St, max_iters, params);
            
            % Store evaluation metrics
            all_errors(g, ir, m) = norm(S_est - S, 'fro')^2 / nS;
            all_fscore(g, ir, m) = fscore(S, S_est);
            all_times(g, ir, m)  = Mtime;
            all_comm(g, ir, m)   = norm(C*S_est - S_est*C,'fro')^2;
        end
    end
end

toc(t1)
save('exp2_results_v1.mat');

%% ===================== PLOTTING RESULTS =====================
load("exp2_results_v1.mat");   % load previously saved results

% Compute mean and median over repetitions
mean_err    = squeeze(nanmean(all_errors, 1));   % [num_r x nM]
mean_f1     = squeeze(nanmean(all_fscore, 1));
mean_time   = squeeze(nanmean(all_times, 1));
mean_comm   = squeeze(nanmean(all_comm, 1));

median_err  = squeeze(nanmedian(all_errors, 1));
median_f1   = squeeze(nanmedian(all_fscore, 1));
median_time = squeeze(nanmedian(all_times, 1));
median_comm = squeeze(nanmedian(all_comm, 1));

% Plot aesthetics
line_styles = {'-','--',':','-.'};
markers     = {'^','s','o','d','*','v','>','<'};
colors      = lines(nM);

lw = 2; ms = 8; fs_axes = 16; fs_lbl = 18; fs_leg = 14;

% Figure: 2 rows x 4 columns (Mean | Median for Error, F1, Time, Comm)
figure('Position',[150 150 1800 900]);

% -------------------- Mean metrics --------------------
% Frobenius error
subplot(2,4,1);
for m = 1:nM
    loglog(r_values, mean_err(:, m), 'LineWidth', lw, 'LineStyle', line_styles{mod(m-1,numel(line_styles))+1}, ...
        'Marker', markers{mod(m-1,numel(markers))+1}, 'MarkerSize', ms, 'Color', colors(m,:));
    hold on
end
grid on; box on;
xlabel('Number of realizations $r$','Interpreter','latex','FontSize',fs_lbl);
ylabel('Frobenius error (mean)','Interpreter','latex','FontSize',fs_lbl);
title(sprintf('Mean Error vs $r$ (N=%d)', N), 'Interpreter','latex');
set(gca,'FontSize',fs_axes);

% F1-score
subplot(2,4,2);
for m = 1:nM
    semilogx(r_values, mean_f1(:, m), 'LineWidth', lw, 'LineStyle', line_styles{mod(m-1,numel(line_styles))+1}, ...
        'Marker', markers{mod(m-1,numel(markers))+1}, 'MarkerSize', ms, 'Color', colors(m,:));
    hold on
end
grid on; box on; ylim([0 1]); yticks(0:0.1:1);
xlabel('Number of realizations $r$','Interpreter','latex','FontSize',fs_lbl);
ylabel('F-score (mean)','Interpreter','latex','FontSize',fs_lbl);
title(sprintf('Mean F1-score vs $r$ (N=%d)', N), 'Interpreter','latex');
set(gca,'FontSize',fs_axes);

% Runtime
subplot(2,4,3);
for m = 1:nM
    loglog(r_values, mean_time(:, m), 'LineWidth', lw, 'LineStyle', line_styles{mod(m-1,numel(line_styles))+1}, ...
        'Marker', markers{mod(m-1,numel(markers))+1}, 'MarkerSize', ms, 'Color', colors(m,:));
    hold on
end
grid on; box on;
xlabel('Number of realizations $r$','Interpreter','latex','FontSize',fs_lbl);
ylabel('Runtime (s, mean)','Interpreter','latex','FontSize',fs_lbl);
title(sprintf('Mean Runtime vs $r$ (N=%d)', N), 'Interpreter','latex');
set(gca,'FontSize',fs_axes);

% Commutativity
subplot(2,4,4);
for m = 1:nM
    loglog(r_values, mean_comm(:, m), 'LineWidth', lw, 'LineStyle', line_styles{mod(m-1,numel(line_styles))+1}, ...
        'Marker', markers{mod(m-1,numel(markers))+1}, 'MarkerSize', ms, 'Color', colors(m,:));
    hold on
end
grid on; box on;
xlabel('Number of realizations $r$','Interpreter','latex','FontSize',fs_lbl);
ylabel('Comm (mean, $\|CS-SC\|_F^2$)','Interpreter','latex','FontSize',fs_lbl);
title(sprintf('Mean Comm vs $r$ (N=%d)', N), 'Interpreter','latex');
set(gca,'FontSize',fs_axes);

% -------------------- Median metrics --------------------
% Frobenius error
subplot(2,4,5);
for m = 1:nM
    loglog(r_values, median_err(:, m), 'LineWidth', lw, 'LineStyle', line_styles{mod(m-1,numel(line_styles))+1}, ...
        'Marker', markers{mod(m-1,numel(markers))+1}, 'MarkerSize', ms, 'Color', colors(m,:));
    hold on
end
grid on; box on;
xlabel('Number of realizations $r$','Interpreter','latex','FontSize',fs_lbl);
ylabel('Frobenius error (median)','Interpreter','latex','FontSize',fs_lbl);
title(sprintf('Median Error vs $r$ (N=%d)', N), 'Interpreter','latex');
set(gca,'FontSize',fs_axes);

% F1-score
subplot(2,4,6);
for m = 1:nM
    semilogx(r_values, median_f1(:, m), 'LineWidth', lw, 'LineStyle', line_styles{mod(m-1,numel(line_styles))+1}, ...
        'Marker', markers{mod(m-1,numel(markers))+1}, 'MarkerSize', ms, 'Color', colors(m,:));
    hold on
end
grid on; box on; ylim([0 1]); yticks(0:0.1:1);
xlabel('Number of realizations $r$','Interpreter','latex','FontSize',fs_lbl);
ylabel('F-score (median)','Interpreter','latex','FontSize',fs_lbl);
title(sprintf('Median F1-score vs $r$ (N=%d)', N), 'Interpreter','latex');
set(gca,'FontSize',fs_axes);

% Runtime
subplot(2,4,7);
for m = 1:nM
    loglog(r_values, median_time(:, m), 'LineWidth', lw, 'LineStyle', line_styles{mod(m-1,numel(line_styles))+1}, ...
        'Marker', markers{mod(m-1,numel(markers))+1}, 'MarkerSize', ms, 'Color', colors(m,:));
    hold on
end
grid on; box on;
xlabel('Number of realizations $r$','Interpreter','latex','FontSize',fs_lbl);
ylabel('Runtime (s, median)','Interpreter','latex','FontSize',fs_lbl);
title(sprintf('Median Runtime vs $r$ (N=%d)', N), 'Interpreter','latex');
set(gca,'FontSize',fs_axes);

% Commutativity
subplot(2,4,8);
for m = 1:nM
    loglog(r_values, median_comm(:, m), 'LineWidth', lw, 'LineStyle', line_styles{mod(m-1,numel(line_styles))+1}, ...
        'Marker', markers{mod(m-1,numel(markers))+1}, 'MarkerSize', ms, 'Color', colors(m,:));
    hold on
end
grid on; box on;
xlabel('Number of realizations $r$','Interpreter','latex','FontSize',fs_lbl);
ylabel('Comm (median, $\|CS-SC\|_F^2$)','Interpreter','latex','FontSize',fs_lbl);
title(sprintf('Median Comm vs $r$ (N=%d)', N), 'Interpreter','latex');
set(gca,'FontSize',fs_axes);

% -------------------- Legends --------------------
lg = legend(Models, 'Interpreter','latex', 'FontSize', fs_leg, ...
    'Location','southoutside','Orientation','horizontal');
set(lg,'Color','none');

%% -------- F1-score & Commutativity on same figure (mean) --------
figure()
yyaxis left
for m = 1:nM
    semilogx(r_values, mean_f1(:, m), 'LineWidth', lw, 'LineStyle','-', ...
        'Marker', markers{mod(m-1,numel(markers))+1}, 'MarkerSize', ms, 'Color', colors(m,:));
    hold on
end
ylim([0 1]); yticks(0:0.1:1);
ylabel('F-score','Interpreter','latex','FontSize',fs_lbl);

yyaxis right
for m = 1:nM
    loglog(r_values, mean_comm(:, m), 'LineWidth', lw, 'LineStyle',':', ...
        'Marker', markers{mod(m-1,numel(markers))+1}, 'MarkerSize', ms, 'Color', colors(m,:));
    hold on
end
ylim([1e-7 1e3]); yticks([1e-7,1e-5,1e-3,1e-1,1e1,1e3])
yticklabels({'10^{-7}','10^{-5}','10^{-3}','10^{-1}','10^1','10^3'})
ylabel('$\|CS-SC\|_F^2$','Interpreter','latex','FontSize',fs_lbl);

xlabel('Number of realizations','Interpreter','latex','FontSize',fs_lbl);
grid on; box on;
set(gca,'FontSize',fs_axes);
lg = legend({'ST','K-ST','SepK-ST'}, 'Location','best','Interpreter','latex','FontSize',12);
set(lg,'color','none');
xticks([1e2, 1e3, 1e4, 1e5, 1e6])
xticklabels({'10^2','10^3','10^4','10^5','10^6'})

