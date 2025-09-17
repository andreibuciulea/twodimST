clear; clc;
addpath(genpath("../opt"))
addpath(genpath("../utils"))

% ===================== PARAMETERS =====================
alpha = 1;
max_iters = 1e3;
lr = 1;
lambda = 1;
nG = 30;           % repetitions per combination (adjust if slow)
verbose = false;
L = 3;
r = 1e6;           % realizations (adjust if slow)

% Fixed graph size
s = 7; t = 7; N = s*t;

% Models
Models = ["pr1-cvx","pr2-cvx","pr3-cvx","pr4-cvx"];
nM = numel(Models);

% 2D sweep over ps, pt
pst_values = linspace(0.1,0.8,8);
nPst = numel(pst_values);

% Preallocations: [nPs x nPt x nG x nM]
all_errors =  nan(nPst, nG, nM);
all_fscore =  nan(nPst, nG, nM);
all_times  =  nan(nPst, nG, nM);
all_comm   =  nan(nPst, nG, nM);

% ===================== 2D SWEEP (ps, pt) =====================
fprintf('Starting 2D sweep: %d combinations, %d graphs each\n', nPst, nG);
tic
for ip = 1:nPst
    pst = pst_values(ip);
    fprintf('ps = %.3f | pt = %.3f\n', pst, pst);

    % parfor over repetitions g
    parfor g = 1:nG
        % generate graph and signals
        [S,Ss,St] = generate_sptemp_graph(s,t,pst,pst);
        out = generate_st_signals(Ss,St,L,r,verbose);
        C  = out.C; Cs = out.Cs; Ct = out.Ct;

        nS = norm(S,'fro')^2;
        errors_g = zeros(1,nM);
        fscore_g = zeros(1,nM);
        times_g  = zeros(1,nM);
        comm_g   = zeros(1,nM);

        for m = 1:nM
            params = struct('alpha',alpha,'lr',lr,'lambda',lambda,'verbose',verbose);
            [S_est, Mtime] = estimate_graph(Models(m), C, Cs, Ct, S, Ss, St, max_iters, params);

            errors_g(m) = norm(S_est - S,'fro')^2 / nS;
            fscore_g(m) = fscore(S, S_est);
            times_g(m)  = Mtime;
            comm_g(m)   = norm(C*S_est - S_est*C,'fro')^2;
        end

        all_errors(ip, g, :) = errors_g;
        all_fscore(ip, g, :) = fscore_g;
        all_times(ip, g, :)  = times_g;
        all_comm(ip, g, :)   = comm_g;
    end % parfor g
end
toc

%% ===================== ADAPTED PLOTTING (1D over pst) + COMM =====================
load("exp_3_ps_pt_2D_v1.mat")
% Aggregates (mean and median over repetitions g)
mean_err  = squeeze(nanmean(all_errors, 2));    % [nPst x nM]
med_err   = squeeze(nanmedian(all_errors, 2));

mean_f1   = squeeze(nanmean(all_fscore, 2));    % [nPst x nM]
med_f1    = squeeze(nanmedian(all_fscore, 2));

mean_time = squeeze(nanmean(all_times, 2));
med_time  = squeeze(nanmedian(all_times, 2));

mean_comm = squeeze(nanmean(all_comm, 2));
med_comm  = squeeze(nanmedian(all_comm, 2));

nM = numel(Models);
% Common aesthetics
line_styles = {'-','--',':','-.'};
markers = {'^','s','o','d','*','v','>','<'}; 
colors = lines(nM);
lw = 2; ms = 8; fs_axes = 14; fs_lbl = 16; fs_leg = 12;
outdir = 'figs_pspt_1D';
if ~exist(outdir,'dir'), mkdir(outdir); end

% X-axis ticks
xticks_idx = round(linspace(1,numel(pst_values), min(numel(pst_values),8)));
xticks_vals = pst_values(xticks_idx);

% Figure: 2 rows x 4 columns (Error | F1 | Time | Comm)
figure('Name','Metrics (mean & median)','Position',[100 100 1800 700]);

% --- Error (mean) ---
subplot(2,4,1);
for m = 1:nM
    semilogy(pst_values, mean_err(:,m), 'LineWidth', lw, ...
        'LineStyle', line_styles{1+mod(m-1,numel(line_styles))}, ...
        'Marker', markers{1+mod(m-1,numel(markers))}, 'MarkerSize', ms, ...
        'Color', colors(m,:));
    hold on
end
grid on; box on;
xlabel('$p_s=p_t$','Interpreter','latex','FontSize',fs_lbl);
ylabel('Frobenius error (mean)','Interpreter','latex','FontSize',fs_lbl);
title('Mean Error','Interpreter','latex','FontSize',fs_lbl);
set(gca,'XTick',xticks_vals,'FontSize',fs_axes);

% --- F1-score (mean) ---
subplot(2,4,2);
for m = 1:nM
    semilogy(pst_values, mean_f1(:,m), 'LineWidth', lw, ...
        'LineStyle', line_styles{1+mod(m-1,numel(line_styles))}, ...
        'Marker', markers{1+mod(m-1,numel(markers))}, 'MarkerSize', ms, ...
        'Color', colors(m,:));
    hold on
end
grid on; box on; ylim([0 1]); yticks(0:0.1:1);
xlabel('$p_s=p_t$','Interpreter','latex','FontSize',fs_lbl);
ylabel('F-score (mean)','Interpreter','latex','FontSize',fs_lbl);
title('Mean F1','Interpreter','latex','FontSize',fs_lbl);
set(gca,'XTick',xticks_vals,'FontSize',fs_axes);

% --- Runtime (mean) ---
subplot(2,4,3);
for m = 1:nM
    semilogy(pst_values, mean_time(:,m), 'LineWidth', lw, ...
        'LineStyle', line_styles{1+mod(m-1,numel(line_styles))}, ...
        'Marker', markers{1+mod(m-1,numel(markers))}, 'MarkerSize', ms, ...
        'Color', colors(m,:));
    hold on
end
grid on; box on;
xlabel('$p_s=p_t$','Interpreter','latex','FontSize',fs_lbl);
ylabel('Runtime (s, mean)','Interpreter','latex','FontSize',fs_lbl);
title('Mean Runtime','Interpreter','latex','FontSize',fs_lbl);
set(gca,'XTick',xticks_vals,'FontSize',fs_axes);

% --- Commutativity (mean) ---
subplot(2,4,4);
for m = 1:nM
    semilogy(pst_values, mean_comm(:,m), 'LineWidth', lw, ...
        'LineStyle', line_styles{1+mod(m-1,numel(line_styles))}, ...
        'Marker', markers{1+mod(m-1,numel(markers))}, 'MarkerSize', ms, ...
        'Color', colors(m,:));
    hold on
end
grid on; box on;
xlabel('$p_s=p_t$','Interpreter','latex','FontSize',fs_lbl);
ylabel('Comm (mean, $\|CS- SC\|_F^2$)','Interpreter','latex','FontSize',fs_lbl);
title('Mean Comm','Interpreter','latex','FontSize',fs_lbl);
set(gca,'XTick',xticks_vals,'FontSize',fs_axes);

% --- Error (median) ---
subplot(2,4,5);
for m = 1:nM
    semilogy(pst_values, med_err(:,m), 'LineWidth', lw, ...
        'LineStyle', line_styles{1+mod(m-1,numel(line_styles))}, ...
        'Marker', markers{1+mod(m-1,numel(markers))}, 'MarkerSize', ms, ...
        'Color', colors(m,:));
    hold on
end
grid on; box on;
xlabel('$p_s=p_t$','Interpreter','latex','FontSize',fs_lbl);
ylabel('Frobenius error (median)','Interpreter','latex','FontSize',fs_lbl);
title('Median Error','Interpreter','latex','FontSize',fs_lbl);
set(gca,'XTick',xticks_vals,'FontSize',fs_axes);

% --- F1-score (median) ---
subplot(2,4,6);
for m = 1:nM
    plot(pst_values, med_f1(:,m), 'LineWidth', lw, ...
        'LineStyle', line_styles{1+mod(m-1,numel(line_styles))}, ...
        'Marker', markers{1+mod(m-1,numel(markers))}, 'MarkerSize', ms, ...
        'Color', colors(m,:));
    hold on
end
grid on; box on; ylim([0 1]); yticks(0:0.1:1);
xlabel('$p_s=p_t$','Interpreter','latex','FontSize',fs_lbl);
ylabel('F-score (median)','Interpreter','latex','FontSize',fs_lbl);
title('Median F1','Interpreter','latex','FontSize',fs_lbl);
set(gca,'XTick',xticks_vals,'FontSize',fs_axes);

% --- Runtime (median) ---
subplot(2,4,7);
for m = 1:nM
    semilogy(pst_values, med_time(:,m), 'LineWidth', lw, ...
        'LineStyle', line_styles{1+mod(m-1,numel(line_styles))}, ...
        'Marker', markers{1+mod(m-1,numel(markers))}, 'MarkerSize', ms, ...
        'Color', colors(m,:));
    hold on
end
grid on; box on;
xlabel('$p_s=p_t$','Interpreter','latex','FontSize',fs_lbl);
ylabel('Runtime (s, median)','Interpreter','latex','FontSize',fs_lbl);
title('Median Runtime','Interpreter','latex','FontSize',fs_lbl);
set(gca,'XTick',xticks_vals,'FontSize',fs_axes);

% --- Commutativity (median) ---
subplot(2,4,8);
for m = 1:nM
    semilogy(pst_values, med_comm(:,m), 'LineWidth', lw, ...
        'LineStyle', line_styles{1+mod(m-1,numel(line_styles))}, ...
        'Marker', markers{1+mod(m-1,numel(markers))}, 'MarkerSize', ms, ...
        'Color', colors(m,:));
    hold on
end
grid on; box on;
xlabel('$p_s=p_t$','Interpreter','latex','FontSize',fs_lbl);
ylabel('Comm (median, $\|CS- SC\|_F^2$)','Interpreter','latex','FontSize',fs_lbl);
title('Median Comm','Interpreter','latex','FontSize',fs_lbl);

% Legend (single, outside subplots)
hL = legend(Models,'Interpreter','latex','FontSize',fs_leg,'Orientation','horizontal');
hL.Position = [0.25 0.01 0.5 0.03]; % adjust bottom position
set(hL,'Color','none');

sgtitle(sprintf('Metrics vs $p_s=p_t$ (N=%d) — mean (row 1) / median (row 2)', N),'Interpreter','latex','FontSize',16);
