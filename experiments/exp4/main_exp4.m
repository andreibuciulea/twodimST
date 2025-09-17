clear; clc;
addpath(genpath("../../opt"))
addpath(genpath("../../utils"))

nG = 100;
ps = 0.3; % connection probability in space
pt = 0.3; % connection probability in time
r = 1e6; % number of realizations
sig_type = 'ST';
L = 3;
alpha = 1;
max_iters = 1e3;
lr = 1;
lambda = 1;
verbose = false;

Models = {'pr3-cvx','DNNLasso','TERRALasso'};
nM = numel(Models);
prms = struct('alpha',alpha,'lr',lr,'lambda',lambda,'verbose',verbose);

% Graph sizes to test
graph_sizes = [7 7; 8 8; 9 9; 10 10; 11 11; 12 12; 13 13; 14 14; 15 15; 16 16; 17 17; 18 18; 19 19; 20 20];
num_sizes = size(graph_sizes,1);

% Preallocate results
all_errors = zeros(num_sizes, nG, nM);
all_fscore = zeros(num_sizes, nG, nM);
all_times  = zeros(num_sizes, nG, nM);
all_comm   = zeros(num_sizes, nG, nM);

tic
for sz = 1:num_sizes
    s = graph_sizes(sz,1);
    t = graph_sizes(sz,2);
    N = s*t;
    fprintf('Running experiments for graph size s=%d, t=%d (N=%d)\n', s, t, N);

    % Local accumulators for parfor
    local_err  = zeros(nG,nM);
    local_fsc  = zeros(nG,nM);
    local_time = zeros(nG,nM);
    local_comm = zeros(nG,nM);

    parfor g = 1:nG
        [S,Ss,St] = generate_sptemp_graph(s,t,ps,pt);
        nS = norm(S,'fro')^2;
        %out = generate_st_signals(Ss,St,L,r,verbose);
        out = generate_gauss_signals(Ss,St,L,r,verbose);

        C  = out.C_sampled;
        Cs = out.Cs_sampled;
        Ct = out.Ct_sampled;

        for m = 1:nM
            [S_est, mtime] = estimate_graph(Models{m},C,Cs,Ct,S,Ss,St,max_iters,prms);
            local_err(g,m)  = norm(S_est - S,'fro')^2 / nS;
            local_fsc(g,m)  = fscore(S, S_est);
            local_time(g,m) = mtime;
            local_comm(g,m) = norm(C*S_est - S_est*C,'fro')^2;
        end
    end

    % Store in global arrays
    all_errors(sz,:,:) = local_err;
    all_fscore(sz,:,:) = local_fsc;
    all_times(sz,:,:)  = local_time;
    all_comm(sz,:,:)   = local_comm;
end
toc

% Compute mean metrics across graphs
mean_errors = squeeze(mean(all_errors,2));
mean_fscore = squeeze(mean(all_fscore,2));
mean_times  = squeeze(mean(all_times,2));
mean_comm   = squeeze(mean(all_comm,2));

% Save everything
save('results_gauss_v1.mat','all_errors','all_fscore','all_times','all_comm', ...
                   'mean_errors','mean_fscore','mean_times','mean_comm');

%%
% Compute mean metrics across graphs
mean_errors = squeeze(nanmean(all_errors,2));
mean_fscore = squeeze(nanmean(all_fscore,2));
mean_times  = squeeze(nanmean(all_times,2));
mean_comm   = squeeze(nanmean(all_comm,2));
% =========================
% Plot results
% =========================
xvals = 1:num_sizes; % index of graph sizes
labels = arrayfun(@(i) sprintf('%dx%d', graph_sizes(i,1), graph_sizes(i,2)), ...
                  1:num_sizes, 'UniformOutput', false);

% ---- Error ----
figure;
hold on;
for m = 1:nM
    plot(xvals, mean_errors(:,m), '-o', 'LineWidth', 1.5);
end
hold off;
xticks(xvals);
xticklabels(labels);
xlabel('Graph size (s x t)');
ylabel('Mean Error');
legend(Models, 'Location', 'best');
title('Error vs Graph Size');
grid on;

% ---- F-score ----
figure;
hold on;
for m = 1:nM
    plot(xvals, mean_fscore(:,m), '-s', 'LineWidth', 1.5);
end
hold off;
xticks(xvals);
xticklabels(labels);
xlabel('Graph size (s x t)');
ylabel('Mean F-score');
legend(Models, 'Location', 'best');
title('F-score vs Graph Size');
grid on;

% ---- Time ----
figure;
hold on;
for m = 1:nM
    plot(xvals, mean_times(:,m), '-d', 'LineWidth', 1.5);
end
hold off;
xticks(xvals);
xticklabels(labels);
xlabel('Graph size (s x t)');
ylabel('Mean Time (s)');
legend(Models, 'Location', 'best');
title('Time vs Graph Size');
grid on;

%%
graph_sizes = [7 7; 8 8; 9 9; 10 10; 11 11; 12 12; 13 13; 14 14; 15 15; 16 16; 17 17; 18 18; 19 19; 20 20];
xvals = prod(graph_sizes,2);
%graph_sizes = [5 5; 5 6; 6 6];
num_sizes = size(graph_sizes,1);
nM = 3;
Models = {'SepK-ST (MRF)','DNNLasso (MRF)','TERRALasso (MRF)','SepK-ST (Poly)','DNNLasso (Poly)','TERRALasso (Poly)'};
% List of result files and associated line styles
files = {'results_gauss.mat', 'results_L3.mat'};
lineStyles = {'-', '--', ':'};   % gauss = dotted, L2 = dashed, L3 = solid
markers = {'+','*','s'};

% Colors for the three models (consistent across datasets)
colors = lines(3);  % or define custom RGB values
my_range = 1:9;

% Loop over files
for f = 1:length(files)
    load(files{f});  % loads all_errors, all_fscore, etc.

    % Compute mean metrics across graphs
    mean_errors = squeeze(nanmean(all_errors,2));
    mean_fscore = squeeze(nanmean(all_fscore,2));

    % Plot Error
    figure(1);
    for m = 1:nM
        semilogy(xvals(my_range), mean_errors(my_range,m), ...
            'LineStyle', lineStyles{f}, ...
            'Color', colors(m,:), ...
            'LineWidth', 2, ...
            'Marker',markers{m}, ...
            'MarkerSize', 8);
        hold on;
    end

    % Plot F-score
    figure(2);
    for m = 1:nM
        plot(xvals(my_range), mean_fscore(my_range,m), ...
            'LineStyle', lineStyles{f}, ...
            'Color', colors(m,:), ...
            'LineWidth', 2,...
            'Marker',markers{m}, ...
            'MarkerSize', 8);
        hold on;
    end
end

% Add legends (same for both figures)
figure(1);
lg = legend(Models, 'Location', 'best','Interpreter','latex','FontSize',12,'NumColumns',2);
set(lg,'color','none');
grid on;
xlabel('Number of nodes N','Interpreter','latex','FontSize',18)
ylabel('Error','Interpreter','latex','FontSize',18)
xlim([49 225])
set(gca,'FontSize',16);
figure(2);
lg = legend(Models, 'Location', 'best','Interpreter','latex','FontSize',12,'NumColumns',2);
set(lg,'color','none');
grid on;
xlabel('Number of nodes N','Interpreter','latex','FontSize',18)
ylabel('F-score','Interpreter','latex','FontSize',18)
xlim([49 225])
set(gca,'FontSize',16);
xticks([50, 75, 100, 125, 150, 175, 200, 225])
xticklabels({'50','75','100','125','150','175','200','225'})

