%% Master of Quantitative Finance
% Approximation of processes: MATLAB Implementation (Enhanced Version)
% 
% This script implements the numerical methods described in the theoretical
% solutions for the tp_DISCR_M2QF.pdf assignment.

clear all;
close all;
clc;


%% Parameters
% Define common parameters used throughout the script
x = 100;          % Initial value
b = 0.05;         % Drift parameter
sigma = 0.2;      % Volatility parameter
T = 1;            % Time horizon
seed = 42;        % Random seed for reproducibility
% Use Octave-compatible random seed setting
rand("state", seed);
randn("state", seed);

%% Exercise 1: Weak and strong error

%% 1. Explicit solution of the SDE
% X_t = x + \int_0^t bX_s ds + \int_0^t \sigma X_s dW_s
% Solution: X_t = x * exp[(b - sigma^2/2)t + sigma*W_t]

% Function to compute the exact solution
exactSolution = @(x, b, sigma, t, W_t) x * exp((b - sigma^2/2)*t + sigma*W_t);

%% 2. Simulate a path using explicit formula and Euler scheme
% Time steps for different discretizations
h_values = [2^-4, 2^-8, 2^-10];

% Create figure for plotting
figure('Position', [100, 100, 1200, 800]);

for i = 1:length(h_values)
    h = h_values(i);
    N = T/h;
    t = 0:h:T;
    
    % Generate Brownian motion
    dW = sqrt(h) * randn(1, N);
    W = [0, cumsum(dW)];
    
    % Exact solution
    X_exact = zeros(1, N+1);
    X_exact(1) = x;
    for j = 2:N+1
        X_exact(j) = exactSolution(x, b, sigma, t(j), W(j));
    end
    
    % Euler scheme
    X_euler = zeros(1, N+1);
    X_euler(1) = x;
    for j = 1:N
        X_euler(j+1) = X_euler(j) + b*X_euler(j)*h + sigma*X_euler(j)*dW(j);
    end
    
    % Plot results
    subplot(length(h_values), 1, i);
    plot(t, X_exact, 'b-', 'LineWidth', 2, 'DisplayName', 'Exact Solution');
    hold on;
    plot(t, X_euler, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Euler Scheme');
    title(sprintf('Comparison: Euler Scheme (h = 2^{-%d})', log2(1/h)), 'FontSize', 12);
    xlabel('Time', 'FontSize', 10);
    ylabel('X_t', 'FontSize', 10);
    legend('Location', 'best', 'FontSize', 9);
    grid on;
    
    % Add error information
    max_error = max(abs(X_exact - X_euler));
    text(0.05, 0.9, sprintf('Max Error: %.4f', max_error), 'Units', 'normalized', 'FontSize', 9);
end

% Add overall title
title('Comparison of Exact Solution and Euler Scheme', 'FontSize', 14);

% Save the figure
print -dpng 'euler_comparison.png';

%% 3. Milstein scheme for general dynamics

%% 3a. Simulation of the stochastic integral
% \int_{t_k}^{t_{k+1}} (W_s - W_{t_k})dW_s = 0.5*[(W_{t_{k+1}} - W_{t_k})^2 - (t_{k+1} - t_k)]

% Function to simulate the stochastic integral
simulateStochasticIntegral = @(dW, h) 0.5 * (dW.^2 - h);

%% 3b. Milstein scheme implementation
% X_{t_{k+1}} = X_{t_k} + b(X_{t_k})*h + sigma(X_{t_k})*dW_k + 0.5*sigma'(X_{t_k})*sigma(X_{t_k})*[(dW_k)^2 - h]

% For our specific SDE, sigma(x) = sigma*x, so sigma'(x) = sigma
% Therefore, sigma'(X_{t_k})*sigma(X_{t_k}) = sigma^2*X_{t_k}

% Implement Milstein scheme
h = 2^-8;  % Moderate step size for visualization
N = T/h;
t = 0:h:T;

% Generate Brownian motion
dW = sqrt(h) * randn(1, N);
W = [0, cumsum(dW)];

% Exact solution
X_exact = zeros(1, N+1);
X_exact(1) = x;
for j = 2:N+1
    X_exact(j) = exactSolution(x, b, sigma, t(j), W(j));
end

% Euler scheme
X_euler = zeros(1, N+1);
X_euler(1) = x;
for j = 1:N
    X_euler(j+1) = X_euler(j) + b*X_euler(j)*h + sigma*X_euler(j)*dW(j);
end

% Milstein scheme
X_milstein = zeros(1, N+1);
X_milstein(1) = x;
for j = 1:N
    stochInt = simulateStochasticIntegral(dW(j), h);
    X_milstein(j+1) = X_milstein(j) + b*X_milstein(j)*h + sigma*X_milstein(j)*dW(j) + ...
                      0.5*sigma^2*X_milstein(j)*stochInt;
end

% Plot results
figure('Position', [100, 100, 1200, 600]);

% Plot full paths
subplot(2, 1, 1);
plot(t, X_exact, 'b-', 'LineWidth', 2, 'DisplayName', 'Exact Solution');
hold on;
plot(t, X_euler, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Euler Scheme');
plot(t, X_milstein, 'g-.', 'LineWidth', 1.5, 'DisplayName', 'Milstein Scheme');
title('Comparison of Exact Solution, Euler Scheme, and Milstein Scheme', 'FontSize', 12);
xlabel('Time', 'FontSize', 10);
ylabel('X_t', 'FontSize', 10);
legend('Location', 'best', 'FontSize', 9);
grid on;

% Plot zoomed section to better see differences
subplot(2, 1, 2);
% Find a good region to zoom in (where differences are visible)
zoom_start = floor(N/2);
zoom_end = zoom_start + floor(N/10);
zoom_t = t(zoom_start:zoom_end);
plot(zoom_t, X_exact(zoom_start:zoom_end), 'b-', 'LineWidth', 2, 'DisplayName', 'Exact Solution');
hold on;
plot(zoom_t, X_euler(zoom_start:zoom_end), 'r--', 'LineWidth', 1.5, 'DisplayName', 'Euler Scheme');
plot(zoom_t, X_milstein(zoom_start:zoom_end), 'g-.', 'LineWidth', 1.5, 'DisplayName', 'Milstein Scheme');
title('Zoomed View of Approximation Methods', 'FontSize', 12);
xlabel('Time', 'FontSize', 10);
ylabel('X_t', 'FontSize', 10);
legend('Location', 'best', 'FontSize', 9);
grid on;

% Add error information
euler_error = max(abs(X_exact - X_euler));
milstein_error = max(abs(X_exact - X_milstein));
text(0.05, 0.9, sprintf('Max Euler Error: %.6f', euler_error), 'Units', 'normalized', 'FontSize', 9);
text(0.05, 0.8, sprintf('Max Milstein Error: %.6f', milstein_error), 'Units', 'normalized', 'FontSize', 9);
text(0.05, 0.7, sprintf('Improvement: %.2f%%', 100*(euler_error-milstein_error)/euler_error), 'Units', 'normalized', 'FontSize', 9);

% Save the figure
print -dpng 'milstein_comparison.png';

%% 3c-d. Strong error analysis
% Compute the strong error (supremum of the difference on the whole path)

% Function to compute the strong error
computeStrongError = @(X_approx, X_exact) max(abs(X_approx - X_exact));

% Compute strong errors for different step sizes
h_values = [2^-4, 2^-5, 2^-6, 2^-7, 2^-8, 2^-9, 2^-10];
euler_errors = zeros(1, length(h_values));
milstein_errors = zeros(1, length(h_values));

% Initialize progress tracking
fprintf('Computing strong errors for different step sizes:\n');

for i = 1:length(h_values)
    h = h_values(i);
    N = T/h;
    fprintf('  Processing h = 2^-%d (%d/%d)...\n', log2(1/h), i, length(h_values));
    
    % Generate multiple paths for more reliable results
    num_paths = 10;
    euler_path_errors = zeros(1, num_paths);
    milstein_path_errors = zeros(1, num_paths);
    
    for path = 1:num_paths
        t = 0:h:T;
        
        % Generate Brownian motion
        dW = sqrt(h) * randn(1, N);
        W = [0, cumsum(dW)];
        
        % Exact solution
        X_exact = zeros(1, N+1);
        X_exact(1) = x;
        for j = 2:N+1
            X_exact(j) = exactSolution(x, b, sigma, t(j), W(j));
        end
        
        % Euler scheme
        X_euler = zeros(1, N+1);
        X_euler(1) = x;
        for j = 1:N
            X_euler(j+1) = X_euler(j) + b*X_euler(j)*h + sigma*X_euler(j)*dW(j);
        end
        
        % Milstein scheme
        X_milstein = zeros(1, N+1);
        X_milstein(1) = x;
        for j = 1:N
            stochInt = simulateStochasticIntegral(dW(j), h);
            X_milstein(j+1) = X_milstein(j) + b*X_milstein(j)*h + sigma*X_milstein(j)*dW(j) + ...
                              0.5*sigma^2*X_milstein(j)*stochInt;
        end
        
        % Compute strong errors for this path
        euler_path_errors(path) = computeStrongError(X_euler, X_exact);
        milstein_path_errors(path) = computeStrongError(X_milstein, X_exact);
    end
    
    % Average errors across paths
    euler_errors(i) = mean(euler_path_errors);
    milstein_errors(i) = mean(milstein_path_errors);
end

% Plot strong errors
figure('Position', [100, 100, 800, 600]);
loglog(h_values, euler_errors, 'ro-', 'LineWidth', 2, 'DisplayName', 'Euler Scheme');
hold on;
loglog(h_values, milstein_errors, 'bs-', 'LineWidth', 2, 'DisplayName', 'Milstein Scheme');

% Add reference lines for convergence rates
h_ref = logspace(log10(min(h_values)), log10(max(h_values)), 100);
euler_ref = h_ref.^0.5 * euler_errors(end) / h_values(end)^0.5;
milstein_ref = h_ref * milstein_errors(end) / h_values(end);
loglog(h_ref, euler_ref, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Order 0.5');
loglog(h_ref, milstein_ref, 'b--', 'LineWidth', 1.5, 'DisplayName', 'Order 1');

title('Strong Error Convergence', 'FontSize', 14);
xlabel('Step Size (h)', 'FontSize', 12);
ylabel('Strong Error', 'FontSize', 12);
legend('Location', 'best', 'FontSize', 10);
grid on;

% Add explanation text
text(0.05, 0.15, 'Euler scheme: O(h^{0.5}) convergence', 'Units', 'normalized', 'FontSize', 10);
text(0.05, 0.10, 'Milstein scheme: O(h) convergence', 'Units', 'normalized', 'FontSize', 10);

% Save the figure
print -dpng 'strong_error.png';

%% 4. Compute E[X_1] and analyze convergence

% Theoretical value of E[X_1]
E_X1_theoretical = x * exp(b*T);
fprintf('\nTheoretical value of E[X_1]: %.6f\n', E_X1_theoretical);

% Parameters for Monte Carlo simulation
n_values = [10^2, 10^3, 10^4, 10^5];
N_values = [10, 100, 500, 1000];
m = 20;  % Number of independent realizations

% Initialize arrays to store results
euler_errors_mc = zeros(length(n_values), length(N_values));
milstein_errors_mc = zeros(length(n_values), length(N_values));

% Initialize progress tracking
fprintf('Computing weak errors for different combinations of n and N:\n');

% Optimized Monte Carlo simulation - vectorized implementation where possible
for i_n = 1:length(n_values)
    n = n_values(i_n);
    
    for i_N = 1:length(N_values)
        N = N_values(i_N);
        h = T/N;
        
        fprintf('  Processing n = %d, N = %d (%d/%d, %d/%d)...\n', n, N, i_n, length(n_values), i_N, length(N_values));
        
        % Initialize arrays for independent realizations
        E_n_N_euler = zeros(1, m);
        E_n_N_milstein = zeros(1, m);
        
        % Perform m independent realizations
        for i_m = 1:m
            % Pre-allocate arrays for n independent paths
            X1_euler = zeros(1, n);
            X1_milstein = zeros(1, n);
            
            % Generate all random numbers at once for efficiency
            dW_all = sqrt(h) * randn(n, N);
            
            % Simulate n independent paths
            for i_path = 1:n
                dW = dW_all(i_path, :);
                
                % Euler scheme
                X_euler = x;
                for j = 1:N
                    X_euler = X_euler + b*X_euler*h + sigma*X_euler*dW(j);
                end
                X1_euler(i_path) = X_euler;
                
                % Milstein scheme
                X_milstein = x;
                for j = 1:N
                    stochInt = simulateStochasticIntegral(dW(j), h);
                    X_milstein = X_milstein + b*X_milstein*h + sigma*X_milstein*dW(j) + ...
                                0.5*sigma^2*X_milstein*stochInt;
                end
                X1_milstein(i_path) = X_milstein;
            end
            
            % Compute E_n_N for this realization
            E_n_N_euler(i_m) = mean(X1_euler);
            E_n_N_milstein(i_m) = mean(X1_milstein);
        end
        
        % Compute average absolute error
        euler_errors_mc(i_n, i_N) = mean(abs(E_n_N_euler - E_X1_theoretical));
        milstein_errors_mc(i_n, i_N) = mean(abs(E_n_N_milstein - E_X1_theoretical));
    end
end

% Plot results for different values of n
figure('Position', [100, 100, 1200, 800]);

for i_n = 1:length(n_values)
    subplot(2, 2, i_n);
    loglog(1./N_values, euler_errors_mc(i_n,:), 'ro-', 'LineWidth', 2, 'DisplayName', 'Euler Scheme');
    hold on;
    loglog(1./N_values, milstein_errors_mc(i_n,:), 'bs-', 'LineWidth', 2, 'DisplayName', 'Milstein Scheme');
    
    % Add reference lines for convergence rates
    h_ref = logspace(log10(min(1./N_values)), log10(max(1./N_values)), 100);
    euler_ref = h_ref * euler_errors_mc(i_n,end) / (1/N_values(end));
    milstein_ref = h_ref.^2 * milstein_errors_mc(i_n,end) / (1/N_values(end))^2;
    loglog(h_ref, euler_ref, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Order 1');
    loglog(h_ref, milstein_ref, 'b--', 'LineWidth', 1.5, 'DisplayName', 'Order 2');
    
    title(sprintf('Weak Error Convergence (n = %d)', n_values(i_n)), 'FontSize', 12);
    xlabel('Step Size (h)', 'FontSize', 10);
    ylabel('Weak Error', 'FontSize', 10);
    legend('Location', 'best', 'FontSize', 9);
    grid on;
end

% Add overall title
sgtitle('Weak Error Convergence for Different Sample Sizes', 'FontSize', 14);

% Save the figure
print -dpng 'weak_error_all.png';

% Plot results for n = 10^4 (most reliable)
figure('Position', [100, 100, 800, 600]);
loglog(1./N_values, euler_errors_mc(3,:), 'ro-', 'LineWidth', 2, 'DisplayName', 'Euler Scheme');
hold on;
loglog(1./N_values, milstein_errors_mc(3,:), 'bs-', 'LineWidth', 2, 'DisplayName', 'Milstein Scheme');

% Add reference lines for convergence rates
h_ref = logspace(log10(min(1./N_values)), log10(max(1./N_values)), 100);
euler_ref = h_ref * euler_errors_mc(3,end) / (1/N_values(end));
milstein_ref = h_ref.^2 * milstein_errors_mc(3,end) / (1/N_values(end))^2;
loglog(h_ref, euler_ref, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Order 1');
loglog(h_ref, milstein_ref, 'b--', 'LineWidth', 1.5, 'DisplayName', 'Order 2');

title('Weak Error Convergence (n = 10^4)', 'FontSize', 14);
xlabel('Step Size (h)', 'FontSize', 12);
ylabel('Weak Error', 'FontSize', 12);
legend('Location', 'best', 'FontSize', 10);
grid on;

% Add explanation text
text(0.05, 0.15, 'Euler scheme: O(h) weak convergence', 'Units', 'normalized', 'FontSize', 10);
text(0.05, 0.10, 'Milstein scheme: O(h^2) weak convergence', 'Units', 'normalized', 'FontSize', 10);

% Save the figure
print -dpng 'weak_error.png';

%% Exercise 2: Sensitivity of option prices through the Monte-Carlo method

%% Parameters for Black-Scholes model
S0 = 100;         % Initial stock price
K = 100;          % Strike price
r = 0.02;         % Risk-free rate
sigma = 0.35;     % Volatility
T = 1;            % Time to maturity
M = 10000;        % Number of Monte Carlo paths

%% Black-Scholes formula implementation
function [call_price, delta, gamma] = blackScholesFormula(S, K, r, sigma, T)
    % Compute d1 and d2
    d1 = (log(S/K) + (r + 0.5*sigma^2)*T) / (sigma*sqrt(T));
    d2 = d1 - sigma*sqrt(T);
    
    % Compute call price
    call_price = S * normcdf(d1) - K * exp(-r*T) * normcdf(d2);
    
    % Compute Greeks
    delta = normcdf(d1);
    gamma = normpdf(d1) / (S * sigma * sqrt(T));
end

%% 1. Finite difference estimator

%% 1.1-1.2. Convergence of finite difference estimator
% Theoretical values from Black-Scholes formula
[call_price, delta_bs, gamma_bs] = blackScholesFormula(S0, K, r, sigma, T);
fprintf('\nBlack-Scholes Call Price: %.6f\n', call_price);
fprintf('Black-Scholes Delta: %.6f\n', delta_bs);
fprintf('Black-Scholes Gamma: %.6f\n', gamma_bs);

%% 1.3. Implement finite difference estimator
% a) Using the same Gaussian realizations
epsilon_values = [5, 1, 0.5, 0.1, 0.05, 0.01];
delta_fd_same = zeros(1, length(epsilon_values));
gamma_fd_same = zeros(1, length(epsilon_values));
delta_fd_var_same = zeros(1, length(epsilon_values));
gamma_fd_var_same = zeros(1, length(epsilon_values));

% b) Using independent realizations
delta_fd_indep = zeros(1, length(epsilon_values));
gamma_fd_indep = zeros(1, length(epsilon_values));
delta_fd_var_indep = zeros(1, length(epsilon_values));
gamma_fd_var_indep = zeros(1, length(epsilon_values));

% Initialize progress tracking
fprintf('Computing finite difference estimators for different epsilon values:\n');

for i = 1:length(epsilon_values)
    epsilon = epsilon_values(i);
    fprintf('  Processing epsilon = %.3f (%d/%d)...\n', epsilon, i, length(epsilon_values));
    
    % a) Using the same Gaussian realizations
    Z = randn(M, 1);
    
    % Simulate terminal stock prices
    ST_plus = (S0 + epsilon) * exp((r - 0.5*sigma^2)*T + sigma*sqrt(T)*Z);
    ST = S0 * exp((r - 0.5*sigma^2)*T + sigma*sqrt(T)*Z);
    ST_minus = (S0 - epsilon) * exp((r - 0.5*sigma^2)*T + sigma*sqrt(T)*Z);
    
    % Compute option payoffs
    payoff_plus = max(ST_plus - K, 0);
    payoff = max(ST - K, 0);
    payoff_minus = max(ST_minus - K, 0);
    
    % Compute finite difference estimators
    delta_fd_same(i) = mean(payoff_plus - payoff_minus) / (2*epsilon);
    gamma_fd_same(i) = mean(payoff_plus - 2*payoff + payoff_minus) / (epsilon^2);
    
    % Compute variances
    delta_fd_var_same(i) = var((payoff_plus - payoff_minus) / (2*epsilon)) / M;
    gamma_fd_var_same(i) = var((payoff_plus - 2*payoff + payoff_minus) / (epsilon^2)) / M;
    
    % b) Using independent realizations
    Z_plus = randn(M, 1);
    Z = randn(M, 1);
    Z_minus = randn(M, 1);
    
    % Simulate terminal stock prices
    ST_plus = (S0 + epsilon) * exp((r - 0.5*sigma^2)*T + sigma*sqrt(T)*Z_plus);
    ST = S0 * exp((r - 0.5*sigma^2)*T + sigma*sqrt(T)*Z);
    ST_minus = (S0 - epsilon) * exp((r - 0.5*sigma^2)*T + sigma*sqrt(T)*Z_minus);
    
    % Compute option payoffs
    payoff_plus = max(ST_plus - K, 0);
    payoff = max(ST - K, 0);
    payoff_minus = max(ST_minus - K, 0);
    
    % Compute finite difference estimators
    delta_fd_indep(i) = mean(payoff_plus - payoff_minus) / (2*epsilon);
    gamma_fd_indep(i) = mean(payoff_plus - 2*payoff + payoff_minus) / (epsilon^2);
    
    % Compute variances
    delta_fd_var_indep(i) = (var(payoff_plus) + var(payoff_minus)) / M / (2*epsilon)^2;
    gamma_fd_var_indep(i) = (var(payoff_plus) + 4*var(payoff) + var(payoff_minus)) / M / (epsilon^2)^2;
end

% Plot results for Delta
figure('Position', [100, 100, 1200, 500]);

subplot(1, 2, 1);
semilogx(epsilon_values, delta_fd_same, 'ro-', 'LineWidth', 2, 'DisplayName', 'Same Realizations');
hold on;
semilogx(epsilon_values, delta_fd_indep, 'bs-', 'LineWidth', 2, 'DisplayName', 'Independent Realizations');
semilogx(epsilon_values, delta_bs*ones(size(epsilon_values)), 'k--', 'LineWidth', 1.5, 'DisplayName', 'Black-Scholes Delta');
title('Finite Difference Estimator for Delta', 'FontSize', 12);
xlabel('Epsilon', 'FontSize', 10);
ylabel('Delta', 'FontSize', 10);
legend('Location', 'best', 'FontSize', 9);
grid on;

% Add error information
[~, best_idx_same] = min(abs(delta_fd_same - delta_bs));
[~, best_idx_indep] = min(abs(delta_fd_indep - delta_bs));
text(0.05, 0.15, sprintf('Best epsilon (same): %.3f', epsilon_values(best_idx_same)), 'Units', 'normalized', 'FontSize', 9);
text(0.05, 0.10, sprintf('Best epsilon (indep): %.3f', epsilon_values(best_idx_indep)), 'Units', 'normalized', 'FontSize', 9);

subplot(1, 2, 2);
loglog(epsilon_values, delta_fd_var_same, 'ro-', 'LineWidth', 2, 'DisplayName', 'Same Realizations');
hold on;
loglog(epsilon_values, delta_fd_var_indep, 'bs-', 'LineWidth', 2, 'DisplayName', 'Independent Realizations');
title('Variance of Delta Estimator', 'FontSize', 12);
xlabel('Epsilon', 'FontSize', 10);
ylabel('Variance', 'FontSize', 10);
legend('Location', 'best', 'FontSize', 9);
grid on;

% Add reference lines for variance scaling
eps_ref = logspace(log10(min(epsilon_values)), log10(max(epsilon_values)), 100);
var_ref_same = eps_ref.^(-2) * delta_fd_var_same(end) / epsilon_values(end)^(-2);
var_ref_indep = eps_ref.^(-2) * delta_fd_var_indep(end) / epsilon_values(end)^(-2);
loglog(eps_ref, var_ref_same, 'r--', 'LineWidth', 1.5, 'DisplayName', 'O(\epsilon^{-2})');

% Save the figure
print -dpng 'delta_fd.png';

% Plot results for Gamma
figure('Position', [100, 100, 1200, 500]);

subplot(1, 2, 1);
semilogx(epsilon_values, gamma_fd_same, 'ro-', 'LineWidth', 2, 'DisplayName', 'Same Realizations');
hold on;
semilogx(epsilon_values, gamma_fd_indep, 'bs-', 'LineWidth', 2, 'DisplayName', 'Independent Realizations');
semilogx(epsilon_values, gamma_bs*ones(size(epsilon_values)), 'k--', 'LineWidth', 1.5, 'DisplayName', 'Black-Scholes Gamma');
title('Finite Difference Estimator for Gamma', 'FontSize', 12);
xlabel('Epsilon', 'FontSize', 10);
ylabel('Gamma', 'FontSize', 10);
legend('Location', 'best', 'FontSize', 9);
grid on;

% Add error information
[~, best_idx_same] = min(abs(gamma_fd_same - gamma_bs));
[~, best_idx_indep] = min(abs(gamma_fd_indep - gamma_bs));
text(0.05, 0.15, sprintf('Best epsilon (same): %.3f', epsilon_values(best_idx_same)), 'Units', 'normalized', 'FontSize', 9);
text(0.05, 0.10, sprintf('Best epsilon (indep): %.3f', epsilon_values(best_idx_indep)), 'Units', 'normalized', 'FontSize', 9);

subplot(1, 2, 2);
loglog(epsilon_values, gamma_fd_var_same, 'ro-', 'LineWidth', 2, 'DisplayName', 'Same Realizations');
hold on;
loglog(epsilon_values, gamma_fd_var_indep, 'bs-', 'LineWidth', 2, 'DisplayName', 'Independent Realizations');
title('Variance of Gamma Estimator', 'FontSize', 12);
xlabel('Epsilon', 'FontSize', 10);
ylabel('Variance', 'FontSize', 10);
legend('Location', 'best', 'FontSize', 9);
grid on;

% Add reference lines for variance scaling
eps_ref = logspace(log10(min(epsilon_values)), log10(max(epsilon_values)), 100);
var_ref_same = eps_ref.^(-4) * gamma_fd_var_same(end) / epsilon_values(end)^(-4);
var_ref_indep = eps_ref.^(-4) * gamma_fd_var_indep(end) / epsilon_values(end)^(-4);
loglog(eps_ref, var_ref_same, 'r--', 'LineWidth', 1.5, 'DisplayName', 'O(\epsilon^{-4})');

% Save the figure
print -dpng 'gamma_fd.png';

%% 1.4-1.5. Optimal choice of parameters
% For a global error of order eta, we need:
% - For Delta: epsilon ~ sqrt(eta), M ~ 1/eta^2
% - For Gamma: epsilon ~ eta^(1/3), M ~ 1/eta^2

% Compute global error for different combinations of epsilon and M
eta_values = [0.1, 0.05, 0.01, 0.005, 0.001];
global_error_delta = zeros(length(eta_values), 1);
global_error_gamma = zeros(length(eta_values), 1);
optimal_epsilon_delta = zeros(length(eta_values), 1);
optimal_epsilon_gamma = zeros(length(eta_values), 1);
optimal_M_delta = zeros(length(eta_values), 1);
optimal_M_gamma = zeros(length(eta_values), 1);

for i = 1:length(eta_values)
    eta = eta_values(i);
    
    % For Delta
    epsilon_delta = sqrt(eta);
    M_delta = ceil(1/eta^2);
    optimal_epsilon_delta(i) = epsilon_delta;
    optimal_M_delta(i) = M_delta;
    
    % For Gamma
    epsilon_gamma = eta^(1/3);
    M_gamma = ceil(1/eta^2);
    optimal_epsilon_gamma(i) = epsilon_gamma;
    optimal_M_gamma(i) = M_gamma;
    
    % Compute global errors
    bias_delta = epsilon_delta^2;  % O(epsilon^2) for Delta
    variance_delta = 1/sqrt(M_delta);  % O(1/sqrt(M)) for Monte Carlo
    global_error_delta(i) = bias_delta + variance_delta;
    
    bias_gamma = epsilon_gamma;  % O(epsilon) for Gamma
    variance_gamma = 1/sqrt(M_gamma);  % O(1/sqrt(M)) for Monte Carlo
    global_error_gamma(i) = bias_gamma + variance_gamma;
end

% Print results
fprintf('\nGlobal Error Analysis for Finite Difference Estimators:\n');
fprintf('Eta\tDelta Epsilon\tDelta M\tDelta Error\tGamma Epsilon\tGamma M\tGamma Error\n');
for i = 1:length(eta_values)
    fprintf('%.3f\t%.6f\t%d\t%.6f\t%.6f\t%d\t%.6f\n', ...
        eta_values(i), optimal_epsilon_delta(i), optimal_M_delta(i), global_error_delta(i), ...
        optimal_epsilon_gamma(i), optimal_M_gamma(i), global_error_gamma(i));
end

% Plot optimal parameters
figure('Position', [100, 100, 1200, 500]);

subplot(1, 2, 1);
loglog(eta_values, optimal_epsilon_delta, 'ro-', 'LineWidth', 2, 'DisplayName', 'Delta Epsilon');
hold on;
loglog(eta_values, optimal_epsilon_gamma, 'bs-', 'LineWidth', 2, 'DisplayName', 'Gamma Epsilon');
loglog(eta_values, sqrt(eta_values), 'r--', 'LineWidth', 1.5, 'DisplayName', 'O(\eta^{1/2})');
loglog(eta_values, eta_values.^(1/3), 'b--', 'LineWidth', 1.5, 'DisplayName', 'O(\eta^{1/3})');
title('Optimal Epsilon vs. Target Error', 'FontSize', 12);
xlabel('Target Error (\eta)', 'FontSize', 10);
ylabel('Optimal Epsilon', 'FontSize', 10);
legend('Location', 'best', 'FontSize', 9);
grid on;

subplot(1, 2, 2);
loglog(eta_values, optimal_M_delta, 'ro-', 'LineWidth', 2, 'DisplayName', 'Delta M');
hold on;
loglog(eta_values, optimal_M_gamma, 'bs-', 'LineWidth', 2, 'DisplayName', 'Gamma M');
loglog(eta_values, 1./eta_values.^2, 'k--', 'LineWidth', 1.5, 'DisplayName', 'O(\eta^{-2})');
title('Optimal Sample Size vs. Target Error', 'FontSize', 12);
xlabel('Target Error (\eta)', 'FontSize', 10);
ylabel('Optimal Sample Size (M)', 'FontSize', 10);
legend('Location', 'best', 'FontSize', 9);
grid on;

% Save the figure
print -dpng 'optimal_parameters.png';

%% Summary of Results

fprintf('\n=== Summary of Results ===\n\n');

fprintf('Exercise 1: Weak and Strong Error\n');
fprintf('Theoretical value of E[X_1]: %.6f\n', E_X1_theoretical);
fprintf('\nStrong Error Convergence:\n');
fprintf('h\tEuler Error\tMilstein Error\tImprovement\n');
for i = 1:length(h_values)
    fprintf('%.6f\t%.6f\t%.6f\t%.2f%%\n', h_values(i), euler_errors(i), milstein_errors(i), ...
        100*(euler_errors(i)-milstein_errors(i))/euler_errors(i));
end

fprintf('\nExercise 2: Sensitivity of Option Prices\n');
fprintf('Black-Scholes Call Price: %.6f\n', call_price);
fprintf('Black-Scholes Delta: %.6f\n', delta_bs);
fprintf('Black-Scholes Gamma: %.6f\n', gamma_bs);

fprintf('\nFinite Difference Estimator for Delta:\n');
fprintf('Epsilon\tSame Realizations\tIndependent Realizations\n');
for i = 1:length(epsilon_values)
    fprintf('%.3f\t%.6f\t%.6f\n', epsilon_values(i), delta_fd_same(i), delta_fd_indep(i));
end

fprintf('\nFinite Difference Estimator for Gamma:\n');
fprintf('Epsilon\tSame Realizations\tIndependent Realizations\n');
for i = 1:length(epsilon_values)
    fprintf('%.3f\t%.6f\t%.6f\n', epsilon_values(i), gamma_fd_same(i), gamma_fd_indep(i));
end

fprintf('\nAnalysis completed successfully. All figures saved.\n');
