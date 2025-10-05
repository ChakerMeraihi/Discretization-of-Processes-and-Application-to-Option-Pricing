%% Financial Mathematics
% Black-Scholes Model and Option Replication: MATLAB Implementation (Enhanced Version)
% 
% This script implements the numerical methods described in the theoretical
% solutions for the TP_1_FINUM_2024.pdf assignment.

clear all;
close all;
clc;


%% Parameters
% Define common parameters used throughout the script
S0 = 100;         % Initial stock price
K = 97;           % Strike price
sigma = 0.2;      % Volatility
mu = 0.03;        % Drift under physical measure P
r = 0.015;        % Risk-free rate
T = 1;            % Time to maturity
seed = 42;        % Random seed for reproducibility
% Use Octave-compatible random seed setting
rand("state", seed);
randn("state", seed);

%% Black-Scholes Model Functions

% Function to simulate stock price paths under the physical measure P
simulateStockP = @(S0, mu, sigma, T, dt, N_paths) S0 * cumprod(exp((mu - sigma^2/2)*dt + sigma*sqrt(dt)*randn(N_paths, T/dt)), 2);

% Function to simulate stock price paths under the risk-neutral measure Q
simulateStockQ = @(S0, r, sigma, T, dt, N_paths) S0 * cumprod(exp((r - sigma^2/2)*dt + sigma*sqrt(dt)*randn(N_paths, T/dt)), 2);

% Function to compute Black-Scholes call option price
bsCallPrice = @(S, K, r, sigma, tau) S * normcdf((log(S/K) + (r + sigma^2/2)*tau)/(sigma*sqrt(tau))) - K * exp(-r*tau) * normcdf((log(S/K) + (r - sigma^2/2)*tau)/(sigma*sqrt(tau)));

% Function to compute Black-Scholes call option delta
bsCallDelta = @(S, K, r, sigma, tau) normcdf((log(S/K) + (r + sigma^2/2)*tau)/(sigma*sqrt(tau)));

% Function to compute Black-Scholes call option gamma
bsCallGamma = @(S, K, r, sigma, tau) normpdf((log(S/K) + (r + sigma^2/2)*tau)/(sigma*sqrt(tau))) / (S * sigma * sqrt(tau));

% Function to compute Black-Scholes call option theta
bsCallTheta = @(S, K, r, sigma, tau) -S * normpdf((log(S/K) + (r + sigma^2/2)*tau)/(sigma*sqrt(tau))) * sigma / (2 * sqrt(tau)) - r * K * exp(-r*tau) * normcdf((log(S/K) + (r - sigma^2/2)*tau)/(sigma*sqrt(tau)));

%% Option Replication Strategy

% Discretization parameters
N_values = [10, 100, 1000, 5000];  % Number of time steps
M = 500;  % Number of Monte Carlo simulations

% Initialize arrays to store results
error_means = zeros(length(N_values), 1);
error_stds = zeros(length(N_values), 1);
error_abs_means = zeros(length(N_values), 1);
error_rmse = zeros(length(N_values), 1);

% Plot setup for replication error distributions
figure('Position', [100, 100, 1200, 800]);

% Define indices for subplot positions
subplot_indices = [1, 2, 3, 4];

for i_N = 1:length(N_values)
    N = N_values(i_N);
    dt = T/N;
    
    % Initialize array to store replication errors
    replication_errors = zeros(M, 1);
    
    % Run M independent simulations
    for i_sim = 1:M
        % Simulate stock price path under P
        Z = randn(1, N);
        S = zeros(1, N+1);
        S(1) = S0;
        
        for j = 1:N
            S(j+1) = S(j) * exp(sigma * sqrt(dt) * Z(j) + (mu - sigma^2/2) * dt);
        end
        
        % Initialize portfolio value and holdings
        V = zeros(1, N+1);
        delta = zeros(1, N+1);
        bond = zeros(1, N+1);
        
        % Initial portfolio value (option price at t=0)
        V(1) = bsCallPrice(S0, K, r, sigma, T);
        
        % Implement replication strategy
        for j = 1:N
            % Current time
            t = (j-1) * dt;
            
            % Compute delta at current time
            delta(j) = bsCallDelta(S(j), K, r, sigma, T-t);
            
            % Compute bond holdings
            bond(j) = (V(j) - delta(j) * S(j)) * exp(-r * t);
            
            % Update portfolio value using self-financing condition
            V(j+1) = delta(j) * S(j+1) + bond(j) * exp(r * (j*dt));
        end
        
        % Compute final payoff
        payoff = max(S(end) - K, 0);
        
        % Store replication error
        replication_errors(i_sim) = V(end) - payoff;
    end
    
    % Compute error statistics
    error_means(i_N) = mean(replication_errors);
    error_stds(i_N) = std(replication_errors);
    error_abs_means(i_N) = mean(abs(replication_errors));
    error_rmse(i_N) = sqrt(mean(replication_errors.^2));
    
    % Plot histogram of replication errors for all N values
    subplot(2, 2, subplot_indices(i_N));
    % Use hist instead of histogram for Octave compatibility
    [counts, centers] = hist(replication_errors, 30);
    bar(centers, counts/sum(counts), 'FaceColor', [0.3, 0.6, 0.9], 'EdgeColor', 'none');
    hold on;
    
    % Add normal distribution fit
    x = linspace(min(replication_errors), max(replication_errors), 100);
    y = normpdf(x, error_means(i_N), error_stds(i_N));
    plot(x, y, 'r-', 'LineWidth', 2);
    
    title(sprintf('Replication Error Distribution (N = %d)', N), 'FontSize', 12);
    xlabel('Replication Error', 'FontSize', 10);
    ylabel('Probability', 'FontSize', 10);
    grid on;
    
    % Add statistics to plot
    text(0.05, 0.9, sprintf('Mean: %.4f', error_means(i_N)), 'Units', 'normalized', 'FontSize', 9);
    text(0.05, 0.8, sprintf('Std Dev: %.4f', error_stds(i_N)), 'Units', 'normalized', 'FontSize', 9);
    text(0.05, 0.7, sprintf('RMSE: %.4f', error_rmse(i_N)), 'Units', 'normalized', 'FontSize', 9);
end

% Add overall title
title('Distribution of Option Replication Errors', 'FontSize', 14);

% Save the figure
print -dpng 'replication_error_distribution.png';

% Plot convergence of replication error
figure('Position', [100, 100, 800, 600]);

% Plot in log-log scale
loglog(N_values, abs(error_means), 'bo-', 'LineWidth', 2, 'DisplayName', 'Mean Absolute Error');
hold on;
loglog(N_values, error_stds, 'ro-', 'LineWidth', 2, 'DisplayName', 'Standard Deviation');
loglog(N_values, error_rmse, 'go-', 'LineWidth', 2, 'DisplayName', 'RMSE');

% Add reference lines for convergence rates
ref_line_05 = error_stds(1) * (N_values / N_values(1)).^(-0.5);
ref_line_1 = error_stds(1) * (N_values / N_values(1)).^(-1);
loglog(N_values, ref_line_05, 'k--', 'LineWidth', 1.5, 'DisplayName', 'Order 0.5');
loglog(N_values, ref_line_1, 'k-.', 'LineWidth', 1.5, 'DisplayName', 'Order 1.0');

title('Convergence of Replication Error', 'FontSize', 14);
xlabel('Number of Time Steps (N)', 'FontSize', 12);
ylabel('Error', 'FontSize', 12);
legend('Location', 'best', 'FontSize', 10);
grid on;

% Add text explaining convergence rates
text(0.05, 0.15, 'Expected convergence rate: O(N^{-0.5})', 'Units', 'normalized', 'FontSize', 10);
text(0.05, 0.10, 'Theoretical limit due to non-smoothness at K', 'Units', 'normalized', 'FontSize', 10);

% Save the figure
print -dpng 'replication_error_convergence.png';

%% Detailed Analysis of a Single Replication Path

% Choose a moderate number of time steps for visualization
N = 100;
dt = T/N;

% Time grid
t = 0:dt:T;

% Simulate stock price path under P
Z = randn(1, N);
S = zeros(1, N+1);
S(1) = S0;

for j = 1:N
    S(j+1) = S(j) * exp(sigma * sqrt(dt) * Z(j) + (mu - sigma^2/2) * dt);
end

% Initialize portfolio value and holdings
V = zeros(1, N+1);
delta = zeros(1, N+1);
bond = zeros(1, N+1);

% Initial portfolio value (option price at t=0)
V(1) = bsCallPrice(S0, K, r, sigma, T);

% Implement replication strategy
for j = 1:N
    % Current time
    t_j = (j-1) * dt;
    
    % Compute delta at current time
    delta(j) = bsCallDelta(S(j), K, r, sigma, T-t_j);
    
    % Compute bond holdings
    bond(j) = (V(j) - delta(j) * S(j)) * exp(-r * t_j);
    
    % Update portfolio value
    V(j+1) = delta(j) * S(j+1) + bond(j) * exp(r * (j*dt));
end

% Compute final delta for plotting
delta(N+1) = bsCallDelta(S(N+1), K, r, sigma, 0);

% Compute final payoff
payoff = max(S(end) - K, 0);

% Compute option prices along the path
option_prices = zeros(1, N+1);
for j = 1:N+1
    t_j = (j-1) * dt;
    option_prices(j) = bsCallPrice(S(j), K, r, sigma, T-t_j);
end

% Plot stock price path and analysis
figure('Position', [100, 100, 1200, 800]);

subplot(2, 2, 1);
plot(t, S, 'b-', 'LineWidth', 2);
hold on;
plot([0, T], [K, K], 'r--', 'LineWidth', 1.5);
title('Stock Price Path', 'FontSize', 12);
xlabel('Time', 'FontSize', 10);
ylabel('Stock Price', 'FontSize', 10);
grid on;
% Add annotation for strike price
text(T/2, K*1.02, 'Strike Price (K)', 'Color', 'r', 'FontSize', 9);

% Plot option price and portfolio value
subplot(2, 2, 2);
plot(t, option_prices, 'b-', 'LineWidth', 2, 'DisplayName', 'Option Price');
hold on;
plot(t, V, 'r--', 'LineWidth', 2, 'DisplayName', 'Portfolio Value');
title('Option Price vs. Portfolio Value', 'FontSize', 12);
xlabel('Time', 'FontSize', 10);
ylabel('Value', 'FontSize', 10);
legend('Location', 'best', 'FontSize', 9);
grid on;

% Plot delta hedging strategy
subplot(2, 2, 3);
plot(t, delta, 'b-', 'LineWidth', 2);
title('Delta Hedging Strategy', 'FontSize', 12);
xlabel('Time', 'FontSize', 10);
ylabel('Delta', 'FontSize', 10);
grid on;
% Add annotation for delta interpretation
text(0.05, 0.9, 'Delta = \partial V / \partial S', 'Units', 'normalized', 'FontSize', 9);
text(0.05, 0.8, 'Number of shares to hold', 'Units', 'normalized', 'FontSize', 9);

% Plot replication error
subplot(2, 2, 4);
plot(t, V - option_prices, 'b-', 'LineWidth', 2);
title('Replication Error', 'FontSize', 12);
xlabel('Time', 'FontSize', 10);
ylabel('Error', 'FontSize', 10);
grid on;
% Add annotation for error interpretation
text(0.05, 0.9, 'Error = Portfolio Value - Option Price', 'Units', 'normalized', 'FontSize', 9);
text(0.05, 0.8, sprintf('Final Error: %.4f', V(end) - payoff), 'Units', 'normalized', 'FontSize', 9);

% Add overall title
sgtitle('Analysis of Option Replication Strategy', 'FontSize', 14);

% Save the figure
print -dpng 'replication_strategy_analysis.png';

%% Bonus: Convergence Analysis

% Function to compute the probability that S_T is in a neighborhood of K
probNearK = @(S0, K, mu, sigma, T, epsilon) normcdf((log((K+epsilon)/S0) + (mu - sigma^2/2)*T)/(sigma*sqrt(T))) - normcdf((log((K-epsilon)/S0) + (mu - sigma^2/2)*T)/(sigma*sqrt(T)));

% Compute probability for different epsilon values
epsilon_values = logspace(-3, 0, 20);  % More points for smoother curve
prob_near_K = zeros(size(epsilon_values));

for i = 1:length(epsilon_values)
    prob_near_K(i) = probNearK(S0, K, mu, sigma, T, epsilon_values(i));
end

% Plot probability
figure('Position', [100, 100, 800, 600]);
loglog(epsilon_values, prob_near_K, 'bo-', 'LineWidth', 2);
hold on;

% Add reference line for order 1 convergence
ref_line = epsilon_values;
ref_line = ref_line * prob_near_K(10) / epsilon_values(10);
loglog(epsilon_values, ref_line, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Order 1');

title('Probability of S_T in Neighborhood of K', 'FontSize', 14);
xlabel('Epsilon (Distance from K)', 'FontSize', 12);
ylabel('Probability', 'FontSize', 12);
legend('Probability', 'Linear Reference', 'Location', 'best', 'FontSize', 10);
grid on;

% Add explanation text
text(0.05, 0.15, 'Probability scales linearly with \epsilon', 'Units', 'normalized', 'FontSize', 10);
text(0.05, 0.10, 'This affects convergence rate of replication error', 'Units', 'normalized', 'FontSize', 10);

% Save the figure
print -dpng 'probability_near_K.png';

% Theoretical convergence rate analysis
% For a fixed probability p that S_T is in a neighborhood of K where the payoff is not smooth,
% the expected replication error is of order O(sqrt(dt)) due to the discretization error.

% Plot theoretical convergence rate
dt_values = 1 ./ N_values;
theoretical_error = sqrt(dt_values);

figure('Position', [100, 100, 800, 600]);
loglog(N_values, theoretical_error, 'k-', 'LineWidth', 2, 'DisplayName', 'Theoretical O(1/sqrt(N))');
hold on;
loglog(N_values, error_stds, 'ro-', 'LineWidth', 2, 'DisplayName', 'Empirical Std Dev');
loglog(N_values, error_rmse, 'go-', 'LineWidth', 2, 'DisplayName', 'Empirical RMSE');

title('Theoretical vs. Empirical Convergence Rate', 'FontSize', 14);
xlabel('Number of Time Steps (N)', 'FontSize', 12);
ylabel('Error', 'FontSize', 12);
legend('Location', 'best', 'FontSize', 10);
grid on;

% Add explanation text
text(0.05, 0.15, 'Theoretical convergence rate: O(N^{-0.5})', 'Units', 'normalized', 'FontSize', 10);
text(0.05, 0.10, 'Limited by non-smoothness of payoff at K', 'Units', 'normalized', 'FontSize', 10);

% Save the figure
print -dpng 'theoretical_convergence.png';

%% Additional Analysis: Impact of Volatility

% Range of volatility values
sigma_values = [0.1, 0.2, 0.3, 0.4, 0.5];
N = 1000;  % Fixed number of time steps
dt = T/N;
M = 200;   % Number of Monte Carlo simulations

% Initialize arrays to store results
error_means_sigma = zeros(length(sigma_values), 1);
error_stds_sigma = zeros(length(sigma_values), 1);
error_rmse_sigma = zeros(length(sigma_values), 1);

% Initialize progress tracking
fprintf('Analyzing impact of volatility:\n');

for i_sigma = 1:length(sigma_values)
    sigma_i = sigma_values(i_sigma);
    fprintf('  Processing sigma = %.1f (%d/%d)...\n', sigma_i, i_sigma, length(sigma_values));
    
    % Initialize array to store replication errors
    replication_errors = zeros(M, 1);
    
    % Run M independent simulations
    for i_sim = 1:M
        % Simulate stock price path under P
        Z = randn(1, N);
        S = zeros(1, N+1);
        S(1) = S0;
        
        for j = 1:N
            S(j+1) = S(j) * exp(sigma_i * sqrt(dt) * Z(j) + (mu - sigma_i^2/2) * dt);
        end
        
        % Initialize portfolio value and holdings
        V = zeros(1, N+1);
        delta = zeros(1, N+1);
        
        % Initial portfolio value (option price at t=0)
        V(1) = bsCallPrice(S0, K, r, sigma_i, T);
        
        % Implement replication strategy
        for j = 1:N
            % Current time
            t = (j-1) * dt;
            
            % Compute delta at current time
            delta(j) = bsCallDelta(S(j), K, r, sigma_i, T-t);
            
            % Update portfolio value
            V(j+1) = delta(j) * (S(j+1) - S(j)) + (V(j) - delta(j) * S(j)) * (exp(r*dt) - 1);
        end
        
        % Compute final payoff
        payoff = max(S(end) - K, 0);
        
        % Store replication error
        replication_errors(i_sim) = V(end) - payoff;
    end
    
    % Compute error statistics
    error_means_sigma(i_sigma) = mean(replication_errors);
    error_stds_sigma(i_sigma) = std(replication_errors);
    error_rmse_sigma(i_sigma) = sqrt(mean(replication_errors.^2));
end

% Plot impact of volatility on replication error
figure('Position', [100, 100, 800, 600]);
plot(sigma_values, abs(error_means_sigma), 'bo-', 'LineWidth', 2, 'DisplayName', 'Mean Absolute Error');
hold on;
plot(sigma_values, error_stds_sigma, 'ro-', 'LineWidth', 2, 'DisplayName', 'Standard Deviation');
plot(sigma_values, error_rmse_sigma, 'go-', 'LineWidth', 2, 'DisplayName', 'RMSE');

title('Impact of Volatility on Replication Error', 'FontSize', 14);
xlabel('Volatility (\sigma)', 'FontSize', 12);
ylabel('Error', 'FontSize', 12);
legend('Location', 'best', 'FontSize', 10);
grid on;

% Add explanation text
text(0.05, 0.15, 'Higher volatility increases replication error', 'Units', 'normalized', 'FontSize', 10);
text(0.05, 0.10, 'Due to larger price jumps between rebalancing', 'Units', 'normalized', 'FontSize', 10);

% Save the figure
print -dpng 'volatility_impact.png';

%% Additional Analysis: Impact of Moneyness

% Range of moneyness values (K/S0)
moneyness_values = [0.8, 0.9, 1.0, 1.1, 1.2];
K_values = moneyness_values * S0;
N = 1000;  % Fixed number of time steps
dt = T/N;
M = 200;   % Number of Monte Carlo simulations

% Initialize arrays to store results
error_means_moneyness = zeros(length(moneyness_values), 1);
error_stds_moneyness = zeros(length(moneyness_values), 1);
error_rmse_moneyness = zeros(length(moneyness_values), 1);

% Initialize progress tracking
fprintf('Analyzing impact of moneyness:\n');

for i_K = 1:length(K_values)
    K_i = K_values(i_K);
    fprintf('  Processing K/S0 = %.1f (%d/%d)...\n', moneyness_values(i_K), i_K, length(K_values));
    
    % Initialize array to store replication errors
    replication_errors = zeros(M, 1);
    
    % Run M independent simulations
    for i_sim = 1:M
        % Simulate stock price path under P
        Z = randn(1, N);
        S = zeros(1, N+1);
        S(1) = S0;
        
        for j = 1:N
            S(j+1) = S(j) * exp(sigma * sqrt(dt) * Z(j) + (mu - sigma^2/2) * dt);
        end
        
        % Initialize portfolio value and holdings
        V = zeros(1, N+1);
        delta = zeros(1, N+1);
        
        % Initial portfolio value (option price at t=0)
        V(1) = bsCallPrice(S0, K_i, r, sigma, T);
        
        % Implement replication strategy
        for j = 1:N
            % Current time
            t = (j-1) * dt;
            
            % Compute delta at current time
            delta(j) = bsCallDelta(S(j), K_i, r, sigma, T-t);
            
            % Update portfolio value
            V(j+1) = delta(j) * (S(j+1) - S(j)) + (V(j) - delta(j) * S(j)) * (exp(r*dt) - 1);
        end
        
        % Compute final payoff
        payoff = max(S(end) - K_i, 0);
        
        % Store replication error
        replication_errors(i_sim) = V(end) - payoff;
    end
    
    % Compute error statistics
    error_means_moneyness(i_K) = mean(replication_errors);
    error_stds_moneyness(i_K) = std(replication_errors);
    error_rmse_moneyness(i_K) = sqrt(mean(replication_errors.^2));
end

% Plot impact of moneyness on replication error
figure('Position', [100, 100, 800, 600]);
plot(moneyness_values, abs(error_means_moneyness), 'bo-', 'LineWidth', 2, 'DisplayName', 'Mean Absolute Error');
hold on;
plot(moneyness_values, error_stds_moneyness, 'ro-', 'LineWidth', 2, 'DisplayName', 'Standard Deviation');
plot(moneyness_values, error_rmse_moneyness, 'go-', 'LineWidth', 2, 'DisplayName', 'RMSE');

title('Impact of Moneyness (K/S_0) on Replication Error', 'FontSize', 14);
xlabel('Moneyness (K/S_0)', 'FontSize', 12);
ylabel('Error', 'FontSize', 12);
legend('Location', 'best', 'FontSize', 10);
grid on;

% Add explanation text
text(0.05, 0.15, 'At-the-money options (K/S_0 = 1) have highest error', 'Units', 'normalized', 'FontSize', 10);
text(0.05, 0.10, 'Due to higher probability of ending near the strike price', 'Units', 'normalized', 'FontSize', 10);

% Save the figure
print -dpng 'moneyness_impact.png';

%% Summary of Results

fprintf('\n=== Summary of Option Replication Results ===\n\n');

fprintf('Parameters:\n');
fprintf('Initial Stock Price (S0): %.2f\n', S0);
fprintf('Strike Price (K): %.2f\n', K);
fprintf('Risk-Free Rate (r): %.4f\n', r);
fprintf('Drift under P (mu): %.4f\n', mu);
fprintf('Volatility (sigma): %.4f\n', sigma);
fprintf('Time to Maturity (T): %.2f\n\n', T);

fprintf('Convergence Analysis:\n');
fprintf('N\tMean Error\tStd Dev\tRMSE\n');
for i = 1:length(N_values)
    fprintf('%d\t%.6f\t%.6f\t%.6f\n', N_values(i), error_means(i), error_stds(i), error_rmse(i));
end
fprintf('\n');

fprintf('Volatility Impact:\n');
fprintf('Sigma\tMean Error\tStd Dev\tRMSE\n');
for i = 1:length(sigma_values)
    fprintf('%.2f\t%.6f\t%.6f\t%.6f\n', sigma_values(i), error_means_sigma(i), error_stds_sigma(i), error_rmse_sigma(i));
end
fprintf('\n');

fprintf('Moneyness Impact:\n');
fprintf('K/S0\tMean Error\tStd Dev\tRMSE\n');
for i = 1:length(moneyness_values)
    fprintf('%.2f\t%.6f\t%.6f\t%.6f\n', moneyness_values(i), error_means_moneyness(i), error_stds_moneyness(i), error_rmse_moneyness(i));
end

fprintf('\nAnalysis completed successfully. All figures saved.\n');
