%--------------------------------------------------------------------------
% VARIABLE SELECTION IN A VAR MODEL - EDUCATIONAL SCRIPT
%--------------------------------------------------------------------------
% PURPOSE:
%   Implements Bayesian variable selection for Vector Autoregressive (VAR) 
%   models using Gibbs sampling. The model can be written as:
%   
%   Y = Z × GAMMA × BETA + e,     e ~ N(0, I ⊗ SIGMA)
%
%   where:
%   - GAMMA is a diagonal matrix of 0-1 restriction indices
%   - BETA are the regression coefficients  
%   - SIGMA is the covariance matrix
%
% REFERENCE:
%   Korobilis, D. (2013). VAR forecasting using Bayesian variable selection.
%   Journal of Applied Econometrics, 28(2), 204-230.
%
% AUTHOR: Dimitris Korobilis (original), Improved version 2025
%--------------------------------------------------------------------------

%% HOUSEKEEPING
clear all;
clc;
tic;

fprintf('==========================================================\n');
fprintf('BAYESIAN VARIABLE SELECTION FOR VAR MODELS\n');
fprintf('==========================================================\n');

%% DATA GENERATION AND PREPARATION
fprintf('\n1. GENERATING ARTIFICIAL DATA...\n');

% Generate artificial data for demonstration
[Y, PHI_true] = simvardgp();

% Demean the data (common practice in VAR modeling)
Y = Y - repmat(mean(Y), size(Y,1), 1);

% Basic dimensions
[Traw, m] = size(Y);
fprintf('   - Time series length: %d\n', Traw);
fprintf('   - Number of variables: %d\n', m);

%% VAR MODEL SETUP
fprintf('\n2. SETTING UP VAR MODEL...\n');

plag = 1; % Number of lags
fprintf('   - Number of lags: %d\n', plag);

% Generate lagged Y matrix using mlag2 function
ylag = mlag2(Y, plag);

% Calculate dimensions
n = (plag * m) * m;  % Total number of coefficients
p = plag * m;        % Number of regressors per equation
T = Traw - plag;     % Effective sample size

fprintf('   - Effective sample size: %d\n', T);
fprintf('   - Total number of coefficients: %d\n', n);
fprintf('   - Regressors per equation: %d\n', p);

% Construct design matrix X using Kronecker product structure
x2 = ylag(plag+1:Traw, :);  % Lagged variables
x = kron(eye(m), x2);       % Full design matrix

% Construct vectorized dependent variable
y2 = Y(plag+1:Traw, :);     % Adjusted Y matrix
y = y2(:);                  % Vectorized form

%% PRIOR SPECIFICATIONS
fprintf('\n3. SETTING PRIOR DISTRIBUTIONS...\n');

% Gibbs sampler settings
nsave = 5000;   % Number of draws to save
nburn = 5000;   % Number of burn-in draws
ntot = nsave + nburn;

fprintf('   - Burn-in draws: %d\n', nburn);
fprintf('   - Saved draws: %d\n', nsave);

% Storage for posterior draws
beta_draws = zeros(nsave, n);
gamma_draws = zeros(nsave, n);
sigma_draws = zeros(nsave, m, m);
log_marglik_draws = zeros(nsave, 1);

% Prior for regression coefficients: beta ~ N(b0, D0)
b0 = zeros(n, 1);           % Prior mean (uninformative)
D0 = 9 * eye(n);            % Prior covariance (relatively diffuse)
D0_inv = inv(D0);

fprintf('   - Beta prior: N(0, %g*I)\n', 9);

% Prior for inclusion indicators: gamma_j ~ Bernoulli(p_j)
p_j = 0.5 * ones(n, 1);     % Equal prior inclusion probability
fprintf('   - Inclusion probability: %g\n', 0.5);

% Prior for error covariance: Sigma ~ IW(eta, a)
a = m;                      % Degrees of freedom
eta = eye(m);               % Scale matrix
eta_inv = inv(eta);

fprintf('   - Sigma prior: IW(I, %d)\n', a);

%% INITIALIZATION
fprintf('\n4. INITIALIZING PARAMETERS...\n');

% Initialize inclusion indicators (start with all variables included)
gamma = ones(n, 1);

% Get OLS estimates for initialization and comparison
beta_OLS = (x' * x) \ (x' * y);           % Vectorized OLS
beta_OLS2 = (x2' * x2) \ (x2' * y2);      % Matrix form OLS

% Initialize error covariance matrix
sse = (y2 - x2 * beta_OLS2)' * (y2 - x2 * beta_OLS2);
sigma = sse / (T - p + 1);

fprintf('   - Starting with all variables included\n');
fprintf('   - Initial sigma condition number: %.2e\n', cond(sigma));

%% GIBBS SAMPLER
fprintf('\n5. RUNNING GIBBS SAMPLER...\n');
fprintf('Progress:\n');

for irep = 1:ntot
    
    % Display progress
    if mod(irep, 500) == 0
        fprintf('   Iteration %d/%d (%.1f%% complete)\n', ...
                irep, ntot, 100*irep/ntot);
    end
    
    %----------------------------------------------------------------------
    % STEP 1: SAMPLE BETA | GAMMA, SIGMA, Y
    %----------------------------------------------------------------------
    
    % Precision matrix for likelihood
    V = kron(inv(sigma), eye(T));
    
    % Apply variable selection (element-wise multiplication)
    x_star = x .* repmat(gamma', T*m, 1);
    
    % Posterior distribution of beta is Normal
    % Precision matrix
    beta_prec = D0_inv + x_star' * V * x_star;
    
    % Add small diagonal term for numerical stability
    beta_prec = beta_prec + 1e-12 * eye(n);
    
    % Posterior covariance (using Cholesky for stability)
    try
        L_beta = chol(beta_prec, 'lower');
        beta_var = inv(beta_prec);
    catch
        % If Cholesky fails, add more regularization
        beta_prec = beta_prec + 1e-6 * eye(n);
        L_beta = chol(beta_prec, 'lower');
        beta_var = inv(beta_prec);
    end
    
    % Posterior mean
    beta_mean = beta_var * (D0_inv * b0 + x_star' * V * y);
    
    % Draw from posterior using Cholesky decomposition
    beta = beta_mean + beta_var^0.5 * randn(n, 1);
    
    % Reshape for convenience
    beta_mat = reshape(beta, p, m);
    
    %----------------------------------------------------------------------
    % STEP 2: SAMPLE GAMMA | BETA, SIGMA, Y (USING LOG-SPACE FOR STABILITY)
    %----------------------------------------------------------------------
    
    % Randomize order for better mixing
    ind_perm = randperm(n)';
    
    for kk = 1:n
        j = ind_perm(kk);
        
        % Current parameter vector
        theta = beta .* gamma;
        
        % Candidate vectors (include vs exclude variable j)
        theta_include = theta;
        theta_exclude = theta;
        theta_include(j) = beta(j);  % Include variable j
        theta_exclude(j) = 0;        % Exclude variable j
        
        % Calculate log-probabilities to avoid numerical overflow
        % Log-likelihood for inclusion
        resid_include = y - x * theta_include;
        log_lik_include = -0.5 * resid_include' * V * resid_include;
        
        % Log-likelihood for exclusion  
        resid_exclude = y - x * theta_exclude;
        log_lik_exclude = -0.5 * resid_exclude' * V * resid_exclude;
        
        % Log-posterior probabilities (up to constant)
        log_post_include = log(p_j(j)) + log_lik_include;
        log_post_exclude = log(1 - p_j(j)) + log_lik_exclude;
        
        % Numerically stable computation of inclusion probability
        log_diff = log_post_include - log_post_exclude;
        if log_diff > 500  % Avoid overflow
            p_include = 1;
        elseif log_diff < -500  % Avoid underflow
            p_include = 0;
        else
            p_include = 1 / (1 + exp(-log_diff));
        end
        
        % Draw indicator
        gamma(j) = bernoullirnd(p_include);
    end
    
    % Reshape gamma for convenience
    gamma_mat = reshape(gamma, p, m);
    
    %----------------------------------------------------------------------
    % STEP 3: SAMPLE SIGMA | BETA, GAMMA, Y
    %----------------------------------------------------------------------
    
    % Apply restrictions
    theta_mat = beta_mat .* gamma_mat;
    
    % Residuals
    residuals = y2 - x2 * theta_mat;
    
    % Posterior parameters for inverse-Wishart
    scale_post = eta_inv + residuals' * residuals;
    df_post = a + T;
    
    % Draw from inverse-Wishart using Wishart
    sigma_inv_draw = wish(inv(scale_post), df_post);
    sigma = inv(sigma_inv_draw);
    
    % Ensure positive definiteness
    [U, S, V] = svd(sigma);
    sigma = U * max(S, 1e-8 * eye(m)) * V';
    
    %----------------------------------------------------------------------
    % COMPUTE LOG MARGINAL LIKELIHOOD (for model comparison)
    %----------------------------------------------------------------------
    if irep > nburn
        % Simple approximation using current draws
        log_marglik_draws(irep - nburn) = log_lik_include;
    end
    
    %----------------------------------------------------------------------
    % SAVE DRAWS (after burn-in)
    %----------------------------------------------------------------------
    if irep > nburn
        beta_draws(irep - nburn, :) = beta;
        gamma_draws(irep - nburn, :) = gamma;
        sigma_draws(irep - nburn, :, :) = sigma;
    end
    
end

%% POST-PROCESSING AND RESULTS
fprintf('\n6. PROCESSING RESULTS...\n');

% Compute posterior means
beta_mean_post = mean(beta_draws)';
gamma_mean_post = mean(gamma_draws)';
sigma_mean_post = squeeze(mean(sigma_draws, 1));

% Compute posterior standard deviations
beta_std_post = std(beta_draws)';
gamma_std_post = std(gamma_draws)';

% Reshape results for easier interpretation
beta_matrix_mean = reshape(beta_mean_post, p, m);
gamma_matrix_mean = reshape(gamma_mean_post, p, m);

%% DISPLAY RESULTS
clc;
fprintf('==========================================================\n');
fprintf('BAYESIAN VARIABLE SELECTION RESULTS\n');
fprintf('==========================================================\n');
fprintf('\nTotal computation time: %.2f seconds\n', toc);
fprintf('\nModel Dimensions:\n');
fprintf('  - Variables (m): %d\n', m);
fprintf('  - Lags (p): %d\n', plag);
fprintf('  - Sample size (T): %d\n', T);
fprintf('  - Total coefficients: %d\n', n);

fprintf('\nMCMC Diagnostics:\n');
fprintf('  - Burn-in: %d\n', nburn);
fprintf('  - Saved draws: %d\n', nsave);

fprintf('\nPosterior Inclusion Probabilities and Coefficient Estimates:\n');
fprintf('%-8s | %-12s | %-12s | %-12s | %-12s\n', ...
        'Coeff', 'Incl.Prob', 'Post.Mean', 'Post.Std', 'OLS');
fprintf('---------|--------------|--------------|--------------|-------------\n');

bb = PHI_true(:);
for i = 1:min(n, 30)  % Show first 20 coefficients
    fprintf('%-8d | %-12.3f | %-12.6f | %-12.6f | %-12.6f\n', ...
            bb(i), gamma_mean_post(i), beta_mean_post(i), ...
            beta_std_post(i), beta_OLS(i));
end

if n > 30
    fprintf('... (showing first 30 of %d coefficients)\n', n);
end

% Summary statistics
fprintf('\nSummary:\n');
fprintf('  - Average inclusion probability: %.3f\n', mean(gamma_mean_post));
fprintf('  - Number of "significant" variables (prob > 0.5): %d\n', ...
        sum(gamma_mean_post > 0.5));
fprintf('  - Posterior mean of log marginal likelihood: %.2f\n', ...
        mean(log_marglik_draws));

fprintf('\n==========================================================\n');
fprintf('All variables saved in workspace for further analysis\n');
fprintf('Key variables: beta_draws, gamma_draws, sigma_draws\n');
fprintf('==========================================================\n');