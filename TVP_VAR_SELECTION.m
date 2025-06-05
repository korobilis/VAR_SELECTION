%--------------------------------------------------------------------------
% TIME-VARYING PARAMETER VAR WITH VARIABLE SELECTION - EDUCATIONAL SCRIPT
%--------------------------------------------------------------------------
% PURPOSE:
%   Implements Bayesian variable selection for Time-Varying Parameter (TVP)
%   Vector Autoregressive models using Gibbs sampling. The model is:
%   
%   Y_t = Z_t × GAMMA × B_t + e_t,  e_t ~ N(0, SIGMA)
%   B_t = B_{t-1} + u_t,            u_t ~ N(0, Q)
%
%   where:
%   - B_t are time-varying coefficients following random walks
%   - GAMMA is a diagonal matrix of 0-1 restriction indices  
%   - SIGMA is the error covariance matrix
%   - Q is the state innovation covariance matrix
%
% REFERENCE:
%   Primiceri, G. E. (2005). Time varying structural vector autoregressions 
%   and monetary policy. The Review of Economic Studies, 72(3), 821-852.
%
% AUTHOR: Dimitris Korobilis (original), Improved version 2025
%--------------------------------------------------------------------------

%% HOUSEKEEPING
clear all;
clc;
randn('state', sum(100*clock));
rand('twister', sum(100*clock));

fprintf('==========================================================\n');
fprintf('TIME-VARYING PARAMETER VAR WITH VARIABLE SELECTION\n');
fprintf('==========================================================\n');

%% DATA GENERATION AND PREPARATION
fprintf('\n1. GENERATING ARTIFICIAL TVP-VAR DATA...\n');

% Generate artificial TVP-VAR data
[Y, b_true] = tvpvarsim();

% Check for explosive behavior
if sum(std(Y) > 5) > 0
    error('Explosive model generated. Please rerun the code.');
end

% Basic dimensions
t = size(Y, 1);
M = size(Y, 2);
p = M;              % Dimension of Y
plag = 1;           % Number of lags
numa = p * (p-1) / 2;  % Number of parameters in lower triangular matrix

fprintf('   - Time series length: %d\n', t);
fprintf('   - Number of variables: %d\n', M);
fprintf('   - Number of lags: %d\n', plag);

%% VAR MODEL SETUP  
fprintf('\n2. SETTING UP TVP-VAR MODEL...\n');

% Generate lagged Y matrix
ylag = mlag2(Y, plag);
ylag = ylag(plag+1:t, :);

% Number of states (time-varying parameters)
m = plag * (p^2);

% Calculate effective sample size
T_eff = t - plag;

fprintf('   - Number of time-varying parameters: %d\n', m);
fprintf('   - Effective sample size: %d\n', T_eff);

% Create design matrix Z_t as in Primiceri (2005)
Z = zeros((T_eff)*p, m);
for i = 1:T_eff
    ztemp = [];
    for j = 1:plag        
        xtemp = ylag(i, (j-1)*p+1:j*p);
        xtemp = kron(eye(p), xtemp);
        ztemp = [ztemp xtemp];
    end
    Z((i-1)*p+1:i*p, :) = ztemp;
end

% Redefine variables for TVP-VAR
y = Y(plag+1:t, :)';

%% GIBBS SAMPLER SETTINGS
fprintf('\n3. SETTING UP GIBBS SAMPLER...\n');

nrep = 500;      % Number of replications  
nburn = 500;     % Number of burn-in draws
nthin = 1;       % Thinning parameter
it_print = 100;  % Print progress every it_print iterations

fprintf('   - Total iterations: %d\n', nrep + nburn);
fprintf('   - Burn-in period: %d\n', nburn);
fprintf('   - Saved draws: %d\n', nrep);

%% PRIOR SPECIFICATIONS
fprintf('\n4. SETTING PRIOR DISTRIBUTIONS...\n');

% Initial conditions for state vector (uninformative)
B_OLS = zeros(m, 1);
VB_OLS = eye(m);

% Hyperparameters
k_Q = 0.01;     % Scaling factor for Q prior

% Prior for initial state: B_0 ~ N(B_OLS, 4*VB_OLS)
B_0_prmean = B_OLS;
B_0_prvar = 4 * VB_OLS;

% Prior for state covariance: Q ~ IW(k_Q^2*(1+m)*VB_OLS, 1+m)
Q_prmean = ((k_Q)^2) * (1 + m) * VB_OLS;
Q_prvar = 1 + m;

% Prior for inclusion indicators: gamma_j ~ Bernoulli(p_j)
p_j = 0.5 * ones(m, 1);

% Prior for error covariance: sigma ~ IW(eta, a)
a = 0;          % Degrees of freedom (uninformative)
eta = 1 * eye(p);
eta_inv = inv(eta);

fprintf('   - State innovation variance scaling: %g\n', k_Q);
fprintf('   - Inclusion probability: %g\n', 0.5);

%% INITIALIZATION
fprintf('\n5. INITIALIZING PARAMETERS...\n');

% Initialize covariance matrices
consQ = 0.0001;
consH = 0.0001;
Qdraw = consQ * eye(m);

% Initialize error covariance (using true values for demonstration)
sigma_sd = [1.0000   -0.5000   -0.2500   -0.1250;
           -0.5000    1.2500   -0.3750   -0.1875;
           -0.2500   -0.3750    1.3125   -0.3437;
           -0.1250   -0.1875   -0.3437    1.3281];

% Ensure we use the right dimensions
if size(sigma_sd, 1) ~= p
    sigma_sd = eye(p);  % Fallback to identity
end

Ht = kron(ones(T_eff, 1), sigma_sd);
Htsd = kron(ones(T_eff, 1), chol(sigma_sd));

% Initialize states and inclusion indicators
Btdraw = zeros(m, T_eff);
gamma = ones(m, 1);

% Storage for posterior draws
Bt_postmean = zeros(m, T_eff);
Qmean = zeros(m, m);
gamma_draws = zeros(nrep, m);
sigma_draws = zeros(nrep, p, p);
log_likelihood_draws = zeros(nrep, 1);

fprintf('   - All parameters initialized\n');

%% GIBBS SAMPLER MAIN LOOP
fprintf('\n6. RUNNING GIBBS SAMPLER...\n');
fprintf('Progress:\n');

tic;

for irep = 1:nrep + nburn
    
    % Print progress
    if mod(irep, it_print) == 0
        fprintf('   Iteration %d/%d (%.1f%% complete)\n', ...
                irep, nrep + nburn, 100*irep/(nrep + nburn));
    end
    
    %----------------------------------------------------------------------
    % STEP 1: SAMPLE TIME-VARYING PARAMETERS B_t | GAMMA, SIGMA, Q, Y
    %----------------------------------------------------------------------
    
    % Use Carter-Kohn algorithm for sampling TVP
    [Btdrawc, log_lik] = carter_kohn(y, Z*diag(gamma), Ht, Qdraw, ...
                                     m, p, T_eff, B_0_prmean, B_0_prvar);     
    Btdraw = Btdrawc';
    %----------------------------------------------------------------------
    % STEP 2: SAMPLE STATE COVARIANCE Q | B_t, GAMMA, SIGMA, Y  
    %----------------------------------------------------------------------
    
    % Compute sum of squared state innovations
    sse_2 = diff(Btdrawc)'*diff(Btdrawc);
    
    % Posterior parameters for inverse-Wishart
    Qinv = inv(sse_2 + Q_prmean + 1e-6*eye(m));  % Add regularization
    Qinvdraw = wish(Qinv, T_eff + Q_prvar);
    Qdraw = inv(Qinvdraw);
    
    % Ensure Q is well-conditioned
    [U, S, V] = svd(Qdraw);
    Qdraw = U * max(S, 1e-8*eye(m)) * V';
    
    %----------------------------------------------------------------------
    % STEP 3: SAMPLE INCLUSION INDICATORS GAMMA | B_t, SIGMA, Q, Y
    %----------------------------------------------------------------------
    
    for j = 1:m
        % Current parameter matrix
        theta = diag(gamma)*Btdraw;
        
        % Alternative scenarios
        theta_j_include = theta;
        theta_j_exclude = theta;
        theta_j_include(j, :) = Btdraw(j, :);  % Include variable j
        theta_j_exclude(j, :) = 0;             % Exclude variable j
        
        % Compute log-likelihoods for numerical stability
        log_lik_include = 0;
        log_lik_exclude = 0;
        
        for i = 1:T_eff
            % Residuals for inclusion case
            resid_inc = y(:, i) - Z((i-1)*p+1:i*p, :) * theta_j_include(:, i);
            Ht_i = Htsd((i-1)*p+1:i*p, :);
            
            % Transform residuals
            trans_resid_inc = Ht_i \ resid_inc;
            log_lik_include = log_lik_include - 0.5 * (trans_resid_inc' * trans_resid_inc);
            
            % Residuals for exclusion case  
            resid_exc = y(:, i) - Z((i-1)*p+1:i*p, :) * theta_j_exclude(:, i);
            trans_resid_exc = Ht_i \ resid_exc;
            log_lik_exclude = log_lik_exclude - 0.5 * (trans_resid_exc' * trans_resid_exc);
        end
        
        % Compute inclusion probability using log-space
        log_prior_inc = log(p_j(j));
        log_prior_exc = log(1 - p_j(j));
        
        log_post_inc = log_prior_inc + log_lik_include;
        log_post_exc = log_prior_exc + log_lik_exclude;
        
        % Numerically stable probability calculation
        log_diff = log_post_inc - log_post_exc;
        if log_diff > 500
            p_j_tilde = 1;
        elseif log_diff < -500
            p_j_tilde = 0;
        else
            p_j_tilde = 1 / (1 + exp(-log_diff));
        end
        
        % Sample inclusion indicator
        gamma(j, 1) = bernoullirnd(p_j_tilde);
    end
    
    %----------------------------------------------------------------------
    % STEP 4: SAMPLE ERROR COVARIANCE SIGMA | B_t, GAMMA, Q, Y
    %----------------------------------------------------------------------
    
    sse_2 = zeros(p, p);
    for i = 1:T_eff
        theta = Btdraw(:, i) .* gamma;
        resid = y(:, i) - Z((i-1)*p+1:i*p, :) * theta;
        sse_2 = sse_2 + resid * resid';
    end
    
    % Posterior parameters for inverse-Wishart
    R_1 = inv(eta_inv + sse_2 + 1e-6*eye(p));  % Add regularization
    R_2 = a + T_eff;
    
    % Draw from inverse-Wishart
    rd = wish(R_1, R_2);
    sigma = inv(rd);
    
    % Ensure positive definiteness and update related matrices
    [U, S, V] = svd(sigma);
    sigma = U * max(S, 1e-8*eye(p)) * V';
    
    Ht = kron(ones(T_eff, 1), sigma);
    try
        Htsd = kron(ones(T_eff, 1), chol(sigma));
    catch
        % If Cholesky fails, use SVD-based decomposition
        [U, S, V] = svd(sigma);
        Htsd = kron(ones(T_eff, 1), U * sqrt(max(S, 1e-8*eye(p))));
    end
    
    %----------------------------------------------------------------------
    % SAVE DRAWS (after burn-in)
    %----------------------------------------------------------------------
    if irep > nburn
        Bt_postmean = Bt_postmean + Btdraw;
        gamma_draws(irep - nburn, :) = gamma';
        Qmean = Qmean + Qdraw;
        sigma_draws(irep - nburn, :, :) = sigma;
        log_likelihood_draws(irep - nburn) = log_lik;
    end
    
end

%% POST-PROCESSING
fprintf('\n7. PROCESSING RESULTS...\n');

% Compute posterior means
Bt_postmean = Bt_postmean / nrep;
Qmean = Qmean / nrep;
gamma_mean = mean(gamma_draws)';
SIGMA_mean = squeeze(mean(sigma_draws, 1));

% Compute posterior inclusion probabilities summary
inclusion_summary = [
    min(gamma_mean), ...
    mean(gamma_mean), ...
    max(gamma_mean), ...
    sum(gamma_mean > 0.5)
];

%% DISPLAY RESULTS
clc;
elapsed_time = toc;

fprintf('==========================================================\n');
fprintf('TIME-VARYING PARAMETER VAR SELECTION RESULTS\n');
fprintf('==========================================================\n');
fprintf('\nTotal computation time: %.2f seconds\n', elapsed_time);

fprintf('\nModel Dimensions:\n');
fprintf('  - Variables (M): %d\n', M);
fprintf('  - Time periods (T_eff): %d\n', T_eff);
fprintf('  - Lags (plag): %d\n', plag);
fprintf('  - Time-varying parameters (m): %d\n', m);

fprintf('\nMCMC Settings:\n');
fprintf('  - Total iterations: %d\n', nrep + nburn);
fprintf('  - Burn-in: %d\n', nburn);
fprintf('  - Saved draws: %d\n', nrep);

fprintf('\nVariable Selection Results:\n');
fprintf('  - Min inclusion probability: %.3f\n', inclusion_summary(1));
fprintf('  - Mean inclusion probability: %.3f\n', inclusion_summary(2));
fprintf('  - Max inclusion probability: %.3f\n', inclusion_summary(3));
fprintf('  - Variables with prob > 0.5: %d out of %d\n', ...
        inclusion_summary(4), m);

fprintf('\nPosterior Means (first 10 parameters):\n');
fprintf('%-8s | %-12s | %-15s\n', 'Param', 'Incl.Prob', 'Final B_T');
fprintf('---------|--------------|----------------\n');

for i = 1:min(m, 10)
    fprintf('%-8d | %-12.3f | %-15.6f\n', ...
            i, gamma_mean(i), Bt_postmean(i, end));
end

if m > 10
    fprintf('... (showing first 10 of %d parameters)\n', m);
end

fprintf('\nModel Diagnostics:\n');
fprintf('  - Mean log-likelihood: %.2f\n', mean(log_likelihood_draws));
fprintf('  - Posterior mean cond. number of Q: %.2e\n', cond(Qmean));
fprintf('  - Posterior mean cond. number of SIGMA: %.2e\n', cond(SIGMA_mean));

fprintf('\n==========================================================\n');
fprintf('All variables saved in workspace for further analysis\n');
fprintf('Key variables: Bt_postmean, gamma_draws, sigma_draws, Qmean\n');
fprintf('==========================================================\n');