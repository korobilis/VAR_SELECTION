function [bdraw, log_lik] = carter_kohn(y, Z, Ht, Qt, m, p, t, B0, V0)
%--------------------------------------------------------------------------
% CARTER-KOHN ALGORITHM FOR SAMPLING TIME-VARYING PARAMETERS
%--------------------------------------------------------------------------
% PURPOSE:
%   Implements the Carter and Kohn (1994) algorithm for sampling time-varying
%   parameters in state space models using forward filtering and backward 
%   sampling (FFBS).
%
% MODEL:
%   Observation equation: Y_t = Z_t * B_t + e_t,  e_t ~ N(0, H_t)
%   State equation:       B_t = B_{t-1} + u_t,   u_t ~ N(0, Q_t)
%
% INPUTS:
%   y       - [p x t] matrix of observations
%   Z       - [t*p x m] design matrix (stacked across time)
%   Ht      - [t*p x p] error covariance (stacked across time)
%   Qt      - [m x m] state innovation covariance  
%   m       - Number of state variables
%   p       - Number of observation variables
%   t       - Number of time periods
%   B0      - [m x 1] initial state mean
%   V0      - [m x m] initial state covariance
%
% OUTPUTS:
%   bdraw   - [m x t] matrix of sampled state vectors
%   log_lik - Scalar log-likelihood value
%
% REFERENCE:
%   Carter, C. K., & Kohn, R. (1994). On Gibbs sampling for state space 
%   models. Biometrika, 81(3), 541-553.
%
% AUTHOR: Dimitris Korobilis (original), Improved version 2025
%--------------------------------------------------------------------------

%% INPUT VALIDATION
if nargin < 9
    error('carter_kohn requires 9 input arguments');
end

% Check dimensions
if size(y, 2) ~= t
    error('y must be p x t matrix');
end
if size(Z, 1) ~= t*p || size(Z, 2) ~= m
    error('Z must be (t*p) x m matrix');
end
if length(B0) ~= m
    error('B0 must be m x 1 vector');
end
if any(size(V0) ~= [m, m])
    error('V0 must be m x m matrix');
end

%% INITIALIZATION
% Storage for filtered means and covariances
bt = zeros(t, m);           % Filtered state means
Vt = zeros(m^2, t);         % Filtered state covariances (vectorized)

% Initialize
bp = B0;                    % Prior mean
Vp = V0;                    % Prior covariance
log_lik = 0;                % Log-likelihood accumulator

% Regularization parameter for numerical stability
reg_param = 1e-8;

%% FORWARD FILTERING PASS
for i = 1:t
    
    %----------------------------------------------------------------------
    % EXTRACT TIME-SPECIFIC MATRICES
    %----------------------------------------------------------------------
    
    % Observation error covariance for time i
    R = Ht((i-1)*p+1:i*p, :);
    
    % Design matrix for time i  
    H = Z((i-1)*p+1:i*p, :);
    
    % Current observation
    y_i = y(:, i);
    
    %----------------------------------------------------------------------
    % PREDICTION STEP
    %----------------------------------------------------------------------
    
    % Conditional forecast error (innovation)
    cfe = y_i - H * bp;
    
    % Innovation covariance matrix
    f = H * Vp * H' + R;
    
    % Add regularization for numerical stability
    f = f + reg_param * eye(p);
    
    %----------------------------------------------------------------------
    % LIKELIHOOD COMPUTATION (NUMERICALLY STABLE)
    %----------------------------------------------------------------------
    
    % Compute log-likelihood using Cholesky decomposition
    try
        L_f = chol(f, 'lower');
        log_det_f = 2 * sum(log(diag(L_f)));
        
        % Solve linear system efficiently
        alpha = L_f \ cfe;
        quad_form = alpha' * alpha;
        
    catch ME
        if strcmp(ME.identifier, 'MATLAB:posdef')
            % If Cholesky fails, use eigenvalue decomposition
            [V_eig, D_eig] = eig(f);
            
            % Regularize eigenvalues
            d_eig = diag(D_eig);
            d_eig = max(d_eig, reg_param);
            D_reg = diag(d_eig);
            
            % Reconstruct matrix
            f = V_eig * D_reg * V_eig';
            
            % Compute determinant and quadratic form
            log_det_f = sum(log(d_eig));
            quad_form = cfe' * (V_eig * diag(1./d_eig) * V_eig') * cfe;
        else
            rethrow(ME);
        end
    end
    
    % Accumulate log-likelihood (excluding constants)
    log_lik = log_lik + log_det_f + quad_form;
    
    %----------------------------------------------------------------------
    % UPDATE STEP (KALMAN FILTER)
    %----------------------------------------------------------------------
    
    % Compute Kalman gain using numerically stable method
    try
        % Method 1: Direct computation with Cholesky
        K = Vp * H' / f;
        
    catch
        % Method 2: More stable computation
        [U, S, V] = svd(f);
        s_inv = diag(1 ./ max(diag(S), reg_param));
        f_inv = V * s_inv * U';
        K = Vp * H' * f_inv;
    end
    
    % Updated state mean (filtered estimate)
    btt = bp + K * cfe;
    
    % Updated state covariance (Joseph form for numerical stability)
    I_KH = eye(m) - K * H;
    Vtt = I_KH * Vp * I_KH' + K * R * K';
    
    % Ensure Vtt is symmetric and positive definite
    Vtt = (Vtt + Vtt') / 2;  % Force symmetry
    
    % Regularize if needed
    [V_eig, D_eig] = eig(Vtt);
    d_eig = real(diag(D_eig));  % Take real part to handle numerical noise
    d_eig = max(d_eig, reg_param);
    Vtt = V_eig * diag(d_eig) * V_eig';
    
    %----------------------------------------------------------------------
    % PREDICTION FOR NEXT PERIOD
    %----------------------------------------------------------------------
    
    if i < t
        bp = btt;              % State prediction mean
        Vp = Vtt + Qt;         % State prediction covariance
        
        % Ensure Vp is well-conditioned
        Vp = (Vp + Vp') / 2;   % Force symmetry
        [V_eig, D_eig] = eig(Vp);
        d_eig = real(diag(D_eig));
        d_eig = max(d_eig, reg_param);
        Vp = V_eig * diag(d_eig) * V_eig';
    end
    
    %----------------------------------------------------------------------
    % STORE FILTERED ESTIMATES
    %----------------------------------------------------------------------
    
    bt(i, :) = btt';
    Vt(:, i) = reshape(Vtt, m^2, 1);
    
end

%% BACKWARD SAMPLING PASS

% Initialize storage for sampled states
bdraw = zeros(m, t);

% Sample final period from filtered distribution
try
    bdraw(:, t) = mvnrnd(btt, Vtt, 1)';
catch
    % Fallback: use Cholesky decomposition
    L = chol(Vtt, 'lower');
    bdraw(:, t) = btt + L * randn(m, 1);
end

% Backward recursion for t-1, t-2, ..., 1
for i = 1:t-1
    
    %----------------------------------------------------------------------
    % EXTRACT STATES AND COVARIANCES
    %----------------------------------------------------------------------
    
    % Future sampled state
    bf = bdraw(:, t-i+1);
    
    % Current filtered estimates
    btt = bt(t-i, :)';
    Vtt = reshape(Vt(:, t-i), m, m);
    
    %----------------------------------------------------------------------
    % BACKWARD SAMPLING EQUATIONS
    %----------------------------------------------------------------------
    
    % Prediction covariance
    f = Vtt + Qt;
    
    % Ensure f is well-conditioned
    f = (f + f') / 2;
    [V_eig, D_eig] = eig(f);
    d_eig = real(diag(D_eig));
    d_eig = max(d_eig, reg_param);
    f = V_eig * diag(d_eig) * V_eig';
    
    % Compute smoothing gain
    try
        A = Vtt / f;
    catch
        % More stable computation
        [U, S, V] = svd(f);
        s_inv = diag(1 ./ max(diag(S), reg_param));
        f_inv = V * s_inv * U';
        A = Vtt * f_inv;
    end
    
    % Innovation
    cfe = bf - btt;
    
    % Smoothed mean
    bmean = btt + A * cfe;
    
    % Smoothed covariance
    bvar = Vtt - A * Vtt;
    
    % Ensure bvar is symmetric and positive definite
    bvar = (bvar + bvar') / 2;
    [V_eig, D_eig] = eig(bvar);
    d_eig = real(diag(D_eig));
    d_eig = max(d_eig, reg_param);
    bvar = V_eig * diag(d_eig) * V_eig';
    
    %----------------------------------------------------------------------
    % SAMPLE FROM SMOOTHED DISTRIBUTION
    %----------------------------------------------------------------------
    
    try
        bdraw(:, t-i) = mvnrnd(bmean, bvar, 1)';
    catch
        % Fallback: use Cholesky decomposition
        try
            L = chol(bvar, 'lower');
            bdraw(:, t-i) = bmean + L * randn(m, 1);
        catch
            % Ultimate fallback: use SVD
            [U, S, V] = svd(bvar);
            L = U * sqrt(max(S, reg_param * eye(m)));
            bdraw(:, t-i) = bmean + L * randn(m, 1);
        end
    end
    
end

%% FINAL ADJUSTMENTS

% Transpose to get m x t matrix
bdraw = bdraw';

% Adjust log-likelihood (add constant terms)
log_lik = -0.5 * log_lik - 0.5 * t * p * log(2 * pi);

end