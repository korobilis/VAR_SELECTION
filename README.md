# Bayesian Variable Selection for VAR Models

This repository contains MATLAB code for implementing Bayesian variable selection in Vector Autoregressive (VAR) models, including both standard VAR and Time-Varying Parameter VAR (TVP-VAR) models.

## Overview

Vector Autoregressive models are widely used in macroeconometrics, but they often suffer from overparameterization, especially when the number of variables or lags is large relative to the sample size. This code implements Bayesian variable selection techniques to address this problem by automatically determining which coefficients should be included in the model.

## Models Implemented

### 1. Standard VAR with Variable Selection (`VAR_SELECTION.m`)

The model can be written as:
```
Y = Z × GAMMA × BETA + e,     e ~ N(0, I ⊗ SIGMA)
```

Where:
- **Y**: Matrix of endogenous variables
- **Z**: Matrix of lagged variables  
- **GAMMA**: Diagonal matrix of 0-1 restriction indices
- **BETA**: Vector of regression coefficients
- **SIGMA**: Error covariance matrix

### 2. Time-Varying Parameter VAR with Variable Selection (`TVP_VAR_SELECTION.m`)

The model extends the standard VAR to allow for time-varying coefficients:
```
Y_t = Z_t × GAMMA × B_t + e_t,  e_t ~ N(0, SIGMA)
B_t = B_{t-1} + u_t,            u_t ~ N(0, Q)
```

Where:
- **B_t**: Time-varying coefficient vector following a random walk
- **Q**: State innovation covariance matrix

## Key Features

### Improvements Over Original Code

1. **Numerical Stability**: 
   - Log-space computation for posterior probabilities to avoid overflow/underflow
   - Regularization techniques for matrix inversions
   - SVD-based matrix decompositions for robust covariance matrix handling

2. **Enhanced Documentation**:
   - Comprehensive comments explaining each step
   - Clear variable naming and organization
   - Educational structure for learning purposes

3. **Better Diagnostics**:
   - MCMC convergence diagnostics
   - Model comparison metrics
   - Computational efficiency monitoring

4. **Robust Implementation**:
   - Error handling for numerical issues
   - Automatic regularization when needed
   - Improved random number generation

## Files Description

### Main Scripts
- **`VAR_SELECTION.m`**: Main script for standard VAR variable selection
- **`TVP_VAR_SELECTION.m`**: Main script for TVP-VAR variable selection

### Supporting Functions
- **`bernoullirnd.m`**: Generate Bernoulli random variables
- **`carter_kohn.m`**: Carter-Kohn algorithm for sampling time-varying parameters
- **`mlag2.m`**: Create lagged matrices
- **`mvnrnd.m`**: Multivariate normal random number generator
- **`simvardgp.m`**: Simulate data from standard VAR
- **`tvpvarsim.m`**: Simulate data from TVP-VAR
- **`wish.m`**: Wishart random number generator

## Usage

### Basic Usage

1. **For Standard VAR Selection**:
```matlab
% Run the main script
VAR_SELECTION

% Key results will be displayed and all variables saved in workspace
% Examine posterior inclusion probabilities
disp('Inclusion probabilities:');
disp(gamma_mean_post);

% Plot time series of coefficients
plot(beta_draws(:,1:5));
title('Posterior Draws of First 5 Coefficients');
```

2. **For TVP-VAR Selection**:
```matlab
% Run the main script  
TVP_VAR_SELECTION

% Examine time-varying parameters
figure;
subplot(2,1,1);
plot(Bt_postmean(1,:));
title('Time-Varying Parameter 1');

subplot(2,1,2);
plot(gamma_mean);
title('Posterior Inclusion Probabilities');
```

### Using Your Own Data

To use your own data instead of simulated data, replace the data generation section:

```matlab
% Replace this section in the scripts:
% [Y, PHI_true] = simvardgp();

% With your own data:
Y = your_data_matrix;  % Should be T x m matrix
```

## Parameters You Can Modify

### In `VAR_SELECTION.m`:
- `plag`: Number of lags (default: 1)
- `nsave`: Number of posterior draws to save (default: 5000)
- `nburn`: Number of burn-in draws (default: 5000)
- `p_j`: Prior inclusion probabilities (default: 0.5)
- Prior parameters for beta and sigma

### In `TVP_VAR_SELECTION.m`:
- `nrep`: Number of posterior draws (default: 50)
- `nburn`: Number of burn-in draws (default: 50)
- `k_Q`: Scaling factor for state covariance prior (default: 0.01)
- `p_j`: Prior inclusion probabilities (default: 0.5)

## Understanding the Output

### Standard VAR Results
The script displays:
- **Inclusion Probabilities**: Posterior probability that each coefficient is non-zero
- **Posterior Means**: Bayesian estimates of coefficients
- **Comparison with OLS**: Classical estimates for reference
- **Model Diagnostics**: MCMC efficiency measures

### TVP-VAR Results  
The script displays:
- **Time-varying inclusion probabilities**
- **Final period coefficient values**
- **Model diagnostics including log-likelihood**
- **Covariance matrix condition numbers**

## Model Selection and Interpretation

### Interpreting Inclusion Probabilities
- **Prob > 0.8**: Strong evidence for inclusion
- **0.2 < Prob < 0.8**: Uncertain evidence  
- **Prob < 0.2**: Strong evidence for exclusion

### Model Comparison
Compare models using:
- **Log marginal likelihood**: Higher values indicate better fit
- **Posterior inclusion probabilities**: Identify important relationships
- **Out-of-sample forecasting**: Ultimate test of model performance

## Theoretical Background

This code implements the methodology from:

1. **Korobilis, D. (2013)**. "VAR forecasting using Bayesian variable selection." *Journal of Applied Econometrics*, 28(2), 204-230.

2. **Primiceri, G. E. (2005)**. "Time varying structural vector autoregressions and monetary policy." *The Review of Economic Studies*, 72(3), 821-852.

### Gibbs Sampling Algorithm

Both models use Gibbs sampling with the following steps:

1. **Sample coefficients** (β or B_t) from their posterior distribution
2. **Sample inclusion indicators** (γ) using Bernoulli distributions  
3. **Sample covariance matrices** (Σ, Q) from inverse-Wishart distributions

The key innovation is step 2, where we use spike-and-slab priors to perform automatic variable selection.

## Tips for Successful Analysis

1. **Start Small**: Begin with fewer variables and shorter time series
2. **Check Convergence**: Examine trace plots of key parameters
3. **Sensitivity Analysis**: Try different prior specifications
4. **Validate Results**: Use out-of-sample forecasting to assess model performance
5. **Compare Models**: Run both standard and TVP versions to see if time variation matters

## Troubleshooting

### Common Issues

1. **Numerical Problems**: 
   - Reduce `k_Q` parameter in TVP-VAR
   - Increase regularization in matrix inversions

2. **Slow Convergence**:
   - Increase burn-in period
   - Check for explosive parameter values

3. **Memory Issues**:
   - Reduce `nsave` parameter
   - Use thinning (`nthin > 1`)

### Error Messages

- **"Explosive model generated"**: Rerun data generation or check your data
- **"Matrix is singular"**: Check for perfect collinearity in your data
- **"Cholesky decomposition failed"**: Matrix conditioning issue, regularization will be applied automatically

## Extensions and Future Work

Possible extensions include:
- **Stochastic volatility**: Allow time-varying error covariances
- **Structural identification**: Impose economic restrictions
- **Factor models**: Combine with factor structures for high-dimensional VARs
- **Forecasting**: Implement formal forecasting procedures

## Citation

If you use this code in your research, please cite:

```
@article{korobilis2013var,
  title={VAR forecasting using Bayesian variable selection},
  author={Korobilis, Dimitris},
  journal={Journal of Applied Econometrics},
  volume={28},
  number={2},
  pages={204--230},
  year={2013},
  publisher={Wiley Online Library}
}
```

## License

This code is provided for educational and research purposes. Please refer to the original author's licensing terms.

## Contact

For questions about the original methodology, contact the original author. For questions about this improved implementation, please open an issue in this repository.

---

**Note**: This is educational code designed for learning Bayesian econometric methods. For production use, consider additional robustness checks and validation procedures.
