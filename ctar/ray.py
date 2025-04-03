import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from tqdm import trange

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Log prior for z = log(lambda) 
def log_prior_z(z, mu_lambda, sigma_lambda):
    return - ((z - mu_lambda)**2) / (2 * sigma_lambda**2)

# Log likelihood for each cell in terms of z (with lam = exp(z))
def log_likelihood_cell_z(X, Y, z, b0, b1, W, gamma):
    lam = torch.exp(z)
    # ATAC likelihood: X ~ Poisson(lam) -> X*z - exp(z)
    ll_X = X * z - lam
    # RNA likelihood: Y ~ Poisson(exp(b0 + b1*lam + W @ gamma))
    eta = b0 + b1 * lam + torch.matmul(W, gamma)
    ll_Y = Y * eta - torch.exp(eta)
    return ll_X + ll_Y

# Full log-posterior (sum over cells)
def log_posterior_z(X, Y, z, b0, b1, mu_lambda, sigma_lambda, W, gamma):
    return torch.sum(log_likelihood_cell_z(X, Y, z, b0, b1, W, gamma) + log_prior_z(z, mu_lambda, sigma_lambda))

# Optionally compile for speed.
log_posterior_z_script = torch.jit.script(log_posterior_z)

def run_map_for_pair(X_pair, Y_pair, W_pair, n_epochs=100000, lr=1e-4, check_interval=1000, threshold_b1=1e-5, patience_b1=10):
    """
    Run MAP estimation for one peak-gene pair incorporating covariates.
    
    Arguments:
      X_pair: numpy array, shape (n_cells,) of ATAC counts.
      Y_pair: numpy array, shape (n_cells,) of RNA counts.
      W_pair: numpy array, shape (n_cells, K) of covariates.
      n_epochs: maximum number of epochs.
      lr: learning rate.
      check_interval: how often (epochs) to check b1 change.
      threshold_b1: minimum absolute change in b1 required to reset counter.
      patience_b1: number of consecutive checkpoints with insufficient b1 change.
      
    Returns:
      b0_est: final estimate for intercept.
      b1_est: final estimate for effect of latent lambda.
      gamma_est: final estimate for covariate effects.
      lambda_est_final: estimated latent lambda per cell (numpy array).
      loss_history: list of loss values per epoch.
    """
    # Prior hyperparameter estimation using method-of-moments on X_pair.
    m_X = np.mean(X_pair)
    s2_X = np.var(X_pair, ddof=1)
    if m_X <= 0:
        m_X = 1.0
    sigma_lambda_est_val = np.sqrt(np.log(1 + max((s2_X - m_X), 0) / (m_X**2)))
    mu_lambda_est_val = np.log(m_X) - sigma_lambda_est_val**2 / 2
    
    mu_lambda_est = torch.tensor(mu_lambda_est_val, dtype=torch.float32, device=device)
    sigma_lambda_est = torch.tensor(sigma_lambda_est_val, dtype=torch.float32, device=device)
    X_tensor = torch.tensor(X_pair, dtype=torch.float32, device=device)
    Y_tensor = torch.tensor(Y_pair, dtype=torch.float32, device=device)
    W_tensor = torch.tensor(W_pair, dtype=torch.float32, device=device)
    
    # Initialize z = log(lambda) with a safe lower bound.
    initial_z = np.log(np.maximum(X_pair, 1e-2))
    z_est = torch.tensor(initial_z, dtype=torch.float32, device=device, requires_grad=True)
    
    # Initialize global parameters: b0 and b1.
    b0_est = torch.tensor(-1.0, dtype=torch.float32, device=device, requires_grad=True)
    b1_est = torch.tensor(0.5, dtype=torch.float32, device=device, requires_grad=True)
    
    # Initialize covariate effects gamma (K-dimensional vector).
    gamma_est = torch.zeros(W_tensor.shape[1], dtype=torch.float32, device=device, requires_grad=True)
    
    optimizer = optim.SGD([b0_est, b1_est, z_est, gamma_est], lr=lr, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.5)
    
    loss_history = []
    prev_b1 = b1_est.item()
    stopping_counter_b1 = 0
    
    progress_bar = trange(n_epochs, desc="MAP Estimation", leave=True)
    for epoch in progress_bar:
        optimizer.zero_grad()
        loss = - log_posterior_z_script(X_tensor, Y_tensor, z_est, b0_est, b1_est, mu_lambda_est, sigma_lambda_est, W_tensor, gamma_est)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([b0_est, b1_est, z_est, gamma_est], max_norm=1.0)
        optimizer.step()
        scheduler.step()
        loss_history.append(loss.item())
        
        if epoch % check_interval == 0:
            current_b1 = b1_est.item()
            delta_b1 = abs(current_b1 - prev_b1)
            progress_bar.set_postfix({"b0": b0_est.item(), "b1": current_b1})
            if delta_b1 < threshold_b1:
                stopping_counter_b1 += 1
            else:
                stopping_counter_b1 = 0
            prev_b1 = current_b1
            if stopping_counter_b1 >= patience_b1:
                progress_bar.write(f"Early stopping at epoch {epoch}: b1 change {delta_b1:.6f} below threshold for {patience_b1*check_interval} epochs.")
                break

    lambda_est_final = torch.exp(z_est).detach().cpu().numpy()
    return b0_est.item(), b1_est.item(), gamma_est.detach().cpu().numpy(), lambda_est_final, loss_history

def get_b0_b1_gamma_from_pair(X_pair, Y_pair, W_pair, n_epochs=100000, lr=1e-4):
    b0_est, b1_est, gamma_est, _, _ = run_map_for_pair(X_pair, Y_pair, W_pair, n_epochs=n_epochs, lr=lr)
    return b0_est, b1_est, gamma_est


''' Example usage:

# load covariates
covars = pd.read_csv("/projects/zhanglab/users/ana/multiome/simulations/for_ray/data/covars.tsv", sep='\t')
covars_encoded = pd.get_dummies(covars[['log_umi', 'GEX_pct_counts_mt', 'batch']], 
                                columns=['batch'], drop_first=True) #fix dimension of one-hot encoding
for col in covars_encoded.columns:
    if covars_encoded[col].dtype == 'bool':
        covars_encoded[col] = covars_encoded[col].astype(int)
W = covars_encoded.values

# example usage
pair_idx = 0
X_pair = X[:, pair_idx]
Y_pair = Y[:, pair_idx]

b0_final, b1_final, gamma_final = get_b0_b1_gamma_from_pair(X_pair, Y_pair, W, n_epochs=300000, lr=1e-4)
print("Final MAP estimates for pair", pair_idx)
print("b0 =", b0_final)
print("b1 =", b1_final)
print("gamma =", gamma_final) '''