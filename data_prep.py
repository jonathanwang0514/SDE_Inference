import numpy as np
import torch
from loss import PDE_loss, pde_residual
from utils import trace_diag_diffusion, terminal_func
from torch.utils.data import Dataset, DataLoader, TensorDataset

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def sample_collocation(n, t_start, t_end, K):
    t = torch.rand(n, 1) * (t_end - t_start)
    x = torch.rand(n, 1) * 2 * K  # e.g., [0, 2K]
    return t.to(device), x.to(device)

#Sampling terminal points at T below. X_k = Y_k - epsilon, where epsilon is N(0, \sigma^2)

def sample_terminal(n, t, K):
    t = torch.full((n, 1), t)
    x = torch.rand(n, 1) * 2 * K  
    return t.to(device), x.to(device)


# class PDE_Dataset(Dataset):
#     def __init__(self, x, t):
#         self.t = t
#         self.x = x

#     def __len__(self):
#         return len(self.t)

#     def __getitem__(self, idx):
#         return self.x[idx], self.t[idx]
class P2PDEDataset(Dataset):
    """
    Returns (t, x, theta).
    Use EITHER param_ranges=[(lo,hi), ...]  OR  param_grid=tensor[M, param_dim].
    """
    def __init__(self, N, t_start, t_end, x_min, x_max,
                 param_ranges=None, param_grid=None, dtype=torch.float32, device="cpu"):
        self.N = N
        self.t = (t_start + (t_end - t_start) * torch.rand(N, 1)).to(device, dtype)
        self.x = (x_min  + (x_max  - x_min)  * torch.rand(N, 1)).to(device, dtype)

        if param_grid is not None:
            grid = torch.as_tensor(param_grid, dtype=dtype, device=device)
            idx = torch.randint(grid.shape[0], (N,), device=device)
            self.theta = grid[idx]
        elif param_ranges is not None:
            lows  = torch.tensor([a for a,_ in param_ranges], dtype=dtype, device=device)
            highs = torch.tensor([b for _,b in param_ranges], dtype=dtype, device=device)
            u = torch.rand(N, len(param_ranges), device=device, dtype=dtype)
            self.theta = lows + u * (highs - lows)
        else:
            raise ValueError("Provide param_ranges or param_grid.")

    def __len__(self): return self.N

    def __getitem__(self, idx):
        return self.t[idx], self.x[idx], self.theta[idx]
    
    
class PDEDataset(Dataset):
    def __init__(self, N, t_start, t_end, x_min, x_max, d, dtype, device):
        """
        Docstring for __init__
        
        :param N: # of datapoints
        :param t_start: initial time
        :param t_end: terminal time
        :param x_min: lower bound of x
        :param x_max: upper bound of x
        :param d: dimension of x 
        :param dtype: data type
        :param device: device
        """
        self.t = torch.rand(N, 1, dtype=dtype, device=device) * (t_end - t_start) + t_start
        self.x = torch.rand(N, d, dtype=dtype, device=device) * (x_max - x_min) + x_min

    def __len__(self):
        return len(self.t)

    def __getitem__(self, idx):
        return self.t[idx], self.x[idx]
    
    
class AdaptivePDEDataset(Dataset):
    def __init__(self, model, N_candidates, N_select, t_start, t_end, x_max, x_min, mu_func, sigma_func, device='cpu'):
        self.mu_func = mu_func
        self.sigma_func = sigma_func
        self.device = device
        model_cpu = model.to(torch.device("cpu"), torch.float32)

        # Step 1: Generate candidate samples uniformly
        t_cand = torch.rand(N_candidates, 1) * (t_end - t_start) + t_start
        x_cand = torch.rand(N_candidates, 1) * (x_max - x_min) + x_min

        t_cand = t_cand.to(dtype=torch.float32)
        x_cand = x_cand.to(dtype=torch.float32)

        t_cand.requires_grad_(True)
        x_cand.requires_grad_(True)
        # Step 2: Compute PDE residuals
        res = pde_residual(model_cpu, x_cand, t_cand)

        # Step 3: Select top-N points with highest residual
        _, idx = torch.topk(res.squeeze(), N_select)
        self.t = t_cand[idx].detach().clone()
        self.x = x_cand[idx].detach().clone()

    def __len__(self):
        return len(self.t)

    def __getitem__(self, idx):
        return self.t[idx], self.x[idx]

class TerminalDataset(Dataset):
    def __init__(self, N, t_terminal, x_max, x_min, d):
        self.t = torch.full((N, 1), t_terminal)
        self.x = torch.rand(N, d) * (x_max - x_min) + x_min

    def __len__(self):
        return len(self.t)

    def __getitem__(self, idx):
        return self.t[idx], self.x[idx]
    
class AdaptiveTerminalDataset(Dataset):
    def __init__(self, model, N, t_terminal, x_min, x_max, y_k, r, device="cpu"):
        """
        Likelihood-weighted adaptive terminal dataset.

        Args:
            N: number of points
            t_terminal: scalar float (time value at boundary)
            x_min, x_max: spatial domain bounds
            y_k: observation value at time t_terminal
            r: observation noise std
            device: "cpu" or "cuda"
        """
        self.device = device
        model = model.to(torch.device("cpu"), torch.float32)
        self.t = torch.full((N, 1), t_terminal, device=device)

        # === Likelihood-weighted sampling over x ===
        x_grid_np = np.linspace(x_min, x_max, 3000)
        x_grid = torch.tensor(x_grid_np, dtype=torch.float32).unsqueeze(1).to(device)  # shape (3000, 1)
        t_grid = torch.full_like(x_grid, t_terminal).to(device)

        with torch.no_grad():
            w_xt_plus = model(x_grid, t_grid).squeeze()
        
        
        likelihood = torch.exp((y_k * x_grid.squeeze() - 0.5 * x_grid.squeeze()**2) / r**2)
        posterior_unnorm = w_xt_plus * likelihood

# 2. Apply mask: must be positive and finite
        valid_mask = (posterior_unnorm > 0) & torch.isfinite(posterior_unnorm)
        x_grid_valid = x_grid[valid_mask]
        posterior_valid = posterior_unnorm[valid_mask]

# 3. Log transform to boost sharp peaks & suppress tiny tails
        log_post = torch.log(posterior_valid + 1e-20)  # log-safe

        # print("== Debug Info ==")
        # print("x_grid shape:", x_grid.shape)
        # print("w_xt_plus shape:", w_xt_plus.shape)
        # print("w_xt_plus min:", w_xt_plus.min().item())
        # print("w_xt_plus max:", w_xt_plus.max().item())
        # print("w_xt_plus contains NaNs:", torch.isnan(w_xt_plus).any().item())

        # print("likelihood min:", likelihood.min().item())
        # print("likelihood max:", likelihood.max().item())
        # print("likelihood contains NaNs:", torch.isnan(likelihood).any().item())

        # print("posterior_unnorm contains NaNs:", torch.isnan(posterior_unnorm).any().item())
        # print("posterior_unnorm min:", posterior_unnorm.min().item())
        # print("posterior_unnorm max:", posterior_unnorm.max().item())
        # print("valid_mask sum:", valid_mask.sum().item())


        log_post -= log_post.max()  # subtract max for numerical stability
        posterior_sharp = torch.exp(log_post)  # re-exponentiate

# 4. Normalize
        posterior_probs = posterior_sharp / (torch.sum(posterior_sharp) + 1e-12)

# 5. Final safeguard
        # posterior_probs = torch.clamp(posterior_probs, min=1e-12)
        posterior_probs /= posterior_probs.sum()

# 6. Sample with torch.multinomial safely
        indices = torch.multinomial(posterior_probs, N, replacement=True)
        self.x = x_grid_valid[indices].detach().clone()
        # posterior_unnorm = w_xt_plus * likelihood
        # valid_mask = posterior_unnorm > 0
        # x_grid_valid = x_grid[valid_mask]
        # posterior_valid = posterior_unnorm[valid_mask]
        # # posterior_unnorm = torch.nan_to_num(posterior_unnorm, nan=0.0, posinf=0.0, neginf=0.0)
        # # posterior_unnorm = torch.clamp(posterior_unnorm, min=1e-10)
        # area = torch.trapz(posterior_valid, x_grid_valid.squeeze())
        # posterior_probs = posterior_valid / (area + 1e-8)

        # cdf = torch.cumsum(posterior_probs, dim=0)
        # cdf = cdf / (cdf[-1] + 1e-8)

        # # Generate uniform random numbers for inverse transform sampling
        # u = torch.rand(N)
        # indices = torch.searchsorted(cdf, u)
        # indices = torch.clamp(indices, max=len(x_grid_valid) - 1)

        # # Final samples
        # self.x = x_grid_valid[indices].detach().clone()

        # posterior_probs = torch.clamp(posterior_probs, min=1e-10)
        # posterior_probs = posterior_probs / posterior_probs.sum()  # for multinomial


        # weights = np.exp((y_k * x_grid - 0.5 * x_grid**2) / r**2)
        # weights /= np.trapz(weights, x_grid)
   
        # x_probs = torch.tensor(weights, dtype=torch.float32)
        # x_vals = torch.tensor(x_grid, dtype=torch.float32)

        # Sample x values based on likelihood weights
        # idx = torch.multinomial(posterior_probs, N, replacement=True)
        # self.x = x_grid_valid[idx].detach().clone()

    def __len__(self):
        return len(self.t)

    def __getitem__(self, idx):
        return self.t[idx], self.x[idx]    
    
class AdaptiveTerminalDataset1(Dataset):
    def __init__(self, N, t_terminal, x_min, x_max, y_k, r, device="cpu"):
        """
        Likelihood-weighted adaptive terminal dataset.

        Args:
            N: number of points
            t_terminal: scalar float (time value at boundary)
            x_min, x_max: spatial domain bounds
            y_k: observation value at time t_terminal
            r: observation noise std
            device: "cpu" or "cuda"
        """
        self.device = device
        self.t = torch.full((N, 1), t_terminal, device=device)

        # === Likelihood-weighted sampling over x ===
        x_grid = np.linspace(x_min, x_max, 3000)
        weights = np.exp((y_k * x_grid - 0.5 * x_grid**2) / r**2)
        weights /= np.trapz(weights, x_grid)
   
        x_probs = torch.tensor(weights, dtype=torch.float32)
        x_vals = torch.tensor(x_grid, dtype=torch.float32)

        # Sample x values based on likelihood weights
        idx = torch.multinomial(x_probs, N, replacement=True)
        self.x = x_vals[idx].detach().clone()

    def __len__(self):
        return len(self.t)

    def __getitem__(self, idx):
        return self.t[idx], self.x[idx]    

def simulate_gbm(k, lamda, T, N, noise_std=0.15, n_paths=1):
    dt = T / N
    t = torch.linspace(0, T, N + 1, device=device)
    paths = torch.zeros((n_paths, N + 1), device=device)
    paths[:, 0] = 1.0
    # paths[:, 0] = torch.exp(mu + sigma * torch.randn(n_paths))

    for i in range(1, N + 1):
        Z = torch.randn(n_paths, device=device)
        increment = (k - 0.5 * lamda ** 2) * dt + lamda * torch.sqrt(torch.tensor(dt, device=device)) * Z
        paths[:, i] = paths[:, i - 1] * torch.exp(increment)

    epsilon = noise_std * torch.randn_like(paths)
    Y = paths + epsilon  # add noise to each simulated datum point

    return t, paths, Y


class MixedTerminalDataset(Dataset):
    def __init__(self, model, N, t_terminal, x_min, x_max,
                 y_k=None, r=None, adaptive_frac=0.5, device="cpu"):
        """
        Dataset mixing uniform and adaptive (posterior-weighted) terminal samples.

        Args:
            model         : PINN model for adaptive weighting (ignored if adaptive_frac=0)
            N             : total number of points
            t_terminal    : scalar float (time value at boundary)
            x_min, x_max  : spatial bounds
            y_k           : observation value at t_terminal (needed for adaptive)
            r             : observation noise std (needed for adaptive)
            adaptive_frac : fraction of adaptive samples (0 to 1)
            device        : "cpu" or "cuda"
        """
        self.device = device
        self.t = torch.full((N, 1), t_terminal, device=device)

        N_adapt = int(N * adaptive_frac)
        N_unif = N - N_adapt

        # --- Uniform samples ---
        if N_unif > 0:
            x_unif = torch.rand(N_unif, 1, device=device) * (x_max - x_min) + x_min
        else:
            x_unif = torch.empty(0, 1, device=device)

        # --- Adaptive samples ---
        if N_adapt > 0 and y_k is not None and r is not None:
            model = model.to("cpu").float().eval()  # eval mode for sampling
            x_grid = torch.linspace(x_min, x_max, 3000).unsqueeze(1)
            t_grid = torch.full_like(x_grid, t_terminal)

            with torch.no_grad():
                w_xt_plus = model(x_grid, t_grid).squeeze()

            likelihood = torch.exp((y_k * x_grid.squeeze() - 0.5 * x_grid.squeeze()**2) / r**2)
            posterior_unnorm = w_xt_plus * likelihood

            # Filter valid
            valid_mask = (posterior_unnorm > 0) & torch.isfinite(posterior_unnorm)
            x_grid_valid = x_grid[valid_mask]
            posterior_valid = posterior_unnorm[valid_mask]

            # Peak sharpening
            log_post = torch.log(posterior_valid + 1e-20)
            log_post -= log_post.max()
            posterior_sharp = torch.exp(log_post)

            # Normalize
            posterior_probs = posterior_sharp / (posterior_sharp.sum() + 1e-12)

            # Sample adaptively
            idx = torch.multinomial(posterior_probs, N_adapt, replacement=True)
            x_adapt = x_grid_valid[idx].to(device)
        else:
            x_adapt = torch.empty(0, 1, device=device)

        # --- Merge ---
        self.x = torch.cat([x_unif, x_adapt], dim=0)

        # Optional shuffle so uniform/adaptive aren’t ordered
        perm = torch.randperm(len(self.x))
        self.x = self.x[perm]
        self.t = self.t[perm]

    def __len__(self):
        return len(self.t)

    def __getitem__(self, idx):
        return self.t[idx], self.x[idx]
    
class BoundaryDataset(Dataset):
    """
    Uniform-in-time boundary sampler for x in {x_min, x_max}.
    side: 'left', 'right', or 'both'
    balanced=True ensures ~50/50 split when side='both'.
    """
    def __init__(self, N, t_start, t_end, x_min, x_max, side='both', balanced=True):
        assert side in ('left', 'right', 'both')
        self.t = torch.rand(N, 1) * (t_end - t_start) + t_start

        if side == 'left':
            self.x = torch.full((N, 1), float(x_min))
        elif side == 'right':
            self.x = torch.full((N, 1), float(x_max))
        else:
            if balanced:
                nL = N // 2
                nR = N - nL
                xL = torch.full((nL, 1), float(x_min))
                xR = torch.full((nR, 1), float(x_max))
                self.x = torch.cat([xL, xR], dim=0)
                # Shuffle so batches contain a mix of left/right points
                perm = torch.randperm(N)
                self.x = self.x[perm]
                self.t = self.t[perm]
            else:
                mask = (torch.rand(N, 1) < 0.5)
                self.x = torch.where(mask, torch.full_like(self.t, float(x_min)),
                                           torch.full_like(self.t, float(x_max)))

    def __len__(self):
        return len(self.t)

    def __getitem__(self, idx):
        # Note: set requires_grad on x *inside your loss* (e.g., Neumann) not here.
        return self.t[idx], self.x[idx]    

class PINNDataFactory:
    """
    Factory that (re)generates DataLoaders for interval collocation points
    and knot (terminal) points given time_knots and box/param settings.
    Call it once per epoch to get fresh loaders with new random samples.
    """
    def __init__(
        self,
        obs_times,                 # 1D tensor/list of monotonically increasing times, length = N+1
        x_min, x_max, d,               # scalars or 0-D tensors
        samples_per_interval,
        batch_size,
        shuffle=True,
        drop_last=True,
        dtype=None,
        device=None,
        dt_epsilon=0.1             # used only for the final tiny interval
    ):
        # store immutable config
        self.time_knots = obs_times
        self.x_min = float(x_min) if hasattr(x_min, "item") else x_min
        self.x_max = float(x_max) if hasattr(x_max, "item") else x_max
        self.samples_per_interval = int(samples_per_interval)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.dtype = dtype
        self.device = device
        self.dt_epsilon = float(dt_epsilon)
        self.d = d
        # sanity
        if len(obs_times) < 2:
            raise ValueError("time_knots must have length >= 2 (N+1).")
    
    def __call__(self):
        """
        Build and return (interval_loaders, knot_loaders) with fresh datasets.
        """
        tk = self.time_knots
        N = len(tk) - 1  # number of intervals
        interval_loaders = []
        knot_loaders = []

        for i in range(N + 1):
            # t0 = t_i, t1 = t_{i+1}, with a small extension for last interval
            t0 = float(tk[i]) if hasattr(tk[i], "item") else tk[i]
            if i < N:
                t1 = float(tk[i + 1]) if hasattr(tk[i + 1], "item") else tk[i + 1]
            else:
                last = float(tk[-1]) if hasattr(tk[-1], "item") else tk[-1]
                t1 = last + self.dt_epsilon

            # Interval collocation dataset
            ds_interval = PDEDataset(
                N=self.samples_per_interval,
                t_start=t0, t_end=t1,
                x_min=self.x_min, x_max=self.x_max, d=self.d,
                dtype=self.dtype, device=self.device
            )
            dl_interval = DataLoader(
                ds_interval,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                drop_last=self.drop_last
            )
            interval_loaders.append(dl_interval)

            # Knot dataset at the *end* of the interval (t1_exact = t_{i+1} when i < N; else last knot)

            ds_knot = PDEDataset(
                N=self.samples_per_interval,
                t_start=t1, t_end=t1,   # single-time slice
                x_min=self.x_min, x_max=self.x_max, d=self.d,
                dtype=self.dtype, device=self.device
            )
            dl_knot = DataLoader(
                ds_knot,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                drop_last=self.drop_last
            )
            knot_loaders.append(dl_knot)

        return interval_loaders, knot_loaders   

# class PINNAdaptiveDataFactory:
#     def __init__(
#         self,
#         obs_times,
#         x_min, x_max, d,
#         samples_per_interval,
#         batch_size,
#         drift_func,
#         diff_func,
#         y_obs,           # Observation values for terminal adaptive sampling
#         r_noise,         # Observation noise std
#         solve_for="logw",
#         shuffle=True,
#         drop_last=True,
#         dtype=torch.float32,
#         device='cpu',
#         dt_epsilon=0.1,
#         oversample_factor=10
#     ):
#         self.time_knots = obs_times
#         self.x_min, self.x_max, self.d = float(x_min), float(x_max), d
#         self.samples_per_interval = int(samples_per_interval)
#         self.batch_size = int(batch_size)
#         self.drift_func, self.diff_func = drift_func, diff_func
#         self.y_obs = y_obs  # Should correspond to the terminal values at each knot
#         self.r_noise = r_noise
#         self.solve_for = solve_for
#         self.shuffle, self.drop_last = shuffle, drop_last
#         self.dtype, self.device = dtype, device
#         self.dt_epsilon = dt_epsilon
#         self.oversample_factor = oversample_factor

#     def __call__(self, model_list, adaptive=True):
#         tk = self.time_knots
#         N = len(tk) - 1
#         interval_loaders, knot_loaders = [], []

#         for i in range(N + 1):
#             t0 = float(tk[i])
#             t1 = float(tk[i+1]) if i < N else float(tk[-1]) + self.dt_epsilon
            
#             # --- 1. PDE INTERVAL SAMPLING (Adaptive Residual-based) ---
#             n_cand = self.samples_per_interval * (self.oversample_factor if adaptive else 1)
#             t_cand = (torch.rand(n_cand, 1, device=self.device) * (t1 - t0) + t0).to(self.dtype)
#             x_cand = (torch.rand(n_cand, self.d, device=self.device) * (self.x_max - self.x_min) + self.x_min).to(self.dtype)

#             if adaptive:
#                 current_model = model_list[i]
#                 current_model.eval() 
#                 with torch.set_grad_enabled(True):
#                     res_vec = pde_residual(current_model, x_cand, t_cand, self.drift_func, self.diff_func, self.solve_for)
#                     _, idx = torch.topk(res_vec.abs().squeeze(), self.samples_per_interval)
#                 idx = idx.to(t_cand.device)
#                 t_pde, x_pde = t_cand[idx].detach(), x_cand[idx].detach()
#             else:
#                 t_pde, x_pde = t_cand[:self.samples_per_interval], x_cand[:self.samples_per_interval]

#             interval_loaders.append(DataLoader(TensorDataset(t_pde, x_pde), batch_size=self.batch_size, shuffle=self.shuffle))

#             # --- 2. KNOT/TERMINAL SAMPLING (Adaptive Likelihood-weighted) ---
#             if adaptive:
#                 target_idx = i + 1 if i < N else i 
#                 y_target = self.y_obs[0, target_idx].to(self.device)
                
#                 # 1. Generate Candidates (centered around the observation to stay physically relevant)
#                 n_cand = self.samples_per_interval * self.oversample_factor
#                 x_cand_knot = y_target + torch.randn(n_cand, self.d, device=self.device) * self.r_noise
#                 t_cand_knot = torch.full((n_cand, 1), t1, device=self.device, dtype=self.dtype)

#                 # # 2. Evaluate the Jump Residual using BOTH models
#                 model_list[i].eval()
#                 if i < N:
#                     model_list[i+1].eval()

#                 # We don't need gradients to calculate the jump error, just forward passes
#                 with torch.no_grad():
#                     h = model_list[i](x_cand_knot, t_cand_knot)
                    
#                     if i < N:
#                         h_plus = model_list[i+1](x_cand_knot, t_cand_knot)
#                         # Same math as your loss function
#                         log_lik_y = (y_target * x_cand_knot - 0.5 * x_cand_knot**2).sum(dim=1, keepdim=True) / (self.r_noise**2)
#                         jump_residual = (h - h_plus - log_lik_y).abs()
#                     else:
#                         # For the very last interval, we use the terminal function
#                         terminal_value = terminal_func(x_cand_knot) # Ensure terminal_func is passed to __init__
#                         jump_residual = (h - terminal_value).abs()

#                 # 3. Select the points where the networks are failing the jump condition the most
#                 _, idx = torch.topk(jump_residual.squeeze(), self.samples_per_interval)
#                 idx = idx.to(x_cand_knot.device)
#                 x_knot = x_cand_knot[idx].detach()

#             else:
#                 x_knot = (torch.rand(self.samples_per_interval, self.d, device=self.device) * (self.x_max - self.x_min) + self.x_min).to(self.dtype)
#             t_knot = torch.full((self.samples_per_interval, 1), t1, device=self.device, dtype=self.dtype)
#             knot_loaders.append(DataLoader(TensorDataset(t_knot, x_knot), batch_size=self.batch_size, shuffle=self.shuffle))

#         return interval_loaders, knot_loaders


class PINNAdaptiveDataFactory:
    def __init__(
        self,
        obs_times,
        x_min, x_max, d,
        samples_per_interval,
        batch_size,
        drift_func,
        diff_func,
        y_obs,           
        r_noise,            # Added this so the final knot evaluation works
        solve_for="logw",
        shuffle=True,
        drop_last=True,
        dtype=torch.float32,
        device='cpu',
        dt_epsilon=0.1,
        oversample_factor=10
    ):
        self.time_knots = obs_times
        self.x_min, self.x_max, self.d = float(x_min), float(x_max), d
        self.samples_per_interval = int(samples_per_interval)
        self.batch_size = int(batch_size)
        self.drift_func, self.diff_func = drift_func, diff_func
        self.y_obs = y_obs  
        self.r_noise = r_noise
        self.solve_for = solve_for
        self.shuffle, self.drop_last = shuffle, drop_last
        self.dtype, self.device = dtype, device
        self.dt_epsilon = dt_epsilon
        self.oversample_factor = oversample_factor

    def __call__(self, model_list, adaptive=True):
        tk = self.time_knots
        N = len(tk) - 1
        interval_loaders, knot_loaders = [], []

        # Define the 80/20 split quantities
        n_adapt = int(self.samples_per_interval * 0.8)
        n_uni = self.samples_per_interval - n_adapt

        for i in range(N + 1):
            t0 = float(tk[i])
            t1 = float(tk[i+1]) if i < N else float(tk[-1]) + self.dt_epsilon
            
            # --- 1. PDE INTERVAL SAMPLING (80% Adaptive / 20% Uniform) ---
            if adaptive:
                current_model = model_list[i]
                orig_mode = current_model.training # Save mode
                current_model.eval() 
                
                # A. Generate and select the 80% Adaptive Points
                n_cand_pde = n_adapt * self.oversample_factor
                t_cand_pde = (torch.rand(n_cand_pde, 1, device=self.device) * (t1 - t0) + t0).to(self.dtype)
                x_cand_pde = (torch.rand(n_cand_pde, self.d, device=self.device) * (self.x_max - self.x_min) + self.x_min).to(self.dtype)

                with torch.set_grad_enabled(True):
                    # Make sure pde_residual is defined globally or imported
                    res_vec = pde_residual(current_model, x_cand_pde, t_cand_pde, self.drift_func, self.diff_func, self.solve_for)
                    _, idx = torch.topk(res_vec.abs().squeeze(), n_adapt)
                
                current_model.train(orig_mode) # Restore mode
                
                idx = idx.to(t_cand_pde.device)
                t_pde_adapt, x_pde_adapt = t_cand_pde[idx].detach(), x_cand_pde[idx].detach()

                # B. Generate the 20% Uniform Background Points
                t_pde_uni = (torch.rand(n_uni, 1, device=self.device) * (t1 - t0) + t0).to(self.dtype)
                x_pde_uni = (torch.rand(n_uni, self.d, device=self.device) * (self.x_max - self.x_min) + self.x_min).to(self.dtype)

                # C. Combine
                t_pde = torch.cat([t_pde_adapt, t_pde_uni], dim=0)
                x_pde = torch.cat([x_pde_adapt, x_pde_uni], dim=0)

            else:
                t_pde = (torch.rand(self.samples_per_interval, 1, device=self.device) * (t1 - t0) + t0).to(self.dtype)
                x_pde = (torch.rand(self.samples_per_interval, self.d, device=self.device) * (self.x_max - self.x_min) + self.x_min).to(self.dtype)

            interval_loaders.append(DataLoader(TensorDataset(t_pde, x_pde), batch_size=self.batch_size, shuffle=self.shuffle))

            # --- 2. KNOT/TERMINAL SAMPLING (80% Active Likelihood / 20% Uniform) ---
            if adaptive:
                target_idx = i + 1 if i < N else i 
                y_target = self.y_obs[0, target_idx].to(self.device)
                
                orig_mode_i = model_list[i].training
                orig_mode_i_plus = model_list[i+1].training if i < N else False
                model_list[i].eval()
                if i < N: model_list[i+1].eval()

                # A. Generate and select the 80% Adaptive Points
                n_cand_knot = n_adapt * self.oversample_factor
                x_cand_knot = y_target + torch.randn(n_cand_knot, self.d, device=self.device) * self.r_noise
                t_cand_knot = torch.full((n_cand_knot, 1), t1, device=self.device, dtype=self.dtype)

                with torch.no_grad():
                    h = model_list[i](x_cand_knot, t_cand_knot)
                    if i < N:
                        h_plus = model_list[i+1](x_cand_knot, t_cand_knot)
                        log_lik_y = (y_target * x_cand_knot - 0.5 * x_cand_knot**2).sum(dim=1, keepdim=True) / (self.r_noise**2)
                        jump_residual = (h - h_plus - log_lik_y).abs()
                    else:
                        terminal_value = terminal_func(x_cand_knot) 
                        jump_residual = (h - terminal_value).abs()

                model_list[i].train(orig_mode_i) # Restore modes
                if i < N: model_list[i+1].train(orig_mode_i_plus)

                _, idx = torch.topk(jump_residual.squeeze(), n_adapt)
                idx = idx.to(x_cand_knot.device)
                x_knot_adapt = x_cand_knot[idx].detach()

                # B. Generate the 20% Uniform Background Points
                x_knot_uni = (torch.rand(n_uni, self.d, device=self.device) * (self.x_max - self.x_min) + self.x_min).to(self.dtype)

                # C. Combine
                x_knot = torch.cat([x_knot_adapt, x_knot_uni], dim=0)

            else:
                x_knot = (torch.rand(self.samples_per_interval, self.d, device=self.device) * (self.x_max - self.x_min) + self.x_min).to(self.dtype)
            
            t_knot = torch.full((self.samples_per_interval, 1), t1, device=self.device, dtype=self.dtype)
            knot_loaders.append(DataLoader(TensorDataset(t_knot, x_knot), batch_size=self.batch_size, shuffle=self.shuffle))

        return interval_loaders, knot_loaders