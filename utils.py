import torch 
from torch.autograd import grad
import numpy as np
import scipy
from scipy.integrate import simpson
from sklearn.linear_model import Ridge

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def grad_net(y, x, order=1):
    weights = torch.ones_like(y)
    if order == 1:
        g = grad(outputs=y, inputs=x, grad_outputs=weights, create_graph=True)[0]
        return g
    elif order == 2:
        g_1 = grad(outputs=y, inputs=x, grad_outputs=weights, create_graph=True)[0]
        g_2 = grad(outputs=g_1, inputs=x, grad_outputs=weights, create_graph=True)[0]
        return g_2
    else:
        raise NotImplementedError


def batch_grad_net(batch_y, batch_x, order=1):
    return torch.stack([grad_net(y, x, order) for y, x in zip(batch_y, batch_x)])


def get_difference(x, dt, order=1):
    global difference
    if order > 0:
        l = x.size()[0]
        difference = torch.tensor([(x[i] - x[i - 1]) / dt if i > 0 else x[i] / dt for i in range(0, l)])
        return get_difference(difference, dt, order - 1)
    else:
        return difference


def batch_get_difference(batch_x, dt, order=1):
    r = batch_x.size()[0]
    return torch.stack([get_difference(batch_x[i], dt, order) for i in range(r)]).unsqueeze(2)

def get_knots(T, N):
    knots = []
    for i in range(N+1):
        t_i = round(i * (T / N), 5)
        knots.append(t_i)

    return knots

def Simpson_rule(model, t, y, x_min, x_max, batch_size, n_points, sigma):
    device = t.device
    x = torch.linspace(x_min, x_max, n_points, device=device).unsqueeze(0).repeat(batch_size, 1)
    t = t.view(batch_size, 1).expand(-1, n_points)            

    # input_flat = torch.cat([x.reshape(-1, 1), t.reshape(-1, 1)], dim=-1)

    with torch.no_grad():
        u_flat = model(x = x.reshape(-1, 1), t = t.reshape(-1, 1)) 
        u = u_flat.view(batch_size, n_points)
        u = u * torch.exp(y * (1/sigma) * x - 0.5 * (x**2) * (1/sigma)) 
    
    x_np = x.cpu().numpy()
    u_np = u.cpu().numpy()

    integrals = simpson(u_np, x=x_np, axis=-1)

    return integrals

def trapz_rule(model, t, y, x_min, x_max, batch_size, n_points, sigma):
    device = t.device
    # Create a 1D tensor of x values
    x = torch.linspace(x_min, x_max, n_points, device=device)
    # Expand x to match the batch size
    x = x.unsqueeze(0).expand(batch_size, -1)  # Shape: (batch_size, n_points)
    # Expand t to match the shape of x
    t = t.view(batch_size, 1).expand(-1, n_points)  # Shape: (batch_size, n_points)

    with torch.no_grad():
        # Flatten x and t for model input
        x_flat = x.reshape(-1, 1)
        t_flat = t.reshape(-1, 1)
        # Evaluate the model
        u_flat = model(x=x_flat, t=t_flat)
        # Reshape the output to match batch processing
        u = u_flat.view(batch_size, n_points)
        # Compute the likelihood term
        exponent = (y * x / (sigma**2)) - (0.5 * (x ** 2) / (sigma**2))
        likelihood = torch.exp(exponent)
        # Compute the integrand
        integrand = u * likelihood
        # Perform trapezoidal integration along the last dimension
        integrals = torch.trapz(integrand, x, dim=-1)

    return integrals

# w_func is meant to be the solution of the backward equation
def w_func(x, t, models, time_knots):
    t_scalar = t[0,0].item() if isinstance(t, torch.Tensor) else t
    if t_scalar == 0:
        return models[0](x, t)
    else:
        idx = next(i for i in range(len(models) - 1) if time_knots[i] < t_scalar <= time_knots[i+1])

        return models[idx](x, t)

# Compute the gradient of log(w)   
def grad_log_w(x, t, nets, obs_times, solve_for="logw"):
    x = x.clone().detach().requires_grad_(True)
    t = t.clone().detach().requires_grad_(False)
    t_scalar = t[0,0].item() if isinstance(t, torch.Tensor) else t
    if t_scalar == 0.0:
        val = nets[0](x, t)
    else:
        idx = next(i for i in range(len(nets) - 1) if obs_times[i] < t_scalar <= obs_times[i+1])
        val = nets[idx](x, t)
    grad = torch.autograd.grad(val, x, grad_outputs=torch.ones_like(val), create_graph=True)[0]

    if solve_for == "logw":
        return grad
    elif solve_for == "w":
        log_grad = grad / (val + 1e-9)
        log_grad = torch.clip(log_grad, -100.0, 100.0)
        return log_grad

def terminal_func(x):
    return torch.tensor(1.0)

def log_likelihood_obs(y, x, r):
    return (y * x).sum(dim=1, keepdim=True) / (r**2) - 0.5 * (x**2).sum(dim=1, keepdim=True) / (r**2)

def simulate_samples(drift_func, diff_func, T, N, x_min, x_max, X0, noise_std=0.2, n_paths=1, seed = None, clamp=False):
    # k is the drift funtion
    d = X0.shape[0]
    if seed is not None:
        torch.manual_seed(seed)
    dt = T / N
    t = torch.linspace(0, T, N + 1, device=device)
    X = torch.zeros((n_paths, N + 1, d), device=device)
    X[:, 0, :] = X0.unsqueeze(0).expand(n_paths,d)

    for i in range(N):
        ti = t[i].expand(n_paths, 1)
        xi = X[:, i, :]
        drift = drift_func(xi, ti)
        sigma = diff_func(xi).view(1, d)
        diffusion = sigma * torch.randn(n_paths, d, device=device) * torch.sqrt(torch.tensor(dt, device=device))
        X[:, i + 1, :] = xi + drift * dt + diffusion
    
    epsilon = noise_std * torch.randn_like(X)    
    Y = X + epsilon
    
    if clamp:
        X = torch.clamp(X, x_min, x_max)

    return t, X.squeeze(-1), Y.squeeze(-1)

def trace_diag_diffusion(u_x: torch.Tensor, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """
    Exact trace term for diagonal diffusion:
        tr(a Hess(u)) = sum_i sigma_i^2 * d2u/dx_i^2
    where a = diag(sigma^2).

    u_x:   (B,d) gradient of u wrt x
    x:     (B,d) requires_grad_(True)
    sigma: (B,d) diagonal diffusion entries

    returns: (B,1)
    """
    B, d = x.shape
    if u_x.shape != (B, d):
        raise ValueError(f"u_x must be (B,d)={(B,d)}; got {tuple(u_x.shape)}")
    
    if sigma.ndim == 1:
        if sigma.shape[0] != d:
            raise ValueError(f"sigma as (d,) must have d={d}; got {tuple(sigma.shape)}")
        sigma_bd = sigma.view(1, d).expand(B, d)  # (B,d)
    elif sigma.ndim == 2:
        if sigma.shape != (B, d):
            raise ValueError(f"sigma as (B,d) must be {(B,d)}; got {tuple(sigma.shape)}")
        sigma_bd = sigma
    else:
        raise ValueError(f"sigma must be (d,) or (B,d); got ndim={sigma.ndim}, shape={tuple(sigma.shape)}")
    
    trace = torch.zeros((B, 1), device=x.device, dtype=x.dtype)

    for i in range(d):
        gi = u_x[:, i:i+1]  # (B,1)

        # second derivative row (B,d), then select i-th component -> (B,1)
        d2u_dxi2 = torch.autograd.grad(
            outputs=gi,
            inputs=x,
            grad_outputs=torch.ones_like(gi),
            create_graph=True,
            retain_graph=True
        )[0][:, i:i+1]

        trace = trace + (sigma_bd[:, i:i+1] ** 2) * d2u_dxi2
        
    return trace


def mle_kappa_mm(X, dt, J, k2_fixed, alpha_ridge=1e-6):
    """
    Estimate k1 and k_minus1 with k2 fixed, using
    Delta X = dt * drift + noise
    """
    if torch.is_tensor(X):
        X = X.detach().cpu().numpy()
    if torch.is_tensor(dt):
        dt = float(dt.detach().cpu().item())

    x = X[:, :-1, :]
    x_next = X[:, 1:, :]

    dX = x_next - x

    x1 = x[..., 0]
    x2 = x[..., 1]

    a = (x1 * x2).reshape(-1)
    b = (J - x2).reshape(-1)

    y1 = dX[..., 0].reshape(-1)
    y2 = dX[..., 1].reshape(-1) - dt * k2_fixed * b

    A1 = np.stack([-dt * a, dt * b], axis=1)
    A2 = np.stack([-dt * a, dt * b], axis=1)

    A = np.concatenate([A1, A2], axis=0)
    y = np.concatenate([y1, y2], axis=0)

    model = Ridge(alpha=alpha_ridge, fit_intercept=False)
    model.fit(A, y)

    k1_hat, k_minus1_hat = model.coef_
    return float(k1_hat), float(k_minus1_hat)

def mle_kappa_4d(X, dt, k):
    """Closed-form M-step update for Ring Coupled Double Well"""
    if not torch.is_tensor(X):
        X = torch.tensor(X)
    if not torch.is_tensor(dt):
        dt = torch.tensor(dt)

    X_t = X[:, :-1, :]
    dX = X[:, 1:, :] - X[:, :-1, :]
    
    X_left = torch.roll(X_t, shifts=1, dims=-1)
    X_right = torch.roll(X_t, shifts=-1, dims=-1)
    f_X = -torch.pow(X_t, 3) + k * (X_left + X_right - 2 * X_t)
    
    numerator = torch.sum((dX - f_X * dt) * X_t, dim=(0, 1))
    denominator = torch.sum((X_t ** 2) * dt, dim=(0, 1))
    kappa_hat = numerator / denominator
    return kappa_hat.tolist()