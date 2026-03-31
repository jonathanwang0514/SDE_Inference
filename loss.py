import torch
import torch.nn as nn
import torch.optim as optim
from utils import terminal_func, log_likelihood_obs, trace_diag_diffusion

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def pde_residual(model, x, t, drift_func, diff_func, solve_for="logw", device=device):
    """
    Unified physics engine used by both the Dataset and the Loss function.
    """
    x.requires_grad_(True)
    t.requires_grad_(True)
    
    x, t = x.to(device), t.to(device)
    
    p = model(x, t)
    d = x.shape[1]

    # Gradients
    p_t = torch.autograd.grad(p, t, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    
    b = drift_func(x, t)
    sigma = diff_func(x)
    
    if d == 1:
        p_xx = torch.autograd.grad(p_x, x, grad_outputs=torch.ones_like(p_x), create_graph=True)[0]
        if solve_for == "logw":
            residual = p_t + b * p_x + 0.5 * (sigma**2) * (p_xx + p_x**2)
        else: # solve_for == "w"
            residual = p_t + b * p_x + 0.5 * (sigma**2) * p_xx
    else:
        # For multi-dimensional cases (assuming trace_diag_diffusion is defined elsewhere)
        trace_term = trace_diag_diffusion(p_x, x, sigma) 
        drift_term = (b * p_x).sum(dim=1, keepdim=True)
        if solve_for == "logw":
            quad_term = ((sigma ** 2) * (p_x ** 2)).sum(dim=1, keepdim=True)
            residual = p_t + drift_term + 0.5 * (trace_term + quad_term)
        else:
            residual = p_t + drift_term + 0.5 * trace_term
            
    return residual

def PDE_loss(model, x, t, drift_func, diff_func, solve_for="logw"):
    residual = pde_residual(model, x, t, drift_func, diff_func, solve_for)

    return (residual ** 2).mean()

def PDE_loss_log_transform(model, x, t, mu_func, sigma_func):
    x.requires_grad_(True)
    t.requires_grad_(True)
    
    u = model(x, t)  # Predict u(x,t) = log w(x,t)
    
    # Compute gradients using autograd
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    
    # Drift and diffusion
    mu_x = mu_func(x)
    sigma = sigma_func(x)
    
    # Log-transformed Kolmogorov backward PDE residual
    residual = u_t + mu_x * u_x + 0.5 * sigma**2 * (u_xx + u_x**2)
    
    return torch.mean(residual**2)

def neumann_loss(model, t_b, x_b):
    """
    Enforce ∂_x w(x_b, t) = 0 for all sampled t in t_b.
    t_b: (B,1) times; x_b: scalar 0.0 or 3.0
    """
    x = torch.full_like(t_b, float(x_b), requires_grad=True)
    t = t_b.detach().requires_grad_(True)  # allow w_t if you need it later
    w = model(x, t)
    (w_x,) = torch.autograd.grad(w, x, grad_outputs=torch.ones_like(w),
                                 create_graph=True, retain_graph=True)
    return torch.mean(w_x**2)

def dirichlet_loss(model, t_b, x_b, g=None):
    """
    Enforce w(x_b, t) = g(t). If g is None, defaults to 0 (absorbing).
    t_b: (B,1) times in [0,T]
    x_b: scalar (e.g., 0.0 or 3.0)
    g:   callable(t_b)->(B,1) or None
    """
    x = x_b.detach().clone().requires_grad_(True)
    t = t_b.detach()
    target = g(t_b) if callable(g) else torch.zeros_like(t_b)
    pred = model(x, t_b)
    return torch.mean((pred - target)**2)

def all_at_once_loss(interval_loaders, knot_loaders, nets, drift_func, diff_func, y, r, solve_for):
    total_loss = 0.0
    N = len(nets) - 1

    for i in range(N+1):
        pde_loss = 0.0
        jump_loss = 0.0
        num_batch_pde = 0
        num_batch_jump = 0

        for t, x in interval_loaders[i]:
            t, x = t.to(device), x.to(device)
            res = PDE_loss(nets[i], x, t, drift_func, diff_func, solve_for=solve_for)
            pde_loss += res
            num_batch_pde += 1

        for t, x in knot_loaders[i]:
            t, x = t.to(device), x.to(device)
            h = nets[i](x, t)

            if i < N:
                h_plus = nets[i+1](x, t)
                log_lik_y = log_likelihood_obs(y[0, i+1], x, r=r)
                jump_loss += torch.mean((h - h_plus - log_lik_y)**2)
            else: 
                terminal_value = terminal_func(x)
                jump_loss += ((h - terminal_value)**2).mean()

            num_batch_jump += 1    

        pde_loss /= num_batch_pde
        jump_loss /= num_batch_jump
        
        total_loss += pde_loss + jump_loss

    # total_loss /= N + 1
    # parts = {
    #     "pde": torch.stack(pde_terms).mean().item(),
    #     "jump": (torch.stack(jump_terms).mean().item() if jump_terms else 0.0),
    #     "total": total_loss.item()
    # }

    return total_loss