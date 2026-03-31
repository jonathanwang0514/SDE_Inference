import torch
from itertools import count
from loss import all_at_once_loss
from utils import simulate_samples, grad_log_w

def PINN_train(nets, drift_func, diff_func, data_factory, y, r, early_stop_threshold=20.0, not_in_EM=True, lr=1e-4):
    """
    Universal training loop for PINNs using all-at-once loss.
    """
    optimizer = torch.optim.AdamW(nets.parameters(), lr=lr)
    consecutive_below = 0

    for epoch in count(0):
        for net in nets:
            net.train()

        optimizer.zero_grad(set_to_none=True)
        interval_loaders, knot_loaders = data_factory()
        
        # y and r are explicitly passed here
        loss = all_at_once_loss(interval_loaders, knot_loaders, nets, drift_func, diff_func, y=y, r=r, solve_for="logw")
        loss.backward()
        optimizer.step()

        loss_val = float(loss.detach().cpu())
        
        if not_in_EM:
            print(f"epoch {epoch}: loss = {loss_val:.4f}")

        if loss_val < early_stop_threshold:
            consecutive_below += 1
        else:
            consecutive_below = 0   

        if consecutive_below > 3:
            break   

def EM_algorithm(
    iteration, models, prior_drift_func, diff_func, data_factory, 
    m_step_func, obs_times, y, r, T, n, x_min, x_max, X0, num_path, 
    posterior_grad_weight=1.0, early_stop_threshold=20.0
):
    """
    Universal EM Algorithm loop.
    
    :param m_step_func: A callable that takes (X, dt) and returns a dictionary 
                        of updated parameters to be unpacked into drift_func.update()
    :param posterior_grad_weight: The scalar multiplier for the diff_func(x)**2 * grad_log_w term
    """
    param_history = []

    for iter in range(iteration):
        # --- E-STEP: Train PINNs ---
        PINN_train(
            models, prior_drift_func, diff_func, data_factory, 
            y=y, r=r, early_stop_threshold=early_stop_threshold, not_in_EM=False
        )
        
        # --- SIMULATE POSTERIOR PATHS ---
        # The lambda captures the current state of prior_drift_func automatically
        posterior_drift_func = lambda x, t: prior_drift_func(x, t) + posterior_grad_weight * (diff_func(x)**2) * grad_log_w(x, t, models, obs_times)
        
        t_grid, X, _ = simulate_samples(
            posterior_drift_func, diff_func, T, n, 
            x_min, x_max, X0=X0, n_paths=num_path, clamp=True
        )
        dt = t_grid[1] - t_grid[0]
        dt, X = dt.to("cpu"), X.to("cpu")
        
        # --- M-STEP: Update Parameters ---
        # m_step_func handles the specific math and returns a dictionary
        new_params = m_step_func(X, dt)
        
        # Universal update step
        prior_drift_func.update(**new_params)
        param_history.append(new_params)

        # Format output string dynamically based on dictionary keys
        param_str = ", ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in new_params.items()])
        print(f"iter {iter}: {param_str}")
    
    return param_history