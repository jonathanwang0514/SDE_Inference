import torch
import argparse
import torch.nn as nn
from utils import simulate_samples, mle_kappa_mm, mle_kappa_4d
from SDEs import RingCoupledDoubleWell, MichaelisMentenDrift, diff_function
from model import PINN_Net
from data_prep import PINNDataFactory
from trainer import EM_algorithm

def get_sde_config(sde_name, device):
    """
    Registry for all SDE configurations. 
    Returns a dictionary of parameters for the requested SDE.
    """
    if sde_name == "michaelis_menten":
        def mm_m_step(X_sim, dt_sim):
            k1, k_minus1, k2 = mle_kappa_mm(X_sim, dt_sim, J=3)
            return {"k1": k1, "k_minus1": k_minus1, "k2": k2}

        return {
            "d": 2,
            "x_min": 0.0,
            "x_max": 2.0,
            "T": 0.5,
            "n_steps": 500,
            "obs_times": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
            "x0": torch.tensor([1.0, 1.0], device=device),
            "sigma": 0.3,
            "r_noise": 0.2,
            "true_drift_func": MichaelisMentenDrift(1.3634, 2.4379, 0.2018, J=3),
            "prior_drift_func": MichaelisMentenDrift(1.3634, 2.4379, 0.2018, J=3),
            "m_step_func": mm_m_step,
            "posterior_grad_weight": 0.1,
            "early_stop_threshold": 20.0
        }

    elif sde_name == "4d_double_well":
        def double_well_m_step(X_sim, dt_sim):
            kappa_hat = mle_kappa_4d(X_sim, dt_sim, k=1.0)
            return {"kappa": kappa_hat}

        return {
            "d": 4,
            "x_min": 0.0,
            "x_max": 2.5,
            "T": 5.0,
            "n_steps": 500,
            "obs_times": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "x0": torch.tensor([1.0, 1.0, 1.0, 1.0], device=device),
            "sigma": 0.4,
            "r_noise": 0.2,
            "true_drift_func": RingCoupledDoubleWell([1.0, 1.0, 1.0, 1.0], k=1.0),
            "prior_drift_func": RingCoupledDoubleWell([0.5, 0.5, 0.5, 0.5], k=1.0),
            "m_step_func": double_well_m_step,
            "posterior_grad_weight": 0.2,
            "early_stop_threshold": 30.0
        }

    else:
        raise ValueError(f"SDE '{sde_name}' is not recognized. Check your spelling.")
    
def main():
    parser = argparse.ArgumentParser(description="Run SDE Inference Pipeline")
    parser.add_argument(
        "--sde", 
        type=str, 
        required=True, 
        choices=["michaelis_menten", "4d_double_well"],
        help="Which SDE experiment to run."
    )
    args = parser.parse_args()

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Running on device: {device} | Experiment: {args.sde}")

    cfg = get_sde_config(args.sde, device)
    dt = cfg["T"] / cfg["n_steps"]
    N_intervals = len(cfg["obs_times"]) - 1
    diff_func = diff_function(cfg["sigma"])

    print(f"Generating synthetic data for {args.sde}...")
    _, X_full, Y_full = simulate_samples(
        cfg["true_drift_func"], diff_func, cfg["T"], cfg["n_steps"], 
        -10.0, 10.0, X0=cfg["x0"], seed=5
    )
    
    obs_idx = [int(t / dt) for t in cfg["obs_times"]]
    y_obs = Y_full[:, obs_idx]

    print("Initializing PINNs and Data Factory...")
    models = nn.ModuleList([PINN_Net(cfg["d"], 4, 70) for _ in range(N_intervals + 1)]).to(device)

    data_factory = PINNDataFactory(
        obs_times=cfg["obs_times"], 
        x_min=cfg["x_min"], x_max=cfg["x_max"], d=cfg["d"], 
        samples_per_interval=1000, 
        batch_size=1000, 
        dtype=torch.float32, 
        device=device
    )

    print("Starting Expectation-Maximization...")
    history = EM_algorithm(
        iteration=1000, 
        models=models, 
        prior_drift_func=cfg["prior_drift_func"], 
        diff_func=diff_func, 
        data_factory=data_factory, 
        m_step_func=cfg["m_step_func"], 
        obs_times=cfg["obs_times"], 
        y=y_obs, 
        r=cfg["r_noise"], 
        T=cfg["T"], 
        n=cfg["n_steps"], 
        x_min=cfg["x_min"], 
        x_max=cfg["x_max"], 
        X0=cfg["x0"], 
        num_path=50, 
        posterior_grad_weight=cfg["posterior_grad_weight"],
        early_stop_threshold=cfg["early_stop_threshold"]
    )
    
    print(f"Training Complete for {args.sde}!")
    print("Final Parameters:", history[-1])

if __name__ == "__main__":
    main()