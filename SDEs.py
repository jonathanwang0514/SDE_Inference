import torch

class MichaelisMentenDrift:
    def __init__(self, k1, k_minus1, k2, J):
        self.k1 = float(k1)
        self.k_minus1 = float(k_minus1)
        self.k2 = float(k2)
        self.J = J

    def __call__(self, x, t):
        """
        x: tensor (..., 3)
        t: unused but kept for SDE/PDE API compatibility
        """
        x1, x2 = x[..., 0], x[..., 1]

        b1 = -self.k1 * x1 * x2 + self.k_minus1 * (self.J - x2)
        b2 = -self.k1 * x1 * x2 + (self.k_minus1 + self.k2) * (self.J - x2)
        # b3 =  self.k1 * x1 * x2 - (self.k_minus1 + self.k2) * x3

        return torch.stack((b1, b2), dim=-1)

    def update(self, *, k1=None, k_minus1=None, k2=None):
        if k1 is not None:
            self.k1 = float(k1)
        if k_minus1 is not None:
            self.k_minus1 = float(k_minus1)
        if k2 is not None:
            self.k2 = float(k2)

class Lorenz4DDrift:
    def __init__(self, k1, k2, F):
        """
        F: The forcing constant (parameter kappa) [cite: 10]
        damping: The damping coefficient
        """
        self.F = float(F)
        self.k1 = float(k1)
        self.k2 = float(k2)

    def __call__(self, x, t):
        """
        x: tensor (..., 4)
        t: unused but kept for SDE/PDE API compatibility [cite: 85, 106]
        """
        x1, x2, x3, x4 = x[..., 0], x[..., 1], x[..., 2], x[..., 3]

        # Lorenz '96 Equations with cyclic indexing:
        # b_i = (x_{i+1} - x_{i-2}) * x_{i-1} - x_i + F
        
        b1 = self.k1 * (x2 - x3) * x4 - self.k2 * x1 + self.F
        b2 = self.k1 * (x3 - x4) * x1 - self.k2 * x2 + self.F
        b3 = self.k1 * (x4 - x1) * x2 - self.k2 * x3 + self.F
        b4 = self.k1 * (x1 - x2) * x3 - self.k2 * x4 + self.F

        return torch.stack((b1, b2, b3, b4), dim=-1)

    def update(self, *, k1=None, k2=None, F=None):
        if F is not None:
            self.F = float(F)
        if k1 is not None:
            self.k1 = float(k1)
        if k2 is not None:
            self.k2 = float(k2)

class RingCoupledDoubleWell:
    def __init__(self, kappa, k):
        """
        kappa: tensor (5,) - The well-depth parameters you want to infer
        k: float - Fixed coupling strength (e.g., 0.1)
        """
        self.kappa = torch.as_tensor(kappa, dtype=torch.float32)
        self.k = float(k)

    def __call__(self, x, t=None):
        """
        x: tensor (..., 5)
        """
        x1, x2, x3, x4 = x[..., 0], x[..., 1], x[..., 2], x[..., 3]

        # Local dynamics + Ring Coupling sum_j A_ij(x_j - x_i)
        # b1 uses neighbors x5 and x2
        b1 = self.kappa[0]*x1 - x1**3 + self.k*((x4 - x1) + (x2 - x1))
        
        # b2 uses neighbors x1 and x3
        b2 = self.kappa[1]*x2 - x2**3 + self.k*((x1 - x2) + (x3 - x2))
        
        # b3 uses neighbors x2 and x4
        b3 = self.kappa[2]*x3 - x3**3 + self.k*((x2 - x3) + (x4 - x3))
        
        # b4 uses neighbors x3 and x5
        b4 = self.kappa[3]*x4 - x4**3 + self.k*((x3 - x4) + (x1 - x4))
        
        # b5 uses neighbors x4 and x1
        # b5 = self.kappa[4]*x5 - x5**3 + self.k*((x4 - x5) + (x1 - x5))

        return torch.stack((b1, b2, b3, b4), dim=-1)

    def update(self, *, kappa=None):
        if kappa is not None:
            self.kappa = torch.as_tensor(kappa, dtype=torch.float32)

class diff_function:
    """
    Docstring for diff_function
    
    :param x: input
    :param sigma: a single value of diffusion 
    """
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, x):
        d = x.shape[-1]
        sigma_tensor = torch.as_tensor(self.sigma, dtype=x.dtype, device=x.device)
        Sigma = sigma_tensor.expand(d)
        return Sigma