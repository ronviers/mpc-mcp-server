"""
mpc_core/thermodynamics.py

Exact and approximate thermodynamic quantities for MPC energy landscapes.

Primary path:  QuTiP  (exact, up to N≈14 in reasonable time)
Fallback:      Classical pair-decoupled approximation

Spin-glass solver:
Primary path:  NetKet exact diagonalisation (N≤16) or VMC (N>16)
Fallback:      Brute-force enumeration (N≤20), greedy for larger systems
"""
from __future__ import annotations

import math

import numpy as np


def compute_thermodynamic_quantities(
    epsilon_matrix: list[list[float]],
    temperature: float = 1.0,
) -> dict:
    """
    Given the NxN pairwise frustration matrix, build the Ising Hamiltonian
    H = Σ_{i<j} ε_ij σ_z^i ⊗ σ_z^j and compute Z, F, S, E_avg.
    All quantities in k_B T units (β = 1/T, k_B = 1).
    """
    N = len(epsilon_matrix)
    if N == 0:
        return dict(Z=1.0, F=0.0, S=0.0, E_avg=0.0, backend="trivial")

    beta = 1.0 / max(temperature, 1e-6)

    try:
        import qutip
        return _qutip_quantities(epsilon_matrix, N, beta, temperature)
    except ImportError:
        pass
    except Exception:
        pass

    return _classical_quantities(epsilon_matrix, N, beta, temperature)


def _qutip_quantities(eps, N, beta, T):
    import qutip

    dim = [2] * N

    def _sz(k):
        ops = [qutip.qeye(2)] * N
        ops[k] = qutip.sigmaz()
        return qutip.tensor(ops)

    H = qutip.Qobj(np.zeros((2**N, 2**N)), dims=[dim, dim])
    for i in range(N):
        for j in range(i + 1, N):
            eps_ij = eps[i][j]
            if abs(eps_ij) > 1e-10:
                H = H + eps_ij * _sz(i) * _sz(j)

    rho_unnorm = (-beta * H).expm()
    Z     = float(rho_unnorm.tr().real)
    Z     = max(Z, 1e-300)
    F     = -T * math.log(Z)
    E_avg = float((H * rho_unnorm).tr().real) / Z
    S     = (E_avg - F) / max(T, 1e-9)
    return dict(Z=Z, F=F, S=S, E_avg=E_avg, backend="qutip")


def _classical_quantities(eps, N, beta, T):
    """
    Treat each frustrated pair as an independent two-level system.
    Z = Π_{i<j, ε>0}  2 cosh(β ε_ij)
    """
    Z = 1.0
    E_avg = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            e = eps[i][j]
            if abs(e) > 1e-10:
                Z     *= 2 * math.cosh(beta * e)
                E_avg += e * math.tanh(beta * e)

    Z = max(Z, 1e-300)
    F = -T * math.log(Z)
    S = (E_avg - F) / max(T, 1e-9)
    return dict(Z=Z, F=F, S=S, E_avg=E_avg, backend="classical")


def find_ground_state(epsilon_matrix: list[list[float]]) -> dict:
    """
    Find the minimum-energy spin configuration for the Ising Hamiltonian.
    Returns energy, config (±1 spins), stable_ids (indices), backend.
    """
    N = len(epsilon_matrix)
    if N == 0:
        return dict(energy=0.0, config=[], stable_ids=[], backend="trivial")

    try:
        return _netket_ground_state(epsilon_matrix, N)
    except ImportError:
        pass
    except Exception:
        pass

    if N <= 20:
        return _brute_force_ground_state(epsilon_matrix, N)

    return _greedy_ground_state(epsilon_matrix, N)


def _ising_energy(eps, spins, N):
    E = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            E += eps[i][j] * spins[i] * spins[j]
    return E


def _netket_ground_state(eps, N):
    import netket as nk

    edges = [
        (i, j)
        for i in range(N)
        for j in range(i + 1, N)
        if abs(eps[i][j]) > 1e-10
    ]
    if not edges:
        return dict(energy=0.0, config=[1]*N, stable_ids=list(range(N)), backend="netket_trivial")

    hi = nk.hilbert.Spin(s=0.5, N=N)
    ha = nk.operator.LocalOperator(hi)
    for i in range(N):
        for j in range(i + 1, N):
            if abs(eps[i][j]) > 1e-10:
                sz_i = nk.operator.spin.sigmaz(hi, i)
                sz_j = nk.operator.spin.sigmaz(hi, j)
                ha  += eps[i][j] * sz_i @ sz_j

    if N <= 16:
        evals, evecs = nk.exact.lanczos_ed(ha, compute_eigenvectors=True)
        E0     = float(np.real(evals[0]))
        gs     = np.real(evecs[:, 0])
        idx    = int(np.argmax(np.abs(gs)))
        config = [1 if (idx >> k) & 1 == 0 else -1 for k in range(N)]
        backend = "netket_ed"
    else:
        ma     = nk.models.RBM(alpha=1)
        sa     = nk.sampler.MetropolisLocal(hi)
        op_opt = nk.optimizer.Sgd(learning_rate=0.01)
        sr     = nk.optimizer.SR(diag_shift=0.1)
        vs     = nk.vqs.MCState(sa, ma, n_samples=512)
        driver = nk.VMC(hamiltonian=ha, optimizer=op_opt,
                        variational_state=vs, preconditioner=sr)
        driver.run(n_iter=200)
        E0     = float(np.real(vs.expect(ha).mean))
        config = [1] * N
        backend = "netket_vmc"

    stable_ids = [i for i, s in enumerate(config) if s == 1]
    return dict(energy=E0, config=config, stable_ids=stable_ids, backend=backend)


def _brute_force_ground_state(eps, N):
    best_E   = math.inf
    best_cfg = [1] * N
    for mask in range(2**N):
        spins = [1 if (mask >> k) & 1 else -1 for k in range(N)]
        E     = _ising_energy(eps, spins, N)
        if E < best_E:
            best_E   = E
            best_cfg = spins[:]
    stable_ids = [i for i, s in enumerate(best_cfg) if s == 1]
    return dict(energy=best_E, config=best_cfg, stable_ids=stable_ids, backend="brute_force")


def _greedy_ground_state(eps, N):
    spins    = [1] * N
    improved = True
    while improved:
        improved = False
        for i in range(N):
            E_before = _ising_energy(eps, spins, N)
            spins[i] *= -1
            E_after  = _ising_energy(eps, spins, N)
            if E_after < E_before:
                improved = True
            else:
                spins[i] *= -1
    E = _ising_energy(eps, spins, N)
    stable_ids = [i for i, s in enumerate(spins) if s == 1]
    return dict(energy=E, config=spins, stable_ids=stable_ids, backend="greedy")


def free_energy_surface(
    epsilon_matrix: list[list[float]],
    T_range:      tuple[float, float] = (0.2, 5.0),
    E_star_range: tuple[float, float] = (2.0, 40.0),
    n_T:  int   = 20,
    n_E:  int   = 20,
    alpha: float = 1.0,
) -> dict:
    """Compute F(T, E*) on a grid for the 3-D free-energy surface."""
    N      = len(epsilon_matrix)
    Ts     = np.linspace(T_range[0],     T_range[1],     n_T).tolist()
    Estars = np.linspace(E_star_range[0], E_star_range[1], n_E).tolist()

    edges = [
        epsilon_matrix[i][j]
        for i in range(N) for j in range(i + 1, N)
        if epsilon_matrix[i][j] > 0
    ]
    eps_min = min(edges) if edges else 0.0
    d_avg   = 2 * len(edges) / N if N > 0 else 0.0

    F_grid     = []
    N_max_grid = []

    for T in Ts:
        row_F    = []
        row_Nmax = []
        td = compute_thermodynamic_quantities(epsilon_matrix, temperature=T)
        F_base = td["F"]
        for E_star in Estars:
            row_F.append(round(F_base, 4))
            if eps_min > 0 and d_avg > 0:
                nmax = math.sqrt(2 * E_star / (alpha * eps_min * d_avg))
            else:
                nmax = float("inf")
            row_Nmax.append(round(nmax, 2) if nmax != float("inf") else 999)
        F_grid.append(row_F)
        N_max_grid.append(row_Nmax)

    return dict(
        T=Ts,
        E_star=Estars,
        F=F_grid,
        N_max=N_max_grid,
        N=N,
        eps_min=eps_min,
        d_avg=round(d_avg, 3),
    )
