"""Two-electron systems and Hamiltonian construction."""

from __future__ import annotations

import numpy as np
from scipy.sparse import kron, diags, eye
from scipy.sparse.linalg import eigsh

from .Core import Grid, Result


class TwoElectronSystem:
    """Two-electron quantum system on a 1D grid."""

    def __init__(
        self,
        grid: Grid,
        external_potential,
        interaction_type: str = "coulomb",
        interaction_strength: float = 1.0,
        softening: float = 0.1,
    ) -> None:
        self.grid = grid
        self.external_potential = external_potential
        self.interaction_type = interaction_type
        self.interaction_strength = interaction_strength
        self.softening = softening

        if callable(external_potential):
            self.V_ext = external_potential(grid.x)
        else:
            self.V_ext = np.asarray(external_potential)

        # Ensure potential callable for compatibility with Result.plot
        if callable(external_potential):
            self.potential = external_potential
        else:
            self.potential = lambda x: np.interp(x, grid.x, self.V_ext)

        self._build_interaction_matrix()

    def _build_interaction_matrix(self) -> None:
        x = self.grid.x
        dx = self.grid.dx
        n = len(x)

        if self.interaction_type == "coulomb":
            X1, X2 = np.meshgrid(x, x, indexing="ij")
            self.V_int = self.interaction_strength / np.sqrt(
                (X1 - X2) ** 2 + self.softening**2
            )
        elif self.interaction_type == "contact":
            self.V_int = np.zeros((n, n))
            np.fill_diagonal(self.V_int, self.interaction_strength / dx)
        elif self.interaction_type == "soft_coulomb":
            X1, X2 = np.meshgrid(x, x, indexing="ij")
            r12 = np.abs(X1 - X2)
            self.V_int = self.interaction_strength / (r12 + self.softening)
        else:
            raise ValueError(f"Unknown interaction type '{self.interaction_type}'")

    def build_hamiltonian_matrix(self):
        n = len(self.grid.x)
        dx = self.grid.dx

        T_1d = -0.5 * (
            -2 * np.eye(n) + np.eye(n, k=1) + np.eye(n, k=-1)
        ) / dx**2

        I = eye(n, format="csr")

        T1 = kron(T_1d, I)
        T2 = kron(I, T_1d)
        V1 = kron(diags(self.V_ext), I)
        V2 = kron(I, diags(self.V_ext))
        V_int = diags(self.V_int.flatten())

        return T1 + T2 + V1 + V2 + V_int

    def solve_ground_state(self, n_states: int = 1, use_sparse: bool = True) -> Result:
        """Solve for the lowest energy states."""
        H = self.build_hamiltonian_matrix()

        if use_sparse and hasattr(self, "solve_davidson"):
            davidson_res = self.solve_davidson(H, n_states=n_states)
            energies = davidson_res.energies
            wavefunctions = davidson_res.wavefunctions
            info = getattr(davidson_res, "info", {})
        else:
            energies, wavefunctions = eigsh(H, k=n_states, which="SA")
            info = {"method": "eigsh", "converged": True}

        return Result(energies=energies, wavefunctions=wavefunctions, system=self, info=info)

    def compute_density(self, psi: np.ndarray) -> np.ndarray:
        n = len(self.grid.x)
        psi_2d = psi.reshape(n, n)
        density = 2 * np.sum(np.abs(psi_2d) ** 2, axis=1) * self.grid.dx
        return density

    def compute_pair_density(self, psi: np.ndarray) -> np.ndarray:
        n = len(self.grid.x)
        return np.abs(psi.reshape(n, n)) ** 2
