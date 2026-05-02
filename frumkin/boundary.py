"""
Boundary geometries for Poisson-Boltzmann models.
"""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
from stern import SternModel


class Boundary(ABC):
    """
    Abstract boundary geometry for the Poisson-Boltzmann BVP.

    A Boundary holds one or two SternModel instances (one per electrode) and
    combines them with the domain geometry (semi-infinite, symmetric, or
    antisymmetric) to produce:

      - residual(ya, yb, y0, eps_diffuse): the 2-vector residual for solve_bvp.
      - left_profile(ya, y0, eps_diffuse): (x, y, yp, eps) through the
        left Stern region (x=0 at electrode, x=stern.ohp at OHP).
      - right_profile(yb, y0, eps_diffuse): (x, y, yp, eps) through the
        right Stern region with the same x convention. For semi-infinite
        systems this returns NaN-filled arrays.

    Parameters
    ----------
    stern : SternModel
        Stern model on the left electrode (and the right one too, by default).
    stern_right : SternModel, optional
        Stern model on the right electrode. Defaults to `stern`. Ignored for
        semi-infinite geometries.
    """

    def __init__(
        self,
        stern: SternModel,
        stern_right: Optional[SternModel] = None,
    ):
        self.stern = stern
        self.stern_right = stern_right if stern_right is not None else stern

    @property
    def ohp(self) -> float:
        """Convenience: thickness of the left Stern layer."""
        return self.stern.ohp

    @abstractmethod
    def residual(self, ya, yb, y0: float, eps_diffuse: float) -> np.ndarray:
        """Boundary condition residual vector passed to solve_bvp."""

    @abstractmethod
    def left_profile(self, ya, y0: float, eps_diffuse: float):
        """(x, y, eps) through the left Stern region."""

    @abstractmethod
    def right_profile(self, yb, y0: float, eps_diffuse: float):
        """(x, y, eps) through the right Stern region; NaNs if absent."""


class SemiInfinite(Boundary):
    """
    Electrode at x=0 with a Stern layer; bulk (phi=0) at x->infinity.
    """

    def residual(self, ya, yb, y0, eps_diffuse):
        drop = self.stern.drop(ya[1], eps_diffuse)
        return np.array(
            [
                ya[0] - y0 - drop,  # phi_OHP = y0 - drop
                yb[0],  # phi(infinity) = 0
            ]
        )

    def left_profile(self, ya, y0, eps_diffuse):
        return self.stern.profile(ya[1], y0, eps_diffuse)

    def right_profile(self, yb, y0, eps_diffuse):
        # No right Stern layer in a semi-infinite system. Return
        # expected bulk values
        x, y, eps = self.stern.profile(0.0, 0.0, eps_diffuse)
        return x, np.zeros(y.shape), np.full(eps.shape, eps)


class Symmetric(Boundary):
    """
    Two equally and identically charged electrodes at x=0 and x=L.
    Both sides see the same applied potential y0, with mirror-image Stern drops.
    """

    def residual(self, ya, yb, y0, eps_diffuse):
        drop_a = self.stern.drop(ya[1], eps_diffuse)
        drop_b = self.stern_right.drop(yb[1], eps_diffuse)
        return np.array(
            [
                ya[0] - y0 - drop_a,
                yb[0]
                - y0
                + drop_b,  # field points the other way at the right electrode
            ]
        )

    def left_profile(self, ya, y0, eps_diffuse):
        return self.stern.profile(ya[1], y0, eps_diffuse)

    def right_profile(self, yb, y0, eps_diffuse):
        # x runs from the right electrode (x=0) inward to the right OHP (x=ohp).
        # The diffuse-side field used by the Stern model along this inward axis
        # is -yb[1] (sign flips relative to the global x-axis).
        return self.stern_right.profile(-yb[1], y0, eps_diffuse)


class Antisymmetric(Boundary):
    """
    Two electrodes with equal but opposite applied potentials (+y0 and -y0).
    """

    def residual(self, ya, yb, y0, eps_diffuse):
        drop_a = self.stern.drop(ya[1], eps_diffuse)
        drop_b = self.stern_right.drop(yb[1], eps_diffuse)
        return np.array(
            [
                ya[0] - y0 - drop_a,
                yb[0] + y0 + drop_b,
            ]
        )

    def left_profile(self, ya, y0, eps_diffuse):
        return self.stern.profile(ya[1], y0, eps_diffuse)

    def right_profile(self, yb, y0, eps_diffuse):
        # Right electrode is at -y0; same axis-flip on the field as in Symmetric.
        return self.stern_right.profile(-yb[1], -y0, eps_diffuse)
