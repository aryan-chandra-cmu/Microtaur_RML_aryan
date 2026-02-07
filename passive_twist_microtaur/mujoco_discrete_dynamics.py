# mujoco_discrete_dynamics.py
"""
Drop-in discrete-time dynamics wrapper for a MuJoCo robot.xml.

What you get:
  - A discrete-time dynamics model:   x_{k+1} = f_d(x_k, u_k)
  - x = [qpos, qvel]  (size nq + nv)
  - u = data.ctrl     (size nu)
    NOTE: In your XML, actuators are <position ...>, so u is *joint position targets*.

Works well for gait rollouts (walking/trotting) and shooting-style trajectory optimization (CEM/MPPI).

Usage:
  python mujoco_discrete_dynamics.py --xml path/to/robot.xml --substeps 10

Import usage:
  from mujoco_discrete_dynamics import MujocoDiscreteDynamics
  dyn = MujocoDiscreteDynamics("robot.xml", substeps=10)
  x1 = dyn.step(x0, u0)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np

try:
    import mujoco as mj
except ImportError as e:
    raise ImportError(
        "Could not import mujoco. Install with e.g. `pip install mujoco` "
        "and ensure MuJoCo is set up on your system."
    ) from e


Array = np.ndarray


@dataclass
class StepInfo:
    """Extra info you may want to log for debugging."""
    base_dt: float
    macro_dt: float
    substeps: int
    ncon: int  # number of contacts at end of step
    ctrl: Array  # last applied ctrl
    qpos: Array
    qvel: Array


class MujocoDiscreteDynamics:
    """
    Discrete-time dynamics model wrapper.

    State:
      x = [qpos, qvel]

    Action:
      u = data.ctrl  (actuator command)
        - In your model this is a position target because actuators are <position ...>.

    Discrete-time transition:
      x_next = step(x, u)  -> executes `substeps` calls to mj_step with constant u
    """

    def __init__(
        self,
        xml_path: str,
        substeps: int = 10,
        use_fresh_data_each_call: bool = False,
    ):
        """
        Args:
          xml_path: path to robot.xml
          substeps: number of MuJoCo integrator steps per discrete-time step (macro-step)
          use_fresh_data_each_call: if True, creates a new MjData for each step call
                                   (slower, but maximally side-effect free).
        """
        self.model = mj.MjModel.from_xml_path(xml_path)
        self.substeps = int(substeps)
        if self.substeps < 1:
            raise ValueError("substeps must be >= 1")

        self.nq = int(self.model.nq)
        self.nv = int(self.model.nv)
        self.nu = int(self.model.nu)

        self.base_dt = float(self.model.opt.timestep)
        self.macro_dt = self.base_dt * self.substeps

        self._fresh = bool(use_fresh_data_each_call)
        self.data = mj.MjData(self.model) if not self._fresh else None

    # ---------------------------
    # State helpers
    # ---------------------------
    def pack_state(self, data: "mj.MjData") -> Array:
        return np.concatenate([data.qpos.copy(), data.qvel.copy()], axis=0)

    def unpack_state(self, data: "mj.MjData", x: Array) -> None:
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.shape[0] != self.nq + self.nv:
            raise ValueError(f"x must have shape ({self.nq + self.nv},) but got {x.shape}")
        data.qpos[:] = x[: self.nq]
        data.qvel[:] = x[self.nq : self.nq + self.nv]
        # Important when you manually set state: recompute kinematics/forces caches.
        mj.mj_forward(self.model, data)

    def _get_data(self) -> "mj.MjData":
        return mj.MjData(self.model) if self._fresh else self.data  # type: ignore

    # ---------------------------
    # Discrete-time transition
    # ---------------------------
    def step(self, x: Array, u: Array, return_info: bool = False) -> Array | Tuple[Array, StepInfo]:
        """
        Compute x_next = f_d(x, u)

        Args:
          x: (nq+nv,) state
          u: (nu,) actuator command (position targets in your XML)
          return_info: if True, returns (x_next, info)

        Returns:
          x_next or (x_next, info)
        """
        data = self._get_data()
        self.unpack_state(data, x)

        u = np.asarray(u, dtype=float).reshape(-1)
        if u.shape[0] != self.nu:
            raise ValueError(f"u must have shape ({self.nu},) but got {u.shape}")

        for _ in range(self.substeps):
            data.ctrl[:] = u
            mj.mj_step(self.model, data)

        x_next = self.pack_state(data)

        if not return_info:
            return x_next

        info = StepInfo(
            base_dt=self.base_dt,
            macro_dt=self.macro_dt,
            substeps=self.substeps,
            ncon=int(data.ncon),
            ctrl=u.copy(),
            qpos=data.qpos.copy(),
            qvel=data.qvel.copy(),
        )
        return x_next, info

    def rollout(self, x0: Array, U: Array) -> Array:
        """
        Roll out a control sequence.

        Args:
          x0: (nq+nv,)
          U: (T, nu)

        Returns:
          X: (T+1, nq+nv) where X[0]=x0
        """
        U = np.asarray(U, dtype=float)
        if U.ndim != 2 or U.shape[1] != self.nu:
            raise ValueError(f"U must be shape (T, {self.nu}) but got {U.shape}")

        T = U.shape[0]
        X = np.zeros((T + 1, self.nq + self.nv), dtype=float)
        X[0] = np.asarray(x0, dtype=float).reshape(-1)

        x = X[0]
        for t in range(T):
            x = self.step(x, U[t])
            X[t + 1] = x
        return X

    # ---------------------------
    # Finite-difference linearization (optional)
    # ---------------------------
    def linearize_fd(
        self,
        x: Array,
        u: Array,
        eps_x: float = 1e-6,
        eps_u: float = 1e-6,
    ) -> Tuple[Array, Array, Array]:
        """
        Finite-difference linearization around (x,u):

          x_next = f_d(x,u)
          A = d f_d / d x
          B = d f_d / d u

        NOTE: With contacts + closed-chain constraints, derivatives can be noisy.
              For first gait trajopt, shooting methods often work better than iLQR/DDP.

        Returns:
          x_next, A, B
        """
        x = np.asarray(x, dtype=float).reshape(-1)
        u = np.asarray(u, dtype=float).reshape(-1)
        if x.shape[0] != self.nq + self.nv:
            raise ValueError("Bad x shape")
        if u.shape[0] != self.nu:
            raise ValueError("Bad u shape")

        x0 = x
        u0 = u

        x_next0 = self.step(x0, u0)

        nx = self.nq + self.nv
        nu = self.nu

        A = np.zeros((nx, nx), dtype=float)
        B = np.zeros((nx, nu), dtype=float)

        # d/dx
        for i in range(nx):
            dx = np.zeros(nx, dtype=float)
            dx[i] = eps_x
            xp = self.step(x0 + dx, u0)
            xm = self.step(x0 - dx, u0)
            A[:, i] = (xp - xm) / (2.0 * eps_x)

        # d/du
        for j in range(nu):
            du = np.zeros(nu, dtype=float)
            du[j] = eps_u
            up = self.step(x0, u0 + du)
            um = self.step(x0, u0 - du)
            B[:, j] = (up - um) / (2.0 * eps_u)

        return x_next0, A, B


# ---------------------------
# CLI demo
# ---------------------------
def _main():
    p = argparse.ArgumentParser()
    p.add_argument("--xml", required=True, help="Path to MuJoCo robot.xml")
    p.add_argument("--substeps", type=int, default=10, help="MuJoCo substeps per macro step")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=5, help="How many discrete steps to run in demo")
    args = p.parse_args()

    np.random.seed(args.seed)
    dyn = MujocoDiscreteDynamics(args.xml, substeps=args.substeps)

    print("Loaded model:")
    print(f"  nq={dyn.nq}, nv={dyn.nv}, nu={dyn.nu}")
    print(f"  base_dt={dyn.base_dt:.6f} s, macro_dt={dyn.macro_dt:.6f} s (substeps={dyn.substeps})")

    # Initialize x from MuJoCo defaults
    data0 = mj.MjData(dyn.model)
    mj.mj_resetData(dyn.model, data0)
    x = dyn.pack_state(data0)

    # Simple random controls demo (note: these are *position targets* for <position> actuators)
    # If you know joint ranges, you should clip targets accordingly.
    for k in range(args.steps):
        u = 0.2 * (2.0 * np.random.rand(dyn.nu) - 1.0)  # small random targets around 0
        x, info = dyn.step(x, u, return_info=True)
        print(f"[k={k}] ncon={info.ncon} | qpos[0:3]={info.qpos[:3]} | qvel[0:3]={info.qvel[:3]}")

    print("Demo done.")


if __name__ == "__main__":
    _main()
