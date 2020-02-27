import cvxpy as cp
import numpy as np
import scipy.sparse as sparse

class MPC:
  def __init__(self, nz, nu, K, Q, R, zmin, zmax, umin, umax):
    self._nz = nz
    self._nu = nu
    self._K = K
    self._Q = Q
    self._R = R
    self._zmin = zmin
    self._zmax = zmax
    self._umin = umin
    self._umax = umax

    # Define parameters
    self.z_init = cp.Parameter(self._nz)
    self.z_goal = cp.Parameter(self._nz)

    self.A = [None] * self._K
    self.B = [None] * self._K
    self.o = [None] * self._K

    for k in range(self._K):
      self.A[k] = cp.Parameter((self._nz, self._nz))
      self.B[k] = cp.Parameter((self._nz, self._nu))

    # Define action and observation vectors (variables)
    self.u = cp.Variable((self._nu, self._K))
    self.z = cp.Variable((self._nz, self._K + 1))

    objective = 0
    constraints = [self.z[:, 0] == self.z_init]

    for k in range(self._K):
        objective += cp.quad_form(self.z[:, k+1] - self.z_goal, self._Q) + cp.quad_form(self.u[:, k], self._R)
        constraints += [self.z[:, k + 1] == self.A[k] * self.z[:, k] + self.B[k] * self.u[:, k]]
        constraints += [self._zmin <= self.z[:, k], self.z[:, k] <= self._zmax]
        constraints += [self._umin <= self.u[:, k], self.u[:, k] <= self._umax]

    self.prob = cp.Problem(cp.Minimize(objective), constraints)

  def run_mpc(self, A, B, z0, zn):
    self.z_init.value = z0
    self.z_goal.value = zn

    for k in range(self._K):
      self.A[k].value = A[k, :, :]
      self.B[k].value = B[k, :, :]

    return self.prob.solve(warm_start=True)