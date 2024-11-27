import functools

from typing import Callable, Literal, Optional

# import scipy
# import numpy as jnp
# import numpy.typing as npt
# Array = npt.ArrayLike

from flax import struct

import jax
import jax.numpy as jnp

from manim import *


jax.config.update("jax_enable_x64", True)


class HamiltonianSolver:

    def __init__(
        self,
        kinetic: Callable[[jax.Array], float],   # T(p)
        potential: Callable[[jax.Array], float], # V(q)
        t0: float,
        q0: jax.Array,
        p0: jax.Array,
        dim: int,
        step_size: float
    ):
        self.q_old = jnp.full((dim,), jnp.nan)
        self.p_old = jnp.full((dim,), jnp.nan)
        self.t_old = jnp.nan
        self._kinetic = kinetic
        self._potential = potential
        try:
            self._kinetic = jax.jit(kinetic)
        except Exception as e:
            print(f"kinetic hamiltonian is failed to jitted: {e}")
        try:
            self._potential = jax.jit(potential)
        except Exception as e:
            print(f"potential hamiltonian is failed to jitted: {e}")
        self._velocity = jax.grad(kinetic)
        potential_gradient = jax.grad(potential)
        self._force = jax.jit(lambda q: -potential_gradient(q)) 
        
        self.t = t0
        self.q = q0
        self.p = p0
        self.dim = dim
        self.step_size = step_size
        
        self._step_impl = NotImplemented
        self._solve_impl = NotImplemented
    
    @property
    def energy(self):
        return self._kinetic(self.p) + self._potential(self.q)
    
    def step(self):
        q_new, p_new = self._step_impl(self.q, self.p)
        self.q_old = self.q
        self.p_old = self.p
        self.t_old = self.t
        self.q = q_new
        self.p = p_new 
        self.t = self.t_old + self.step_size
        return self.q, self.p, self.t
    
    def solve(self, n_steps):
        q0 = self.q
        p0 = self.p
        t0 = self.t
        qs, ps = self._solve_impl(q0, p0, n_steps)
        ts = t0 + self.step_size * jnp.arange(1, n_steps + 1, 1.)
        if n_steps < 2:
            self.q_old = self.q
            self.p_old = self.p
            self.t_old = self.t
        else:
            self.q_old = qs[-2]
            self.p_old = ps[-2]
            self.t_old = ts[-2]
        self.q = qs[-1]
        self.p = ps[-1]
        self.t = ts[-1]
        return qs, ps, ts

def eular_step(q, p, h, velocity, force):
    dq = velocity(p) * h
    dp = force(q) * h
    return q + dq, p + dp

def eular_solve(q0, p0, n_steps, h, velocity, force):
    def step(carry, _):
        q, p = carry
        q_new, p_new = eular_step(q, p, h, velocity, force)
        return (q_new, p_new), (q_new, p_new)
    
    initial_carry = (q0, p0)
    ys, _ = jax.lax.scan(step, initial_carry, None, length=n_steps)
    qs, ps = ys[:, 0], ys[:, 1]
    return qs, ps

class Eular(HamiltonianSolver):
    
    def __init__(
        self,
        kinetic: Callable[[jax.Array], float],   # T(p)
        potential: Callable[[jax.Array], float], # V(q)
        t0: float,
        q0: jax.Array,
        p0: jax.Array,
        dim: int,
        step_size: float
    ):
        super().__init__(kinetic, potential, t0, q0, p0, dim, step_size)
        
        self._step_impl = jax.jit(functools.partial(eular_step,
            h=self.step_size, velocity=self._velocity, force=self._force),
            static_argnames=("h", "velocity", "force")
        )
        self._solve_impl = jax.jit(functools.partial(eular_solve, 
            h=self.step_size, velocity=self._velocity, force=self._force),
            static_argnames=("h", "velocity", "force")
        )
    
def rk_step(q, p, h, velocity, force, A, B, n_stages, dim):
    def loop_body(carry, a):
        Kq, Kp, s = carry
        dq = jnp.dot(Kq[:s].T, a[:s]) * h
        dp = jnp.dot(Kp[:s].T, a[:s]) * h
        
        Kq_new = velocity(p + dp)
        Kp_new = force(q + dq)
        
        Kq = Kq.at[s].set(Kq_new)
        Kp = Kp.at[s].set(Kp_new)
        
        return (Kq, Kp, s), None
    Kq = jnp.empty((n_stages, dim))
    Kp = jnp.empty((n_stages, dim))
    Kq = Kq.at[0].set(velocity(p))
    Kp = Kp.at[0].set(force(q))
    carry, _ = jax.lax.scan(loop_body, (Kq, Kp, 1), A[1:])
    Kq, Kp = carry
    q_new = q + h * jnp.dot(Kq[-1].T, B)
    p_new = p + h * jnp.dot(Kp[-1].T, B)
    return q_new, p_new

def rk_solve(q0, p0, n_steps, h, velocity, force, A, B, n_stages, dim):
    def step(carry, _):
        q, p = carry
        q_new, p_new = rk_step(q, p, h, velocity, force, A, B, n_stages, dim)
        return (q_new, p_new), (q_new, p_new)
    
    initial_carry = (q0, p0)
    ys, _ = jax.lax.scan(step, initial_carry, None, length=n_steps)
    qs, ps = ys[:, 0], ys[:, 1]
    return qs, ps

class RungeKutta(HamiltonianSolver):
    A: jax.Array = NotImplemented
    B: jax.Array = NotImplemented
    order: int = NotImplemented
    n_stages: int = NotImplemented
    
    def __init__(
        self,
        kinetic: Callable[[jax.Array], float],   # T(p)
        potential: Callable[[jax.Array], float], # V(q)
        t0: float,
        q0: jax.Array,
        p0: jax.Array,
        dim: int,
        step_size: float
    ):
        super().__init__(kinetic, potential, t0, q0, p0, dim, step_size)
        
        self._step_impl = jax.jit(
            functools.partial(rk_step, 
            h=self.step_size, velocity=self._velocity, force=self._force,
            A=self.A, B=self.B, n_stages=self.n_stages, dim=self.dim),
            static_argnames=("h", "velocity", "force", "A", "B", "n_stages", "dim")
        )
        self._solve_impl = jax.jit(
            functools.partial(rk_solve, 
            h=self.step_size, velocity=self._velocity, force=self._force,
            A=self.A, B=self.B, n_stages=self.n_stages, dim=self.dim),
            static_argnames=("h", "velocity", "force", "A", "B", "n_stages", "dim")
        )
        
class RK23(RungeKutta):
    A = jnp.array([
        [0, 0, 0],
        [1/2, 0, 0],
        [0, 3/4, 0]
    ])
    B = jnp.array([2/9, 1/3, 4/9])
    order = 3
    n_stages = 3
    
class RK45(RungeKutta):
    A = jnp.array([
        [0, 0, 0, 0, 0],
        [1/5, 0, 0, 0, 0],
        [3/40, 9/40, 0, 0, 0],
        [44/45, -56/15, 32/9, 0, 0],
        [19372/6561, -25360/2187, 64448/6561, -212/729, 0],
        [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656]
    ])
    B = jnp.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84])
    order = 5
    n_stages = 6
    
def symplectic_step(q, p, h, velocity, force, C, D):
    def loop_body(carry, inputs):
        q, p = carry
        c, d = inputs
        dp = d * h * force(q)
        p_new = p + dp
        dq = c * h * velocity(p_new)
        q_new = q + dq
        return (q_new, p_new), None
    carry, _ = jax.lax.scan(loop_body, (q, p), (C, D))
    q_new, p_new = carry
    return q_new, p_new

def symplectic_solve(q0, p0, n_steps, h, velocity, force, C, D):
    def step(carry, _):
        q, p = carry
        q_new, p_new = symplectic_step(q, p, h, velocity, force, C, D)
        return (q_new, p_new), (q_new, p_new)
    
    initial_carry = (q0, p0)
    ys, _ = jax.lax.scan(step, initial_carry, None, length=n_steps)
    qs, ps = ys[:, 0], ys[:, 1]
    return qs, ps


class Symplectic(HamiltonianSolver):
    C: jax.Array = NotImplemented
    D: jax.Array = NotImplemented
    order: int = NotImplemented
    
    def __init__(
        self,
        kinetic: Callable[[jax.Array], float],   # T(p)
        potential: Callable[[jax.Array], float], # V(q)
        t0: float,
        q0: jax.Array,
        p0: jax.Array,
        dim: int,
        step_size: float
    ):
        super().__init__(kinetic, potential, t0, q0, p0, dim, step_size)
        
        self._step_impl = jax.jit(
            functools.partial(symplectic_step, 
            h=self.step_size, velocity=self._velocity, force=self._force,
            C=self.C, D=self.D),
            # static_argnames=("h", "velocity", "force", "C", "D")
        )
        self._solve_impl = jax.jit(
            functools.partial(symplectic_solve, 
            h=self.step_size, velocity=self._velocity, force=self._force,
            C=self.C, D=self.D),
            # static_argnames=("h", "velocity", "force", "C", "D")
        )
        
class Leapfrog(Symplectic):
    C = jnp.array([1/2, 0])
    D = jnp.array([1/2, 1/2])
    order = 2
    
class ForestRuth(Symplectic):
    C = jnp.array([
        1/(2 - 2**(1/3)),
        -(2**(1/3))/(2 - 2**(1/3)),
        1/(2 - 2**(1/3)),
        0
    ], dtype=jnp.float64)
    D = jnp.array([
        1/(2 * (2 - 2**(1/3))), 
        (1 - 2**(1/3))/(2 * (2 - 2**(1/3))), 
        (1 - 2**(1/3))/(2 * (2 - 2**(1/3))),
        1/(2 * (2 - 2**(1/3))),
    ], dtype=jnp.float64)
    order = 4
    
# simulation and visualization for saperable Hamiltonian H(q, p)=T(p)+V(q)
class HamiltonianSystem(VGroup):
    def __init__(
        self,
        n_coordinates,
        hamiltonian: Optional[Callable[[jax.Array, jax.Array, float], float]] = None,
        q_derivative: Optional[Callable[[jax.Array, jax.Array, float], jax.Array]] = None,
        p_derivative: Optional[Callable[[jax.Array, jax.Array, float], jax.Array]] = None,
        reference_time: float = 1,
        reference_length: float = 1,
        simulation_step_size: float = 1e-3,
        initial_t: float = 0,
        initialvalues: Optional[tuple[jax.Array, jax.Array]] = None,
        method: Literal["eular", "RK23", "RK45", "leapfrog", "forest-ruth"] = "eular",
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # System settings
        self.n_coordinates = n_coordinates
        self.hamiltonian = hamiltonian
        self.q_derivative = q_derivative
        self.p_derivative = p_derivative
        
        # Simulation settings
        self.simulation_step_size = simulation_step_size
        self.integrator = NotImplemented
        
        # visualization settings
        self.reference_time = reference_time
        self.reference_length = reference_length
        
        # simulation initializations
        self.hamiltonian = hamiltonian or (lambda q, p, t: 0)
        self.q_derivative = q_derivative or (lambda q, p, t: jnp.zeros(n_coordinates))
        self.p_derivative = p_derivative or (lambda q, p, t: jnp.zeros(n_coordinates))
                
        # submobjects
        self.dummy = Dot().set_opacity(0) # dummy for origin
        self.add(self.dummy)
        
        self.ttracker = ValueTracker(initial_t)
        initial_qs, initial_ps = initialvalues or (jnp.zeros(n_coordinates), jnp.zeros(n_coordinates))
        self.qtrakers = [ValueTracker(q) for q in initial_qs]
        self.ptrakers = [ValueTracker(p) for p in initial_ps]
        self.htracker = hamiltonian(initial_qs, initial_ps, initial_t)
        self.add(self.ttracker)
        self.add(self.qtrakers)
        self.add(self.ptrakers)
        
        self.labeled_objects = {}
        self.unlabeled_objects = []
        

        # adding updater
        
        
        
    def initialize_tracker_updaters(self):
        pass
    
    def set_hamiltonian(self, hamiltonian=None, q_derivative=None, p_derivative=None):
        pass
    
    def add_submobject(self, *unlabeled_mobjs, **labeled_mobjs):
        pass
    
    def set_variables(self, q, p):
        pass
    
    def get_energy(self):
        pass
    
    
    
    
    
    
