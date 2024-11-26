from typing import Callable, Literal, Optional


import scipy
import numpy as jnp
import numpy.typing as npt
Array = npt.ArrayLike

# import jax.numpy as jnp
# from jax import Array
# from jax import grad

from manim import *


class HamiltonianSolver:
    def __init__(
        self,
        kinetic: Callable[[Array], float],   # T(p)
        potential: Callable[[Array], float], # V(q)
        t0: float,
        q0: Array,
        p0: Array,
        dim: int,
        step_size: float
    ):
        self.q_old = None
        self.p_old = None
        self.t_old = None
        self.t = t0
        self._kinetic = kinetic
        self._potential = potential
        # self._velocity = grad(kinetic)
        # self._force = lambda q : -grad(potential)(q)
        self._velocity = None
        self._force = None
        self.q = q0
        self.p = p0
        self.dim = dim
        self.step_size = step_size
    
    
    @property
    def energy(self):
        return self._kinetic(self.p) + self._potential(self.q)
    
    def solve(self, n_steps):
        raise NotImplementedError

class Eular(HamiltonianSolver):
    
    def step(self):
        dq = self._velocity(self.p) * self.step_size
        dp = self._force(self.q) * self.step_size
        self.q_old = self.q
        self.p_old = self.p
        self.t_old = self.t
        
        self.q += dq
        self.p += dp 
        self.t += self.step_size
        
def rk_step(velocity, force, q, p, h, A, B, C, Kq, Kp):

    Kq[0] = velocity(p)
    Kp[0] = force(q)
    for s, (a, c) in enumerate(zip(A[1:], C[1:]), start=1):
        dq = jnp.dot(Kq[:s].T, a[:s]) * h
        dp = jnp.dot(Kp[:s].T, a[:s]) * h
        Kq[s] = velocity(p + dp)
        Kp[s] = force(q + dq)

    q_new = q + h * np.dot(Kq[:-1].T, B)
    p_new = p + h * np.dot(Kp[:-1].T, B)

    return q_new, p_new


class RungeKutta(HamiltonianSolver):
    C: Array = NotImplemented
    A: Array = NotImplemented
    B: Array = NotImplemented
    order: int = NotImplemented
    n_stages: int = NotImplemented
    
    def __init__(
        self,
        kinetic: Callable[[Array, Array, float], float],   # T(p)
        potential: Callable[[Array, Array, float], float], # V(q)
        t0: float,
        q0: Array,
        p0: Array,
        dim: int,
        step_size: float
    ):
        super().__init__(kinetic, potential, t0, q0, p0, dim, step_size)
        
        self.Kq = jnp.empty((self.n_stages + 1, self.dim), dtype=jnp.float64)
        self.Kp = jnp.empty((self.n_stages + 1, self.dim), dtype=jnp.float64)

    def step(self):
        t = self.t
        q = self.q
        p = self.p
        h = self.step_size

        t_new = t + h
        q_new, p_new = rk_step(
            self._velocity, self._force, q, p, h, self.A,
            self.B, self.C, self.Kq, self.Kp)
        
        self.q_old = q
        self.p_old = p
        self.t_old = t
        self.q = q_new
        self.p = p_new
        self.t = t_new
        
class RK23(RungeKutta):
    order = 3
    n_stages = 3
    C = jnp.array([0, 1/2, 3/4])
    A = jnp.array([
        [0, 0, 0],
        [1/2, 0, 0],
        [0, 3/4, 0]
    ])
    B = jnp.array([2/9, 1/3, 4/9])
    
class RK45(RungeKutta):
    order = 5
    n_stages = 6
    C = jnp.array([0, 1/5, 3/10, 4/5, 8/9, 1])
    A = jnp.array([
        [0, 0, 0, 0, 0],
        [1/5, 0, 0, 0, 0],
        [3/40, 9/40, 0, 0, 0],
        [44/45, -56/15, 32/9, 0, 0],
        [19372/6561, -25360/2187, 64448/6561, -212/729, 0],
        [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656]
    ])
    B = jnp.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84])
    
class Symplectic(HamiltonianSolver):
    C: Array = NotImplemented
    D: Array = NotImplemented
    order: int = NotImplemented
    
    def step(self):
        q = self.q
        p = self.p
        h = self.step_size
        
        self.q_old = q
        self.p_old = p
        self.t_old = self.t
        
        for c, d in zip(self.C, self.D):
            p += c * h * self._force(q)
            q += d * h * self._velocity(p)
            
        self.q = q
        self.p = p
        self.t += h
        
class Leapfrog(Symplectic):
    C = jnp.array([1/2, 1/2])
    D = jnp.array([1/2, 0])
    order = 2
    
class ForestRuth(Symplectic):
    C = jnp.array([
        1/(2 * (2 - 2**(1/3))), 
        (1 - 2**(1/3))/(2 * (2 - 2**(1/3))), 
        (1 - 2**(1/3))/(2 * (2 - 2**(1/3))),
        1/(2 * (2 - 2**(1/3))),
    ], dtype=jnp.float64)
    D = jnp.array([
        1/(2 - 2**(1/3)),
        -(2**(1/3))/(2 - 2**(1/3)),
        1/(2 - 2**(1/3)),
        0
    ], dtype=jnp.float64)
    order = 4
    
# simulation and visualization for saperable Hamiltonian H(q, p)=T(p)+V(q)
class HamiltonianSystem(VGroup):
    def __init__(
        self,
        n_coordinates,
        hamiltonian: Optional[Callable[[Array, Array, float], float]] = None,
        q_derivative: Optional[Callable[[Array, Array, float], Array]] = None,
        p_derivative: Optional[Callable[[Array, Array, float], Array]] = None,
        reference_time: float = 1,
        reference_length: float = 1,
        simulation_step_size: float = 1e-3,
        initial_t: float = 0,
        initialvalues: tuple[Array, Array] = None,
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
    
    
    
    
    
    
