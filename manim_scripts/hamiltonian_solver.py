import functools

from typing import Callable, Optional

import jax
import jax.numpy as jnp

# jax.config.update("jax_debug_nans", True)
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
        step_size: float,
        collision_handler: Optional[Callable[[jax.Array], jax.Array]] = None
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
        
        if collision_handler:
            try:
                self._collision_handler = jax.jit(collision_handler)
            except Exception as e:
                print(f"collision handler is failed to jitted: {e}")
        else:
            self._collision_handler = None
            
        self._step_impl = NotImplemented
        self._solve_impl = NotImplemented
    
    @property
    def energy(self):
        return self._kinetic(self.p) + self._potential(self.q)
    
    def set_initial_values(
        self,
        q0: jax.Array, 
        p0: jax.Array, 
        t0: float
    ):
        self.q_old = jnp.full((self.dim,), jnp.nan)
        self.p_old = jnp.full((self.dim,), jnp.nan)
        self.t_old = jnp.nan
        self.q = q0
        self.p = p0
        self.t = t0
        
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

def _eular_step(q, p, h, velocity, force):
    dq = velocity(p) * h
    dp = force(q) * h
    return q + dq, p + dp

def _eular_step_with_collision(q, p, h, velocity, force, collision_handler):
    return collision_handler(*_eular_step(q, p, h, velocity, force))

def _eular_solve(q0, p0, n_steps, h, velocity, force):
    def step(carry, _):
        q, p = carry
        q_new, p_new = _eular_step(q, p, h, velocity, force)
        return (q_new, p_new), (q_new, p_new)
    
    initial_carry = (q0, p0)
    _, ys = jax.lax.scan(step, initial_carry, None, length=n_steps)
    qs, ps = ys
    return qs, ps

def _eular_solve_with_collision(q0, p0, n_steps, h, velocity, force, collision_handler):
    def step(carry, _):
        q, p = carry
        q_new, p_new = _eular_step_with_collision(q, p, h, velocity, force, collision_handler)
        return (q_new, p_new), (q_new, p_new)
    
    initial_carry = (q0, p0)
    _, ys = jax.lax.scan(step, initial_carry, None, length=n_steps)
    qs, ps = ys
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
        step_size: float,
        collision_handler: Optional[Callable[[jax.Array], jax.Array]] = None
    ):
        super().__init__(kinetic, potential, t0, q0, p0, dim, step_size)
        
        if self._collision_handler:
            # When a collision handler is given:
            _step = functools.partial(
                _eular_step_with_collision, 
                h=self.step_size, velocity=self._velocity, force=self._force, collision_handler=self._collision_handler
            )
            _solve = functools.partial(
                _eular_solve_with_collision, 
                h=self.step_size, velocity=self._velocity, force=self._force, collision_handler=self._collision_handler
            )
        else:
            # no collision handler
            _step = functools.partial(
                _eular_step, 
                h=self.step_size, velocity=self._velocity, force=self._force
            )
            _solve = functools.partial(
                _eular_solve, 
                h=self.step_size, velocity=self._velocity, force=self._force
            )
                
        self._step_impl = jax.jit(_step)
        self._solve_impl = jax.jit(_solve, static_argnames=("n_steps"))
    
def _rk_step(q, p, h, velocity, force, A, B, n_stages, order, dim):
    def loop_body(carry, a):
        kq, kp, s = carry
        
        dq = jnp.dot(kq[:order].T, a) * h
        dp = jnp.dot(kp[:order].T, a) * h
        
        Kq_new = velocity(p + dp)
        Kp_new = force(q + dq)
        
        kq = kq.at[s].set(Kq_new)
        kp = kp.at[s].set(Kp_new)
        
        return (kq, kp, s + 1), None
    Kq = jnp.zeros((n_stages + 1, dim))
    Kp = jnp.zeros((n_stages + 1, dim))
    Kq = Kq.at[0].set(velocity(p))
    Kp = Kp.at[0].set(force(q))
    carry, _ = jax.lax.scan(loop_body, (Kq, Kp, 1), A[1:])
    Kq, Kp, _ = carry
    q_new = q + h * jnp.dot(Kq[:-1].T, B)
    p_new = p + h * jnp.dot(Kp[:-1].T, B)
    return q_new, p_new

def _rk_step_with_collision(q, p, h, velocity, force, A, B, n_stages, order, dim, collision_handler):
    return collision_handler(*_rk_step(q, p, h, velocity, force, A, B, n_stages, order, dim))

def _rk_solve(q0, p0, n_steps, h, velocity, force, A, B, n_stages, order, dim):
    def step(carry, _):
        q, p = carry
        q_new, p_new = _rk_step(q, p, h, velocity, force, A, B, n_stages, order, dim)
        return (q_new, p_new), (q_new, p_new)
    
    initial_carry = (q0, p0)
    _, ys = jax.lax.scan(step, initial_carry, None, length=n_steps)
    qs, ps = ys
    return qs, ps

def _rk_solve_with_collision(q0, p0, n_steps, h, velocity, force, A, B, n_stages, order, dim, collision_handler):
    def step(carry, _):
        q, p = carry
        q_new, p_new = _rk_step_with_collision(q, p, h, velocity, force, A, B, n_stages, order, dim, collision_handler)
        return (q_new, p_new), (q_new, p_new)
    
    initial_carry = (q0, p0)
    _, ys = jax.lax.scan(step, initial_carry, None, length=n_steps)
    qs, ps = ys
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
        step_size: float,
        collision_handler: Optional[Callable[[jax.Array], jax.Array]] = None
    ):
        super().__init__(kinetic, potential, t0, q0, p0, dim, step_size, collision_handler)
        
        if self._collision_handler:
            # When a collision handler is given:
            _step = functools.partial(
                _rk_step_with_collision, 
                h=self.step_size, velocity=self._velocity, force=self._force,
                A=self.A, B=self.B, n_stages=self.n_stages, order=self.order,
                dim=self.dim,  collision_handler=self._collision_handler
            )
            _solve = functools.partial(
                _rk_solve_with_collision, 
                h=self.step_size, velocity=self._velocity, force=self._force,
                A=self.A, B=self.B, n_stages=self.n_stages, order=self.order,
                dim=self.dim,  collision_handler=self._collision_handler
            )
        else:
            # no collision handler
            _step = functools.partial(
                _rk_step, 
                h=self.step_size, velocity=self._velocity, force=self._force,
                A=self.A, B=self.B, n_stages=self.n_stages, order=self.order, dim=self.dim
            )
            _solve = functools.partial(
                _rk_solve, 
                h=self.step_size, velocity=self._velocity, force=self._force,
                A=self.A, B=self.B, n_stages=self.n_stages, order=self.order, dim=self.dim
            )
                
        self._step_impl = jax.jit(_step)
        self._solve_impl = jax.jit(_solve, static_argnames=("n_steps"))
        

class RK4(RungeKutta):
    A = jnp.array([
        [0, 0, 0, 0],
        [1/2, 0, 0, 0],
        [0, 1/2, 0, 0],
        [0, 0, 1, 0]
    ])
    B = jnp.array([1/3, 2/3, 2/3, 1/3])
    order = 4
    n_stages = 4

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
    
def _symplectic_step(q, p, h, velocity, force, C, D):
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

def _symplectic_step_with_coliision(q, p, h, velocity, force, C, D, collision_handler):
    return collision_handler(*_symplectic_step(q, p, h, velocity, force, C, D))


def _symplectic_solve(q0, p0, n_steps, h, velocity, force, C, D):
    def step(carry, _):
        q, p = carry
        q_new, p_new = _symplectic_step(q, p, h, velocity, force, C, D)
        return (q_new, p_new), (q_new, p_new)
    
    initial_carry = (q0, p0)
    _, ys = jax.lax.scan(step, initial_carry, None, length=n_steps)
    qs, ps = ys
    return qs, ps

def _symplectic_solve_with_collision(q0, p0, n_steps, h, velocity, force, C, D, collision_handler):
    def step(carry, _):
        q, p = carry
        q_new, p_new = _symplectic_step_with_coliision(q, p, h, velocity, force, C, D, collision_handler)
        return (q_new, p_new), (q_new, p_new)
    
    initial_carry = (q0, p0)
    _, ys = jax.lax.scan(step, initial_carry, None, length=n_steps)
    qs, ps = ys
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
        step_size: float,
        collision_handler: Optional[Callable[[jax.Array], jax.Array]] = None
    ):
        super().__init__(kinetic, potential, t0, q0, p0, dim, step_size, collision_handler)
        
        if self._collision_handler:
            # When a collision handler is given:
            _step = functools.partial(
                _symplectic_step_with_coliision, 
                h=self.step_size, velocity=self._velocity, force=self._force,
                C=self.C, D=self.D, collision_handler=self._collision_handler
            )
            _solve = functools.partial(
                _symplectic_solve_with_collision, 
                h=self.step_size, velocity=self._velocity, force=self._force,
                C=self.C, D=self.D, collision_handler=self._collision_handler
            )
        else:
            # no collision handler
            _step = functools.partial(
                _symplectic_step, 
                h=self.step_size, velocity=self._velocity, force=self._force,
                C=self.C, D=self.D
            )
            _solve = functools.partial(
                _symplectic_solve, 
                h=self.step_size, velocity=self._velocity, force=self._force,
                C=self.C, D=self.D
            )
                
        self._step_impl = jax.jit(_step)
        self._solve_impl = jax.jit(_solve, static_argnames=("n_steps"))

        
class Leapfrog(Symplectic):
    C = jnp.array([1, 0])
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