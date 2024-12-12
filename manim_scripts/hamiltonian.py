from typing import Callable, Literal, Optional, Self
from manim.typing import Vector3D, Point3D

import numpy as np
import numpy.typing as npt
import jax
import jax.numpy as jnp
from manim import *

import hamiltonian_solver


METHODS = {
    "eular": hamiltonian_solver.Eular,
    "RK4": hamiltonian_solver.RK4,
    "RK23": hamiltonian_solver.RK23,
    "RK45": hamiltonian_solver.RK45,
    "leapfrog": hamiltonian_solver.Leapfrog,
    "forest-ruth": hamiltonian_solver.ForestRuth,
}

def _hermite_spline(t, t1, x1, dxdt1, t2, x2, dxdt2):
    s = (t - t1) / (t2 - t1)
    ss = np.array([1, s, s**2, s**3])
    h00 = np.dot([1, 0, -3, 2], ss)
    h10 = np.dot([0, 1, -2, 1], ss)
    h01 = np.dot([0, 0, 3, -2], ss)
    h11 = np.dot([0, 0, -1, 1], ss)
    x = h00 * x1 + h10 * (t2 - t1) * dxdt1 + h01 * x2 + h11 * (t2 - t1) * dxdt2
    return x

class CoordinateArrows2D(VGroup):
    def __init__(
        self,
        origin_color=WHITE,
        xarrow_color=PURE_RED,
        yarrow_color=PURE_GREEN,
        **kwargs
    ):
        self.origin = Dot(color=origin_color).set_opacity(0)
        self.xarrow = Arrow(start=ORIGIN, end=RIGHT, color=xarrow_color, buff=0).set_opacity(0)
        self.yarrow = Arrow(start=ORIGIN, end=UP, color=yarrow_color, buff=0).set_opacity(0)
        
        super().__init__(self.xarrow, self.yarrow, self.origin, **kwargs)

    def rotate(
        self,
        angle: float,
        axis: Vector3D = OUT,
        about_point: Point3D | None = None,
        **kwargs,
    ) -> Self:
        about_point = about_point or self.origin.get_center()
        super().rotate(angle, axis, about_point, **kwargs)
        return self
    
    def scale(
        self,
        scale_factor: float,
        about_point: Point3D | None = None,
        **kwargs
    ) -> Self:
        about_point = about_point or self.origin.get_center()
        super().scale(scale_factor, about_point=about_point, **kwargs)
        return self

    def get_origin(self):
        return self.origin.get_center()
    
    def get_x_vec(self):
        return self.xarrow.get_vector()
    
    def get_y_vec(self):
        return self.yarrow.get_vector()
    
    def turn_on(self):
        self.origin.set_opacity(1)
        self.xarrow.set_opacity(1)
        self.yarrow.set_opacity(1)
        
    def turn_off(self):
        self.origin.set_opacity(0)
        self.xarrow.set_opacity(0)
        self.yarrow.set_opacity(0)
    
# simulation and visualization for saperable Hamiltonian H(q, p)=T(p)+V(q)
class HamiltonianSystem(VGroup):
    def __init__(
        self,
        n_coordinates: int,
        hamiltonian: Optional[Callable[[jax.Array, jax.Array], float]] = None,
        kinetic: Optional[Callable[[jax.Array], float]] = None,
        potential: Optional[Callable[[jax.Array], float]] = None,
        reference_time: float = 1,
        reference_length: float = 1,
        simulation_step_size: float = 1e-3,
        initial_t: float = 0,
        initialvalues: Optional[tuple[npt.NDArray, npt.NDArray]] = None,
        method: Literal["eular", "RK23", "RK45", "leapfrog", "forest-ruth"] = "eular",
        simulation_batches = 8000,
        collision_handler: Optional[Callable[[jax.Array], jax.Array]] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # System settings
        self.n_coordinates = n_coordinates
        self.kinetic = kinetic or (lambda p: 0)
        self.potential = potential or (lambda q: 0)
        self.hamiltonian = hamiltonian or (lambda q, p: self.kinetic(p) + self.potential(q))
        self.velocity = jax.grad(kinetic)
        potential_gradient = jax.grad(potential)
        self.force = lambda q: -potential_gradient(q)
        self.collision_handler = collision_handler
        
        # Simulation settings
        self.simulation_step_size = simulation_step_size
        self.simulation_batches = simulation_batches
        
        # visualization settings
        self.reference_time_tracker = ValueTracker(reference_time)
        self.reference_length_tracker = ValueTracker(reference_length)
        self.freezed = False
        
        # simulation initializations
        self.t_cache = np.array([initial_t])
        initial_q, initial_p = initialvalues or (np.zeros(n_coordinates), np.zeros(n_coordinates))
        self.q_cache = np.array(initial_q)[np.newaxis, :]
        self.p_cache = np.array(initial_p)[np.newaxis, :]
        initial_h = self.hamiltonian(initial_q, initial_p)
        self.h_cache = np.array([initial_h])
        self.integrator: hamiltonian_solver.HamiltonianSolver = METHODS[method](
            self.kinetic,
            self.potential,
            initial_t,
            jnp.array(initial_q),
            jnp.array(initial_p),
            self.n_coordinates,
            self.simulation_step_size,
            self.collision_handler
        )
        
        # trackers
        self.ttracker = ValueTracker(initial_t)
        self.qtrackers = [ValueTracker(q) for q in initial_q]
        self.ptrackers = [ValueTracker(p) for p in initial_p]
        self.htracker = ValueTracker(self.hamiltonian(jnp.array(initial_q), jnp.array(initial_p)))
        # self.add(self.ttracker)
        # self.add(self.qtrackers)
        # self.add(self.ptrackers)
        
        self.labeled_objects = {}
        self.unlabeled_objects = []
        
        # adding updater
        def updater(system: HamiltonianSystem, dt:float):
            if self.freezed == False:
                system.ttracker += dt * self.reference_time_tracker.get_value()
            system.update_trakers()
        self.add_updater(updater)
        
        
    def force_step(self, n_steps=1):
        self.ttracker += self.simulation_step_size * n_steps
        self.update_trakers()
            
    def update_trakers(self):
        t = self.ttracker.get_value()
        # print(f"{t=}, {self.t_cache[0]=}, {self.t_cache[-1]=}, {self.simulation_step_size=}")
        if self.t_cache[-1] <= t:
            print(1)
            self.simulate()
        idxl = int(np.floor((t - self.t_cache[0]) / self.simulation_step_size))
        idxh = idxl + 1
        
        t1 = self.t_cache[idxl]
        t2 = self.t_cache[idxh]
        
        q1 = self.q_cache[idxl]
        q2 = self.q_cache[idxh]
        
        p1 = self.p_cache[idxl]
        p2 = self.p_cache[idxh]
        
        h1 = self.hamiltonian(q1, p1)
        h2 = self.hamiltonian(q2, p2)
        
        grad_q1 = self.velocity(p1)
        grad_q2 = self.velocity(p2)
        
        grad_p1 = self.force(q1)
        grad_p2 = self.force(q2)
        
        q = _hermite_spline(t, t1, q1, grad_q1, t2, q2, grad_q2)
        p = _hermite_spline(t, t1, p1, grad_p1, t2, p2, grad_p2)
        h = _hermite_spline(t, t1, h1, 0, t2, h2, 0)
        for i in range(self.n_coordinates):
            self.qtrackers[i].set_value(q[i])
            self.ptrackers[i].set_value(p[i])
        self.htracker.set_value(h)
    
    def simulate(self):
        qs, ps, ts = self.integrator.solve(self.simulation_batches)
        hs = jax.vmap(self.hamiltonian)(qs, ps)
        self.t_cache = np.concatenate([self.t_cache, np.array(ts)])
        self.q_cache = np.concatenate([self.q_cache, np.array(qs)])
        self.p_cache = np.concatenate([self.p_cache, np.array(ps)])
        self.h_cache = np.concatenate([self.h_cache, np.array(hs)])
        
    def add_elements(self, *unlabeled_mobjs, **labeled_mobjs):
        self.add(*unlabeled_mobjs)
        idxl = len(self.unlabeled_objects)
        self.unlabeled_objects += list(unlabeled_mobjs)
        idxh = len(self.unlabeled_objects)
        added_keys = []
        for key, mobj in labeled_mobjs.items():
            if key in self.labeled_objects:
                print(f"Failed to add a mobject, key collision: {key}")
            else:
                self.add(mobj)
                self.labeled_objects[key] = mobj
                added_keys.append(key)
        return (list(range(idxl, idxh)), added_keys)
    
    def set_variables(self, q, p, t=0):
        self.t_cache = np.array([t])
        self.q_cache = np.array(q)[np.newaxis, :]
        self.p_cache = np.array(p)[np.newaxis, :]
        self.integrator.set_initial_values(
            jnp.array(q), 
            jnp.array(p),
            t
        )
        self.update_trakers()
    
    def get_time(self):
        return self.ttracker.get_value()
    
    def get_coordinates(self):
        q = np.array((qtracker.get_value() for qtracker in self.qtrackers), dtype=np.float64)
        return q
    
    def get_momentums(self):
        p = np.array((ptracker.get_value() for ptracker in self.ptrackers), dtype=np.float64)
        return p
    
    def get_energy(self):
        return self.htracker.get_value()
    
    def get_step_per_second(self):
        return int(self.reference_time_tracker.get_value() / self.integrator.step_size)
    
    def get_times(self, sampling_rate=None):
        sampling_rate = sampling_rate or (self.get_step_per_second() // 25)
        upper = np.searchsorted(self.t_cache, self.ttracker.get_value())
        ts = self.t_cache[0:upper + 1:sampling_rate]
        return ts
        
    def get_energies(self, sampling_rate=None):
        sampling_rate = sampling_rate or (self.get_step_per_second() // 25)
        upper = np.searchsorted(self.t_cache, self.ttracker.get_value())
        hs = self.h_cache[0:upper + 1:sampling_rate]
        return hs
    
    def freeze(self):
        self.freezed = True
        
    def unfreeze(self):
        self.freezed = False
        
    def to_system_time(self, t):
        return t * self.reference_time_tracker.get_value()
    
    def to_system_length(self, x):
        return x * self.reference_length_tracker.get_value()
    
    def to_camera_time(self, t):
        return t / self.reference_time_tracker.get_value()
    
    def to_camera_length(self, x):
        return x / self.reference_length_tracker.get_value()
    
    @property
    def t(self):
        return self.ttracker.get_value()
    
    @property
    def q(self):
        return np.array(qtracker.get_value() for qtracker in self.qtrackers)
    
    @property
    def p(self):
        return np.array(ptracker.get_value() for ptracker in self.ptrackers)
    
    @property
    def h(self):
        return self.htracker.get_value()
    
class CelestialSystem(HamiltonianSystem):
    def __init__(
        self,
        n_bodies: int,
        masses: Optional[npt.NDArray] = None,
        gravit_const: float = 1,
        reference_time: float = 1,
        reference_length: float = 1,
        simulation_step_size: float = 1e-3,
        initial_t: float = 0,
        initialvalues: Optional[tuple[npt.NDArray, npt.NDArray]] = None,
        method: Literal["eular", "RK23", "RK45", "leapfrog", "forest-ruth"] = "forest-ruth",
        simulation_batches = 8000,
        collision_handler: Optional[Callable[[jax.Array], jax.Array]] = None,
        **kwargs
    ):
        self.masses = jnp.array(masses)
        self.gravit_const = gravit_const
        
        def kinetic(p, masses=self.masses):
            p2 = p ** 2
            return jnp.sum((p2[0::2] + p2[1::2]) / (2 * masses))

        def potential(q, masses=self.masses, gravit_const=self.gravit_const):
            x = q[0::2]
            y = q[1::2]
            epsilon = 1e-12
            
            delta_xij2 = (x[jnp.newaxis, :] - x[:, jnp.newaxis]) ** 2
            delta_yij2 = (y[jnp.newaxis, :] - y[:, jnp.newaxis]) ** 2
            delta_rij2 = delta_xij2 + delta_yij2 + epsilon
            
            mimj = masses[jnp.newaxis, :] * masses[:, jnp.newaxis]
            return -gravit_const * jnp.sum(jnp.triu(mimj / jnp.sqrt(delta_rij2), k=1))
        
        def hamiltonian(q, p):
            return kinetic(p) + potential(q)
        
        super().__init__(
            n_bodies * 2,
            hamiltonian,
            kinetic,
            potential,
            reference_time,
            reference_length,
            simulation_step_size,
            initial_t,
            initialvalues,
            method,
            simulation_batches,
            collision_handler
            **kwargs
        )
    
    def initialize_by_data(
        self,
        names,
        radii,
        colors,
        q0,
        p0,
    ):
        for name, radius, color in zip(names, radii, colors):
            mobj = Circle(
                radius
            )

    
    
    def initialize_solar_system(
        self,
        names,
        radii,
        perihelions,
        perih_args,
        perih_speeds,
        colors,
    ):
        data = (
            names,
            self.masses,
            radii,
            perihelions,
            perih_args,
            perih_speeds,
            colors,
        )
        q = np.empty((self.n_coordinates,))
        p = np.empty((self.n_coordinates,))
        for i, (name, mass, radius, perihelion, perih_arg, perih_speed, color) in enumerate(zip(*data)):
            mobj = Circle(
                radius=self.to_camera_length(radius),
                color=color,
                fill_opacity=1
            )
            ix = 2 * i
            iy = 2 * i + 1
            def updater(mobj:Circle):
                x = self.to_camera_length(self.qtrackers[ix].get_value())
                y = self.to_camera_length(self.qtrackers[iy].get_value())
                mobj.move_to(x * RIGHT + y * UP)
            mobj.add_updater(updater)
            perih_momentum = mass * perih_speed
            q[ix] = perihelion * np.cos(perih_arg)
            q[iy] = perihelion * np.sin(perih_arg)
            p[ix] = -perih_momentum * np.sin(perih_arg)
            p[iy] = perih_momentum * np.cos(perih_arg)
            self.add_elements(**{name:mobj})
        self.set_variables(q, p, 0)