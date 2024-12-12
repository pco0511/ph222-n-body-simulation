import jax
import jax.numpy as jnp
import numpy as np

from manim import *
from hamiltonian import *

config.disable_caching = True

class TestScene1(Scene):
    def construct(self):
        debugstrings = []
        
        origin = Dot(color=WHITE)
        xarrow = Arrow(start=ORIGIN, end=RIGHT, color=PURE_RED, buff=0)
        yarrow = Arrow(start=ORIGIN, end=UP, color=PURE_GREEN, buff=0)
        coordinate_system = VGroup(xarrow, yarrow, origin)
        # coordinate_system.shift(LEFT).rotate(PI/4, about_point=origin.get_center()).scale(2, about_point=origin.get_center())
        Mobject.scale
        debugstrings.append(f"{origin.get_center()=}")
        debugstrings.append(f"{xarrow.get_start()=}")
        debugstrings.append(f"{xarrow.get_end()=}")
        debugstrings.append(f"{xarrow.get_vector()=}")
        debugstrings.append(f"{yarrow.get_start()=}")
        debugstrings.append(f"{yarrow.get_end()=}")
        debugstrings.append(f"{yarrow.get_vector()=}")
        
        self.add(coordinate_system)
        
        debugtextobjs = VGroup(*(Text(string, font_size=24) for string in debugstrings))
        debugtextobjs.arrange(DOWN, aligned_edge=LEFT)
        debugtextobjs.align_on_border(LEFT + UP)
        self.add(debugtextobjs)
        
class TestScene2(Scene):
    def construct(self):
        debugstrings = []
        
        coordinate_system = CoordinateArrows2D()
        coordinate_system.shift(LEFT).rotate(PI/4)#.scale(2)
        coordinate_system.turn_on()
        self.add(coordinate_system)
        
        debugstrings.append(f"{coordinate_system.get_origin()=}")
        debugstrings.append(f"{coordinate_system.get_x_vec()=}")
        debugstrings.append(f"{coordinate_system.get_y_vec()=}")

        
        debugtextobjs = VGroup(*(Text(string, font_size=24) for string in debugstrings))
        debugtextobjs.arrange(DOWN, aligned_edge=LEFT)
        debugtextobjs.align_on_border(LEFT + UP)
        self.add(debugtextobjs)
        





class TestScene3(Scene):
    def construct(self):
        G = 1
        m1 = 10
        m2 = 1
        m = jnp.array([m1, m2])
        q0 = jnp.array([-0.25, 0, 2.5, 0])
        P = m1 * m2 * jnp.sqrt(G / ((m1 + m2) * 2.75))
        p0 = jnp.array([0, -P, 0, P])
        
        def kinetic(p):
            p2 = p ** 2
            return jnp.sum((p2[0::2] + p2[1::2]) / (2 * m))

        def potential(q):
            x = q[0::2]
            y = q[1::2]
            epsilon = 1e-12
            
            delta_xij2 = (x[jnp.newaxis, :] - x[:, jnp.newaxis]) ** 2
            delta_yij2 = (y[jnp.newaxis, :] - y[:, jnp.newaxis]) ** 2
            delta_rij2 = delta_xij2 + delta_yij2 + epsilon
            
            mimj = m[jnp.newaxis, :] * m[:, jnp.newaxis]
            return -G * jnp.sum(jnp.triu(mimj / jnp.sqrt(delta_rij2), k=1))
        
        earth = Circle(radius=np.cbrt(10) / 12, color=BLUE, fill_opacity=1)
        moon = Circle(radius=1 / 12, color=GRAY, fill_opacity=1)
        
        N = 2
        celestialsystem = HamiltonianSystem(
            2 * N,
            kinetic=kinetic,
            potential=potential,
            initialvalues=(q0, p0),
            method="forest-ruth",
            simulation_step_size=1e-3
        )
        # celestialsystem.add_elements(earth, moon)

        def earth_updater(earth:Circle):
            x = celestialsystem.qtrackers[0].get_value()
            y = celestialsystem.qtrackers[1].get_value()
            earth.move_to(x * RIGHT + y * UP)
        def moon_updater(moon:Circle):
            x = celestialsystem.qtrackers[2].get_value()
            y = celestialsystem.qtrackers[3].get_value()
            moon.move_to(x * RIGHT + y * UP)   
        
        earth.add_updater(earth_updater)
        moon.add_updater(moon_updater)
        
        self.add(earth, moon)
        self.add(celestialsystem)
        
        self.wait(20)
        

class TestScene4(Scene):
    def construct(self):
        G = 1
        m1 = 10
        m2 = 1
        m = jnp.array([m1, m2])
        q0 = jnp.array([-0.25, 0, 2.5, 0])
        P = m1 * m2 * jnp.sqrt(G / ((m1 + m2) * 2.75))
        p0 = jnp.array([0, -P, 0, P])
        radii = np.array([np.cbrt(10) / 12, 1 / 12])
        
        earth = Circle(radius=radii[0], color=BLUE, fill_opacity=1)
        moon = Circle(radius=radii[1], color=GRAY, fill_opacity=1)

        N = 2
        celestialsystem = HamiltonianSystem(
            2 * N,
            kinetic=kinetic,
            potential=potential,
            initialvalues=(q0, p0),
            method="forest-ruth",
            simulation_step_size=1e-3
        )
        # celestialsystem.add_elements(earth, moon)

        def earth_updater(earth:Circle):
            x = celestialsystem.qtrackers[0].get_value()
            y = celestialsystem.qtrackers[1].get_value()
            earth.move_to(x * RIGHT + y * UP)
        def moon_updater(moon:Circle):
            x = celestialsystem.qtrackers[2].get_value()
            y = celestialsystem.qtrackers[3].get_value()
            moon.move_to(x * RIGHT + y * UP)   
        
        earth.add_updater(earth_updater)
        moon.add_updater(moon_updater)
        
        self.add(earth, moon)
        self.add(celestialsystem)
        
        self.wait(20)
        
        
class SolarSystem(Scene):
    def construct(self):
        G = 1
        M = jnp.array([])
        m1 = 10
        m2 = 1
        m = jnp.array([m1, m2])
        q0 = jnp.array([-0.25, 0, 2.5, 0])
        P = m1 * m2 * jnp.sqrt(G / ((m1 + m2) * 2.75))
        p0 = jnp.array([0, -P, 0, P])
        
        def kinetic(p):
            p2 = p ** 2
            return jnp.sum((p2[0::2] + p2[1::2]) / (2 * m))

        def potential(q):
            x = q[0::2]
            y = q[1::2]
            epsilon = 1e-12
            
            delta_xij2 = (x[jnp.newaxis, :] - x[:, jnp.newaxis]) ** 2
            delta_yij2 = (y[jnp.newaxis, :] - y[:, jnp.newaxis]) ** 2
            delta_rij2 = delta_xij2 + delta_yij2 + epsilon
            
            mimj = m[jnp.newaxis, :] * m[:, jnp.newaxis]
            return -G * jnp.sum(jnp.triu(mimj / jnp.sqrt(delta_rij2), k=1))
        
        earth = Circle(radius=np.cbrt(10) / 12, color=BLUE, fill_opacity=1)
        moon = Circle(radius=1 / 12, color=GRAY, fill_opacity=1)
        
        N = 2
        celestialsystem = HamiltonianSystem(
            2 * N,
            kinetic=kinetic,
            potential=potential,
            initialvalues=(q0, p0),
            method="forest-ruth",
            simulation_step_size=1e-3
        )
        # celestialsystem.add_elements(earth, moon)

        def earth_updater(earth:Circle):
            x = celestialsystem.qtrackers[0].get_value()
            y = celestialsystem.qtrackers[1].get_value()
            earth.move_to(x * RIGHT + y * UP)
        def moon_updater(moon:Circle):
            x = celestialsystem.qtrackers[2].get_value()
            y = celestialsystem.qtrackers[3].get_value()
            moon.move_to(x * RIGHT + y * UP)   
        
        earth.add_updater(earth_updater)
        moon.add_updater(moon_updater)
        
        self.add(earth, moon)
        self.add(celestialsystem)
        
        self.wait(20)
