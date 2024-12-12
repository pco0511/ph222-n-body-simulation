import functools
from manim import *
from hamiltonian import *
import numpy as np

config.disable_caching = True


class CelestialMechanics(Scene):
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
        dummy = Dot(fill_opacity = 0)
        ref_length_tracker = ValueTracker(1.0)
        N = 2
        system = HamiltonianSystem(
            2 * N,
            kinetic=kinetic,
            potential=potential,
            initialvalues=(q0, p0),
            method="forest-ruth",
            simulation_step_size=1e-3
        )
        
        def updater(planet_or_moon:Circle, xtracker, ytracker):
            origin = dummy.get_center()
            ref_len = ref_length_tracker.get_value()
            x = xtracker.get_value()
            y = ytracker.get_value()
            planet_or_moon.move_to(origin + x * ref_len * RIGHT + y * ref_len * UP)

        earth.add_updater(functools.partial(
            updater,
            xtracker = system.qtrackers[0],
            ytracker = system.qtrackers[1]
        ))
        moon.add_updater(functools.partial(
            updater,
            xtracker = system.qtrackers[2],
            ytracker = system.qtrackers[3]
        ))
        
        self.add(earth, moon)
        self.add(system)
        
        
        
        cover = Rectangle(height=10, width=15, color=BLACK)
        self.add(cover)
        
        self.wait(0.5)
        self.play(FadeOut(cover))
        self.wait(6)
        
        self.play(
            dummy.animate.move_to(5 * LEFT),
            ref_length_tracker.animate.set_value(0.7)
        )
        self.wait(1.5)
        system.freeze()
        
        tex_m1 = MathTex(r"m_1").next_to(earth, LEFT + DOWN)
        tex_m2 = MathTex(r"m_2").next_to(moon, LEFT + UP)
        tex_x1p1 = MathTex(r"\mathbf{x}_1, \mathbf{p}_1").next_to(earth, RIGHT + 0.5 * DOWN)
        tex_x2p2 = MathTex(r"\mathbf{x}_2, \mathbf{p}_2").next_to(moon, RIGHT + 0.5 * DOWN)
        
        bidirarrow = DoubleArrow(earth, moon, buff=0.2)
        self.wait(1)
        self.play(
            LaggedStart(
                Write(tex_m1), Write(tex_m2), Write(tex_x1p1), Write(tex_x2p2),
            )
        )
        tex_hamiltonian = MathTex(
            r"H", r" = ", r"\frac{\|\mathbf p_1\|^2}{2m_1}", r"+", r"\frac{\|\mathbf p_2\|^2}{2m_2}", r"-\frac{Gm_1m_2}{\|\mathbf x_1-\mathbf x_2\|}", 
            font_size=36
        )
        tex_hamiltonian.move_to(0.95 * RIGHT + 2 * UP)
        
        self.play(
            Write(tex_hamiltonian),
        )
        self.wait(2)
        
        kinetic_term_1 = tex_hamiltonian.get_part_by_tex(
            r"\frac{\|\mathbf p_1\|^2}{2m_1}"
        )
        kinetic_term_2 = tex_hamiltonian.get_part_by_tex(
            r"\frac{\|\mathbf p_2\|^2}{2m_1}"
        )
        potential_term = tex_hamiltonian.get_part_by_tex(
            r"-\frac{Gm_1m_2}{\|\mathbf x_1-\mathbf x_2\|}"
        )
        
        self.play(
            Circumscribe(kinetic_term_1), 
            Circumscribe(kinetic_term_2)
        )
        self.wait(2)
        self.play(
            Circumscribe(potential_term),
            GrowArrow(bidirarrow)
        )
        
        extra_planets = [
            Circle(radius = 0.12, color=GOLD, fill_opacity=1, stroke_opacity=0),
            Circle(radius = 0.1, color=RED, fill_opacity=1, stroke_opacity=0)
        ]
        tex_full_hamiltonian = MathTex(
            r"H", r" = ", r"\sum_{i=1}^{N}", r"\frac{\|\mathbf p_i\|^2}{2m_1}", r"\sum_{i<j}", r"-\frac{Gm_1m_2}{\|\mathbf x_i-\mathbf x_j\|}", 
            font_size=36
        ).move_to(tex_hamiltonian, aligned_edge=LEFT)
        self.wait(4)
        self.play(
            TransformMatchingTex(
                tex_hamiltonian,
                tex_full_hamiltonian
            )
        )
        
        
        self.wait(4)