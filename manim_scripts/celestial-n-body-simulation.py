import itertools
import functools
from manim import *
from hamiltonian import *
import numpy as np

config.disable_caching = True
# config.update("jax_debug_nans", True)

class IntroductionStart(Scene):
    def construct(self):
        kwargs = dict(
            font_size = 22,
            opacity = 0.2
        )
        text_mdsim = Text("1. Molecular Dynamics", **kwargs).set_opacity(0.5)
        text_plasmaph = Text("2. Plasma Physics", **kwargs).set_opacity(0.5)
        text_space_probe = Text("3. Launching Space Probe", **kwargs).set_opacity(0.5)
        text_solar_sim = Text("4. Solar System Simulation", **kwargs).set_opacity(0.5)
        text_group = VGroup(text_mdsim, text_plasmaph, text_space_probe, text_solar_sim)
        text_group.arrange(DOWN, buff=1.5, aligned_edge=LEFT).to_corner(LEFT + UP).shift(0.25 * DOWN)
        
        self.wait(0.5)
        self.play(Write(text_group))
        self.wait(2)
        
class IntroductionMDSim(Scene):
    def construct(self):
        kwargs = dict(
            font_size = 22,
            opacity = 0.2
        )
        text_mdsim = Text("1. Molecular Dynamics", **kwargs).set_opacity(0.5)
        text_plasmaph = Text("2. Plasma Physics", **kwargs).set_opacity(0.5)
        text_space_probe = Text("3. Launching Space Probe", **kwargs).set_opacity(0.5)
        text_solar_sim = Text("4. Solar System Simulation", **kwargs).set_opacity(0.5)
        text_group = VGroup(text_mdsim, text_plasmaph, text_space_probe, text_solar_sim)
        text_group.arrange(DOWN, buff=1.5, aligned_edge=LEFT).to_corner(LEFT + UP).shift(0.25 * DOWN)
        self.add(text_group)

        
        self.wait(0.5)
        left_edge = text_mdsim.get_left()
        
        box = Rectangle(width=5, height=5).move_to(2.5 * RIGHT)
        cover = Rectangle(color=BLACK, width=5, height=5).scale(1.1).set_opacity(1).move_to(box)
        m = 0.01
        e = 1e-6
        sigma = 0.01
        g = 1
        molecule_radius = 0.008
        visualization_radius = molecule_radius * 5
        n_particles = 200
        
        @jax.vmap
        def f(r):
            return ((sigma / r) ** 12 - (sigma / r) ** 6)
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
            
            interaction = 4 * e * jnp.sum(jnp.triu(f(jnp.sqrt(delta_rij2)), k=1))
            return interaction
        def collision_handler(q, p):
            collided_lower = q < 0.0 + molecule_radius
            collided_upper = q > 1 - molecule_radius
            # assert jnp.all(q > 0), f"{q=}"
            p = jnp.where(collided_lower | collided_upper, -p, p)
            
            q = jnp.where(collided_lower, 2 * molecule_radius - q, q)
            q = jnp.where(collided_upper, 2 * (1 - molecule_radius) - q, q)
            return q, p
        q0 = np.random.uniform(0, 1, (n_particles * 2,))
        p0 = np.random.normal(0, 0.0005, (n_particles * 2,))
        mdsystem = HamiltonianSystem(
            2 * n_particles,
            kinetic=kinetic,
            potential=potential,
            initialvalues=(q0, p0),
            method='forest-ruth',
            collision_handler=collision_handler
        )
        molecules = [
            Circle(
                visualization_radius, 
                color=BLUE, 
                stroke_opacity=0,
                fill_opacity=0.9
            ) for _ in range(n_particles)
        ]
        def molecule_updater(molecule: Circle, xtracker: ValueTracker, ytracker: ValueTracker):
            origin = box.get_edge_center(LEFT + DOWN)
            frame_up = 5 * UP
            frame_right = 5 * RIGHT
            x = xtracker.get_value()
            y = ytracker.get_value()
            molecule.move_to(origin + x * frame_right + y * frame_up)
        for i, molecule in enumerate(molecules):
            xidx = 2 * i
            yidx = 2 * i + 1
            molecule.add_updater(
                functools.partial(
                    molecule_updater, 
                    xtracker=mdsystem.qtrackers[xidx],
                    ytracker=mdsystem.qtrackers[yidx]
                )
            )
        self.add(box)
        self.add(mdsystem)
        self.add(*molecules)
        self.add(cover)
        self.play(
            text_mdsim.animate.scale(1.25, about_point=left_edge).set_opacity(1),
            FadeOut(cover)
        )
        self.wait(10)
        
        self.play(
            text_mdsim.animate.scale(0.8, about_point=left_edge).set_opacity(0.5),
            FadeIn(cover)
        )
        self.wait(0.5)
        
class IntroductionPlasmaPh(Scene):
    def construct(self):
        kwargs = dict(
            font_size = 22,
            opacity = 0.2
        )
        text_mdsim = Text("1. Molecular Dynamics", **kwargs).set_opacity(0.5)
        text_plasmaph = Text("2. Plasma Physics", **kwargs).set_opacity(0.5)
        text_space_probe = Text("3. Launching Space Probe", **kwargs).set_opacity(0.5)
        text_solar_sim = Text("4. Solar System Simulation", **kwargs).set_opacity(0.5)
        text_group = VGroup(text_mdsim, text_plasmaph, text_space_probe, text_solar_sim)
        text_group.arrange(DOWN, buff=1.5, aligned_edge=LEFT).to_corner(LEFT + UP).shift(0.25 * DOWN)
        self.add(text_group)
        url = Text(
            r"www.youtube.com/watch?v=lGk4QyBFR0s",
            font_size=20
        ).move_to(0.45 * RIGHT + 3 * DOWN)
     
        self.wait(0.5)
        left_edge = text_plasmaph.get_left()
        self.play(text_plasmaph.animate.scale(1.25, about_point=left_edge).set_opacity(1))
        
        self.play(Write(url))
        self.wait(8)
        self.play(Unwrite(url))
        self.play(text_plasmaph.animate.scale(0.8, about_point=left_edge).set_opacity(0.5))
        self.wait(0.5)
    
class IntroductionSpaceProbe(Scene):
    def construct(self):
        kwargs = dict(
            font_size = 22,
            opacity = 0.2
        )
        text_mdsim = Text("1. Molecular Dynamics", **kwargs).set_opacity(0.5)
        text_plasmaph = Text("2. Plasma Physics", **kwargs).set_opacity(0.5)
        text_space_probe = Text("3. Launching Space Probe", **kwargs).set_opacity(0.5)
        text_solar_sim = Text("4. Solar System Simulation", **kwargs).set_opacity(0.5)
        text_group = VGroup(text_mdsim, text_plasmaph, text_space_probe, text_solar_sim)
        text_group.arrange(DOWN, buff=1.5, aligned_edge=LEFT).to_corner(LEFT + UP).shift(0.25 * DOWN)
        self.add(text_group)
        url = Text(
            r"www.youtube.com/watch?v=alfZC3nJOGI",
            font_size=20
        ).move_to(0.45 * RIGHT + 3 * DOWN)
        
        self.wait(0.5)
        left_edge = text_space_probe.get_left()
        self.play(text_space_probe.animate.scale(1.25, about_point=left_edge).set_opacity(1))
        
        self.play(Write(url))
        self.wait(8)
        self.play(Unwrite(url))
        
        self.play(text_space_probe.animate.scale(0.8, about_point=left_edge).set_opacity(0.5))
        self.wait(0.5)
    
class IntroductionSolarSystem(Scene):
    def construct(self):
        kwargs = dict(
            font_size = 22,
            opacity = 0.2
        )
        text_mdsim = Text("1. Molecular Dynamics", **kwargs).set_opacity(0.5)
        text_plasmaph = Text("2. Plasma Physics", **kwargs).set_opacity(0.5)
        text_space_probe = Text("3. Launching Space Probe", **kwargs).set_opacity(0.5)
        text_solar_sim = Text("4. Solar System Simulation", **kwargs).set_opacity(0.5)
        text_group = VGroup(text_mdsim, text_plasmaph, text_space_probe, text_solar_sim)
        text_group.arrange(DOWN, buff=1.5, aligned_edge=LEFT).to_corner(LEFT + UP).shift(0.25 * DOWN)
        self.add(text_group)

        
        
        colors = [
            ORANGE, 
            GRAY, GOLD, GREEN, RED,
            GOLD_A, YELLOW_A, TEAL, PURE_BLUE
        ]
        visualization_radii = [
            0.23, 
            0.05, 0.09, 0.09, 0.07, 
            0.18, 0.16, 0.12, 0.15
        ]
        masses = jnp.array([
            1e7,
            1, 8, 10, 5,
            400, 250, 150, 200            
        ])
        initial_angle = jnp.array([
            0,
            1.1912686 , 4.17235915, 6.17671852, 1.5862776,
            3.98737739, 2.20751569, 3.07580263, 3.78725046
        ])
        # real-scale
        # orbit_radii = jnp.array([
        #     1e-12,
        #     4.6, 10.7, 14.7, 20.7,
        #     74.1, 135, 274, 446
        # ])
        # reduced-scale
        orbit_radii = jnp.array([
            1e-12,
            18, 26, 36, 47,
            74, 94, 112, 130
        ])
        
        
        G = 1e-2
        p_norm = masses * jnp.sqrt(G * masses[0] / orbit_radii)
        # *3/446
        planets = [
            Circle(
                radius=r,
                color=c,
                fill_opacity=1,
                stroke_opacity=0
            ) for r, c in zip(
                visualization_radii, colors
            )
        ]
        orbit_trajectories = [
            Circle(
                radius=r / 40,
                color=WHITE,
                fill_opacity=0,
                stroke_opacity=1,
                stroke_width=1
            ).move_to(2.5 * RIGHT) for r in orbit_radii
        ]
        q0 = np.empty((18,))
        p0 = np.empty((18,))
        cos = np.cos(initial_angle)
        sin = np.sin(initial_angle)
        qx = orbit_radii * cos
        qy = orbit_radii * sin
        px = -p_norm * sin
        py = p_norm * cos
        q0[0::2] = qx
        q0[1::2] = qy
        p0[0::2] = px
        p0[1::2] = py
        p0[0:2] = 0
        
        
        def kinetic(p):
            p2 = p ** 2
            return jnp.sum((p2[0::2] + p2[1::2]) / (2 * masses))
        
        def potential(q):
            x = q[0::2]
            y = q[1::2]
            epsilon = 1e-12
            
            delta_xij2 = (x[jnp.newaxis, :] - x[:, jnp.newaxis]) ** 2
            delta_yij2 = (y[jnp.newaxis, :] - y[:, jnp.newaxis]) ** 2
            delta_rij2 = delta_xij2 + delta_yij2 + epsilon
            
            mimj = masses[jnp.newaxis, :] * masses[:, jnp.newaxis]
            return -G * jnp.sum(jnp.triu(mimj / jnp.sqrt(delta_rij2), k=1))
        
        
        solarsystem = HamiltonianSystem(
            2 * 9,
            kinetic=kinetic,
            potential=potential,
            initialvalues=(q0, p0),
            method='forest-ruth',
        )
        
        def planet_updater(planet: Circle, xtracker: ValueTracker, ytracker: ValueTracker):
            origin = 2.5 * RIGHT
            frame_up = (3 / 120) * UP
            frame_right = (3 / 120) * RIGHT
            x = xtracker.get_value()
            y = ytracker.get_value() 
            planet.move_to(origin + x * frame_right + y * frame_up)
        for i, planet in enumerate(planets):
            xidx = 2 * i
            yidx = 2 * i + 1
            planet.add_updater(
                functools.partial(
                    planet_updater, 
                    xtracker=solarsystem.qtrackers[xidx],
                    ytracker=solarsystem.qtrackers[yidx]
                )
            )
        self.add(*orbit_trajectories)
        self.add(solarsystem)
        self.add(*planets)
        
        cover = Rectangle(color=BLACK, width=7.2, height=7.2).set_opacity(1).move_to(2.5 * RIGHT)
        self.add(cover)
        
        self.wait(0.5)
        left_edge = text_solar_sim.get_left()
        
        self.play(
            text_solar_sim.animate.scale(1.25, about_point=left_edge).set_opacity(1),
            FadeOut(cover)
        )
        self.wait(12)
        
        self.play(
            text_solar_sim.animate.scale(0.8, about_point=left_edge).set_opacity(0.5),
            FadeIn(cover)
        )
        self.wait(0.5)
    
class IntroductionManybody(Scene):
    def construct(self):
        kwargs = dict(
            font_size = 22,
            opacity = 0.2
        )
        text_mdsim = Text("1. Molecular Dynamics", **kwargs).set_opacity(0.5)
        text_plasmaph = Text("2. Plasma Physics", **kwargs).set_opacity(0.5)
        text_space_probe = Text("3. Launching Space Probe", **kwargs).set_opacity(0.5)
        text_solar_sim = Text("4. Solar System Simulation", **kwargs).set_opacity(0.5)
        text_group = VGroup(text_mdsim, text_plasmaph, text_space_probe, text_solar_sim)
        text_group.arrange(DOWN, buff=1.5, aligned_edge=LEFT).to_corner(LEFT + UP).shift(0.25 * DOWN)
        self.add(text_group)
        
        self.wait(0.5)
        text_mdsim_left = text_mdsim.get_left()
        text_plasmaph_left = text_plasmaph.get_left()
        text_space_probe_left = text_space_probe.get_left()
        text_solar_sim_left = text_solar_sim.get_left()
        
        self.play(
            LaggedStart(
                text_mdsim.animate.scale(1.25, about_point=text_mdsim_left).set_opacity(1),
                text_plasmaph.animate.scale(1.25, about_point=text_plasmaph_left).set_opacity(1),
                text_space_probe.animate.scale(1.25, about_point=text_space_probe_left).set_opacity(1),
                text_solar_sim.animate.scale(1.25, about_point=text_solar_sim_left).set_opacity(1),
                lag_ratio=0.15
            )
        )
        
        brace = Brace(text_group, direction=RIGHT)
        
        text_manybody = Text("Many-Body Problem", font_size=48)
        text_manybody.next_to(brace, direction=RIGHT, buff=1)
        
        self.play(
            LaggedStart(
                GrowFromCenter(brace),
                FadeIn(text_manybody),
                lag_ratio=0.5
            )
        )
        self.wait(1)
        self.play(
            Circumscribe(text_manybody)
        )
        # self.play(GrowFromCenter(brace))
        # self.play(FadeIn(text_manybody), buff=1)
        
        
        self.wait(15)
        
        self.play(*(FadeOut(mobj) for mobj in self.mobjects))
        
        self.wait(0.5)
        
class CelestialMechanics(Scene):
    def construct(self):
        G = 0.09
        # n_planets = 30
        n_planets = 8
        m = jnp.full((n_planets,), 1)
        
        np.random.seed(9876)
        q0 = np.random.normal(0, 2, (2 * n_planets,))
        
        v_noise = np.random.normal(0, 1, (2 * n_planets,))
        v_angular = np.empty((2 * n_planets,))
        r = np.sqrt(q0[1::2]**2 + q0[0::2]**2)
        v_angular[0::2] = -(q0[1::2] / r) * (np.tanh(r / 3) + 0.01) 
        v_angular[1::2] = (q0[0::2] / r) * (np.tanh(r / 3) + 0.01)
        p0 = 0.15 * v_noise + 0.8 * v_angular
        p0[0::2] *= m
        p0[1::2] *= m
        # centering
        q0[0::2] -= np.mean(q0[0::2])
        q0[1::2] -= np.mean(q0[1::2])
        p0[0::2] -= np.mean(p0[0::2])
        p0[1::2] -= np.mean(p0[1::2])
        q0 = jnp.array(q0)
        p0 = jnp.array(p0)
        
        visualization_radius = 0.05
        
        def kinetic(p):
            p2 = p ** 2
            return jnp.sum((p2[0::2] + p2[1::2]) / (2 * m))
        def potential(q):
            x = q[0::2]
            y = q[1::2]
            epsilon = 1e-3
            
            delta_xij2 = (x[jnp.newaxis, :] - x[:, jnp.newaxis]) ** 2
            delta_yij2 = (y[jnp.newaxis, :] - y[:, jnp.newaxis]) ** 2
            delta_rij2 = delta_xij2 + delta_yij2 + epsilon
            
            mimj = m[jnp.newaxis, :] * m[:, jnp.newaxis]
            return -G * jnp.sum(jnp.triu(mimj / jnp.sqrt(delta_rij2), k=1))
        
        
        celestial_system = HamiltonianSystem(
            2 * n_planets,
            kinetic=kinetic,
            potential=potential,
            initialvalues=(q0, p0),
            method='forest-ruth'
        )
        planets = [
            Circle(
                visualization_radius, 
                color=WHITE, 
                stroke_opacity=0,
                fill_opacity=1
            ) for _ in range(n_planets)
        ]
        dummy = Dot(fill_opacity = 0)
        ref_length_tracker = ValueTracker(1.0)
        def planet_updater(planet: Circle, xtracker: ValueTracker, ytracker: ValueTracker):
            origin = dummy.get_center()
            ref_len = ref_length_tracker.get_value()
            x = xtracker.get_value()
            y = ytracker.get_value()
            planet.move_to(origin + x * ref_len * RIGHT + y * ref_len * UP)
            
        for i, planet in enumerate(planets):
            xidx = 2 * i
            yidx = 2 * i + 1
            planet.add_updater(
                functools.partial(
                    planet_updater, 
                    xtracker=celestial_system.qtrackers[xidx],
                    ytracker=celestial_system.qtrackers[yidx]
                )
            )
        self.add(celestial_system) 
        self.add(*planets)
        
        
        cover = Rectangle(height=10, width=15, color=BLACK, fill_opacity=1)
        self.add(cover)
        
        celestial_system.reference_time = 1
        
        self.wait(0.5)
        self.play(FadeOut(cover))
        print(celestial_system.h)
        self.wait(6)
        print(celestial_system.h)
        
        self.play(
            dummy.animate(run_time=1.5).move_to(5.25 * LEFT),
            ref_length_tracker.animate(run_time=1.5).set_value(0.5),
        )      
        self.wait(1.5)
        celestial_system.freeze()
        
        labels = VGroup()
        for i, planet in enumerate(planets):
            label = MathTex(f"{i}").next_to(planet, direction=RIGHT + DOWN)
            labels.add(label)
        # self.add(labels)
        
        self.wait(3)
        
            
        
        font_size = 36
        hamiltonian = MathTex(
            "H", "=", "T", "+", "V",
            font_size=font_size
        ).move_to(2 * UP + 1.5 * LEFT)
        
        kinetic_term = hamiltonian.get_part_by_tex("T")
        potential_term = hamiltonian.get_part_by_tex("V")
        
        
        kinetic_hamiltonian = MathTex(
            "T", "=", r"\sum_{j=1}^N", r"\left[", r"\sum_{i=1}^3\frac{p_{ij}^2}{2m_j}", "+", r"\sum_{i=1}^3\frac{L_{ij}^2}{2I_j}", r"\right]",
            font_size=font_size
        ).next_to(hamiltonian, direction=DOWN, aligned_edge =LEFT)
        reduced_kinetic_hamiltonian = MathTex(
            "T", "=", r"\sum_{j=1}^N", r"\sum_{i=1}^3\frac{p_{ij}^2}{2m_j}",
            font_size=font_size
        ).next_to(hamiltonian, direction=DOWN, aligned_edge =LEFT)  
        tanslation_part = kinetic_hamiltonian.get_part_by_tex(r"\sum_{i=1}^3\frac{p_{ij}^2}{2m_j}")
        rotation_part = kinetic_hamiltonian.get_part_by_tex(r"\sum_{i=1}^3\frac{L_{ij}^2}{2I_j}")
        
        
        # self.add(hamiltonian)
        # self.add(kinetic_hamiltonian)
        
        self.play(Write(hamiltonian))
        
        self.wait(1)
        
        self.play(
            kinetic_term.animate.set_color(YELLOW)
        )
        self.play(
            Write(kinetic_hamiltonian)
        )
        self.play(
            kinetic_term.animate.set_color(WHITE)
        )
        
        self.wait(1)
        
        self.play(Circumscribe(tanslation_part))
        
        self.wait(1)
        
        self.play(Circumscribe(rotation_part))
        
        self.wait(2)
        
        
        def in_screen(point: Point3D):
            x, y, _ = point
            return -4 * (16 / 9) < x < 4 * (16 / 9) and -4 < y < 4
        planets_in_scene = [planet for planet in planets if in_screen(planet.get_center())]
        pairs = itertools.combinations(planets_in_scene, 2)
        interactions = VGroup(*(Line(obj1, obj2, buff=0.05, stroke_width=1) for obj1, obj2 in pairs))
        # interactions.set_stroke_width(0.05)
        interactions.set_stroke_color(YELLOW)
        potential_hamiltonian = MathTex(
            r"V=\frac{1}{2}G\int\frac{\rho(\mathbf x_1)\rho(\mathbf x_2)}{\|\mathbf x_1-\mathbf x_2\|}dV_1dV_2",
            font_size=font_size
        )
        V_only = MathTex(
            r"V=",
            font_size=font_size
        ) 
        point_particle_interaction = MathTex(
            r"V=-\sum_{i>j} \frac{Gm_im_j}{\|\mathbf x_i - \mathbf x_j\|}",
            font_size=font_size
        )
        potential_hamiltonian.next_to(kinetic_hamiltonian, direction=DOWN, aligned_edge=LEFT)
        V_only.next_to(kinetic_hamiltonian, direction=DOWN, aligned_edge=LEFT)
        point_particle_interaction.next_to(kinetic_hamiltonian, direction=DOWN, aligned_edge=LEFT)
        
        self.wait(2)
        
        # self.add(potential_hamiltonian)
        # self.add(interactions)
        self.play(
            potential_term.animate.set_color(YELLOW)
        )
        self.play(
            Write(potential_hamiltonian)
        )
        self.play(
            potential_term.animate.set_color(WHITE)
        )
        
        
        self.wait(3)
        
        
        multipole_expansion = MathTex(
            r"\int\frac{G\rho(\mathbf x')}{\|\mathbf x-\mathbf x'\|} dV'",
            r"=", r"\frac{Gm}{\|\mathbf x\|}", r"+", r"\frac{G\mathbf p\cdot\mathbf x}{\|\mathbf x\|^3}",
            r"+", r"\frac{G\mathbf x^\top\mathbf Q\mathbf x}{\|\mathbf x\|^5}", 
            r"+\cdots",
            font_size=font_size 
        )
        reduced_multipole_expansion = MathTex(
            r"\int\frac{G\rho(\mathbf x')}{\|\mathbf x-\mathbf x'\|} dV'",
            r"=", r"\frac{Gm}{\|\mathbf x\|}",
            r"+", r"\frac{G\mathbf x^\top\mathbf Q\mathbf x}{\|\mathbf x\|^5}", 
            r"+\cdots",
            font_size=font_size
        )
        monopole_only = MathTex(
            r"\int\frac{G\rho(\mathbf x')}{\|\mathbf x-\mathbf x'\|} dV'",
            r"=", r"\frac{Gm}{\|\mathbf x\|}",
            font_size=font_size
        )
        multipole_expansion.next_to(potential_hamiltonian, direction=DOWN, aligned_edge=LEFT)
        reduced_multipole_expansion.next_to(potential_hamiltonian, direction=DOWN, aligned_edge=LEFT)
        monopole_only.next_to(potential_hamiltonian, direction=DOWN, aligned_edge=LEFT)
        
        dipole_term = multipole_expansion.get_part_by_tex(r"\frac{G\mathbf p\cdot\mathbf x}{\|\mathbf x\|^3}")
        quadrupole_term = reduced_multipole_expansion.get_part_by_tex(r"\frac{G\mathbf x^\top\mathbf Q\mathbf x}{\|\mathbf x\|^5}")
        higher_order_terms = reduced_multipole_expansion.get_part_by_tex(r"+\cdots")
        
        self.play(Write(multipole_expansion))
        
        self.wait(2)
        
        self.play(Circumscribe(dipole_term))
        
        self.wait(2)
        
        self.play(
            TransformMatchingTex(
                multipole_expansion,
                reduced_multipole_expansion
            )
        )
        
        self.wait(2)
        text_font_size = 24
        text_precession = Text("Precession", font_size=text_font_size).next_to(quadrupole_term, direction=DOWN)
        text_satellite_orbit = Text("Satellite Orbit", font_size=text_font_size).next_to(text_precession, direction=DOWN, aligned_edge=LEFT)
        indication_box1 = SurroundingRectangle(quadrupole_term)
        indication_box2 = SurroundingRectangle(higher_order_terms)
        self.play(
            Create(indication_box1)
        )
        self.play(
            Write(text_precession)
        )
        self.wait(1)
        self.play(
            Write(text_satellite_orbit)
        )
        self.wait(2)
        self.play(
            ReplacementTransform(
                indication_box1,
                indication_box2
            )
        )
        self.wait(2)
        plus = reduced_multipole_expansion.get_parts_by_tex(r"+")[0]
        self.play(
            Uncreate(indication_box2),
            Unwrite(plus),
            Unwrite(quadrupole_term),
            Unwrite(higher_order_terms),
            Unwrite(text_precession),
            Unwrite(text_satellite_orbit)
        )
        
        self.wait(4)
        
        
        self.play(Unwrite(potential_hamiltonian))
        self.play(Write(point_particle_interaction))
        self.wait(0.5)
        self.play(Create(interactions))
        
        self.wait(2)
        
        box_rotation = SurroundingRectangle(rotation_part)
        self.play(Create(box_rotation))
        
        self.wait(2)
        
        self.play(
            Uncreate(box_rotation),
            TransformMatchingTex(
                kinetic_hamiltonian,
                reduced_kinetic_hamiltonian,
                fade_transform_mismatches=True,
            )
        )
        
        self.wait(4)
        cover2 = FullScreenRectangle(color=BLACK)
        
        self.play(
            FadeIn(cover2)
        )
        self.wait(0.5)
        
class AnalyticApproaches(Scene):
    def construct(self):
        tex_font_size = 36
        text_font_size = 32
        hamiltonian = MathTex(
            "H", "=", r"\frac{\mathbf p_1}{2m_1}", "+",
            r"\frac{\mathbf p_2}{2m_2}", "+", r"\frac{\mathbf p_3}{2m_3}",
            r"-\frac{Gm_1m_2}{\|\mathbf x_1-\mathbf x_2\|}", r"-\frac{Gm_2m_3}{\|\mathbf x_2-\mathbf x_3\|}", r"-\frac{Gm_3m_1}{\|\mathbf x_3-\mathbf x_1\|}",
            font_size=tex_font_size
        ).move_to(2 * UP)
         
        initial_values = MathTex(
            r"(\mathbf x_1(0), \mathbf p_1(0))", ",",
            r"(\mathbf x_2(0), \mathbf p_2(0))", ",",
            r"(\mathbf x_3(0), \mathbf p_3(0))",
            font_size=tex_font_size
        ).next_to(hamiltonian, direction=DOWN, buff=0.75)
        solution = MathTex(
            r"(\mathbf x_1(t), \mathbf p_1(t))", ",",
            r"(\mathbf x_2(t), \mathbf p_2(t))", ",",
            r"(\mathbf x_3(t), \mathbf p_3(t))",
            font_size=tex_font_size
        ).next_to(initial_values, direction=DOWN, buff=2)
        arrow = Arrow(initial_values.get_bottom(), solution.get_top())
        question_marks = Text("???", font_size=32).next_to(arrow, direction=RIGHT, buff=0.6)
        X = VGroup(
            Line(UP+LEFT, DOWN+RIGHT), 
            Line(UP+RIGHT, DOWN+LEFT)
        ).scale(0.4).set_stroke_width(5).set_color(RED).move_to(arrow)
        no_closed_form_sol_text = Text(
            "\"No closed-form solution.\"",
            font_size=text_font_size
        ).next_to(solution, direction=DOWN, buff=0.75).shift(1 * LEFT)
        poincare_text = Text(
            "-Henri Poincare",
            font_size=24
        ).next_to(no_closed_form_sol_text, direction=RIGHT, aligned_edge=DOWN, buff=0.45).shift(0.15 * DOWN)
        
        self.wait(0.5)
        self.play(Write(hamiltonian))
        self.wait(1)
        
        self.play(FadeIn(initial_values))
        self.play(GrowArrow(arrow))
        self.play(FadeIn(solution))
        self.wait(1)
        self.play(Write(question_marks, run_time=1.5))
        
        self.wait(2)
        
        self.play(Create(X))
        self.play(Write(no_closed_form_sol_text))
        self.wait(0.75)
        self.play(Write(poincare_text))
         
        self.wait(2)
        
        self.play(
            Unwrite(initial_values, run_time=0.5),
            Unwrite(solution, run_time=0.5),
            Uncreate(X, run_time=0.5),
            Uncreate(arrow, run_time=0.5),
            Unwrite(question_marks, run_time=0.5),
            Unwrite(no_closed_form_sol_text, run_time=0.5),
            Unwrite(poincare_text, run_time=0.5),
        )
        puiseux_series_text = Text(
            "Puiseux Series", 
            font_size=32
        ).next_to(hamiltonian, direction=DOWN, buff=0.5)
        puiseux_series_solution_x = MathTex(
            r"\mathbf x_i(t)", "=",
            r"\mathbf x_i^{(0)}", 
            "+", r"\mathbf x_i^{(1)}", r"t^{1/3}",
            "+", r"\mathbf x_i^{(2)}", r"t^{2/3}",
            "+", r"\mathbf x_i^{(3)}", r"t",
            "+", r"\mathbf x_i^{(4)}", r"t^{4/3}",
            r"+\cdots", 
            font_size=tex_font_size
        ).next_to(puiseux_series_text, direction=DOWN, buff=0.75)
        puiseux_series_solution_p = MathTex(
            r"\mathbf p_i(t)", "=",
            r"\mathbf p_i^{(0)}", 
            "+", r"\mathbf p_i^{(1)}", r"t^{1/3}",
            "+", r"\mathbf p_i^{(2)}", r"t^{2/3}",
            "+", r"\mathbf p_i^{(3)}", r"t",
            "+", r"\mathbf p_i^{(4)}", r"t^{4/3}",
            r"+\cdots", 
            font_size=tex_font_size
        ).next_to(puiseux_series_solution_x, direction=DOWN, buff=0.6)
        sundman_text = Text(
            "-Karl Fritiof Sundman",
            font_size=24
        ).next_to(puiseux_series_solution_p, direction=DOWN, buff=0.8).shift(2 * RIGHT)
        self.play(
            Write(puiseux_series_text)
        )
        
        self.wait(1)
        
        
        self.play(
            Write(puiseux_series_solution_x),
            Write(puiseux_series_solution_p),
        )
        
        self.play(
            Write(sundman_text)
        )
        
        self.wait(3)
        
        
        coefficients = [
            puiseux_series_solution_x.get_part_by_tex(f"\mathbf x_i^{{({i})}}") 
            for i in range(5)
        ] + [
            puiseux_series_solution_p.get_part_by_tex(f"\mathbf p_i^{{({i})}}") 
            for i in range(5)
        ]
        
        
        self.play(
            *(
                Circumscribe(coefficient) for coefficient in coefficients
            )
        )
        
        self.wait(4)
        self.play(
            *(
                FadeOut(mobj) for mobj in self.mobjects
            )
        )
        
        
        self.wait(0.5)
    
class NumericalApproaches(Scene):
    def construct(self):
        tex_font_size = 36
        text_font_size = 32
        hamiltonian = MathTex(
            r"H", "=", r"\sum_{j=1}^{N}\frac{\mathbf{P}_j}{2m_j}", "-", r"\sum_{j<k}\frac{Gm_jm_k}{\|\mathbf x_i-\mathbf x_j\|}",
            font_size=tex_font_size
        ).move_to(2 * UP)
        ode_text = Tex(
            "6$N$-dimensional Ordinay Differential Equation",
            font_size=tex_font_size
        )
        
        self.wait(0.5)
        self.play(
            Write(hamiltonian)
        )
        self.wait(1)
        
        self.play(
            Write(ode_text)
        )
        self.wait(2)
        self.play(
            hamiltonian.animate.shift(0.85 * UP),
            FadeOut(ode_text)
        )
        
        self.wait(2)
    
class EularMethod(Scene):
    def construct(self):
        
        tex_font_size = 36
        text_font_size = 32
        hamiltonian_tex = MathTex(
            r"H", "=", r"\sum_{j=1}^{N}\frac{\mathbf{P}_j}{2m_j}", "-", r"\sum_{j<k}\frac{Gm_jm_k}{\|\mathbf x_i-\mathbf x_j\|}",
            font_size=tex_font_size
        ).move_to(2.85 * UP)
        
        
        
        G = 4
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
        dummy = Dot(fill_opacity = 0).move_to(0.5 * DOWN + 3 * LEFT)
        ref_length_tracker = ValueTracker(0.5)
        N = 2
        system = HamiltonianSystem(
            2 * N,
            kinetic=kinetic,
            potential=potential,
            initialvalues=(q0, p0),
            method="eular",
            simulation_step_size=5e-3
        )
        
        def updater(planet_or_moon:Circle, xtracker, ytracker):
            origin = dummy.get_center()
            ref_len = ref_length_tracker.get_value()
            x = xtracker.get_value()
            y = ytracker.get_value()
            planet_or_moon.move_to(origin + x * ref_len * RIGHT + y * ref_len * UP)

        earth.add_updater(
            functools.partial(
                updater,
                xtracker = system.qtrackers[0],
                ytracker = system.qtrackers[1]
            ),
            call_updater=True
        )
        moon.add_updater(
            functools.partial(
                updater,
                xtracker = system.qtrackers[2],
                ytracker = system.qtrackers[3]
            ),
            call_updater=True
        )
        system.freeze()
        self.add(hamiltonian_tex)
        
        moon_trace = TracedPath(moon.get_center, dissipating_time=4)
        self.add(moon_trace)
        self.add(earth, moon)
        self.add(system)
        cover = Rectangle(width=5, height=5, color=BLACK, fill_opacity=1).move_to(dummy)
        self.add(cover)

        eularmethod_text = Text(
            "Eular Method",
            font_size=text_font_size + 4
        ).move_to(1.5 * UP + 3 * RIGHT)
        
        eularmethod_formula_x = MathTex(
            r"\Delta\mathbf x=\frac{\partial H}{\partial\mathbf p}\Delta t",
            font_size=tex_font_size
        ).next_to(eularmethod_text, direction=DOWN, buff=0.65)
        eularmethod_formula_p = MathTex(
            r"\Delta\mathbf p=-\frac{\partial H}{\partial\mathbf x}\Delta t",
            font_size=tex_font_size
        ).next_to(eularmethod_formula_x, direction=DOWN, buff=0.5)
        
        
        self.wait(0.5)
        system.unfreeze()
        self.play(FadeOut(cover))
        print(system.h)
        self.wait(1)
        
        self.play(Write(eularmethod_text))
        self.wait(0.5)
        self.play(
            Write(eularmethod_formula_x),
            Write(eularmethod_formula_p),
        )
        
        self.wait(1.85)
        self.play(
            system.reference_time_tracker.animate.set_value(3)
        )
        self.wait(6)
        print(system.h)
        self.play(
            Unwrite(eularmethod_text),
            Unwrite(eularmethod_formula_x),
            Unwrite(eularmethod_formula_p),
        )
        self.wait(0.5)
        
        energy_text = Text(
            "Energy",
            font_size=text_font_size
        ).move_to(1.65 * UP + 3 * RIGHT)
        
        
        energy_plot_axis = Axes(
            x_range=[0, 60],
            y_range=[-4, -8],
            x_length=4.5,
            y_length=4,
            tips=False,
            axis_config={
                "include_ticks":False
            }
        ).move_to(3 * RIGHT + 0.65 * DOWN)
        ts = system.get_times()
        hs = system.get_energies()
        plot = energy_plot_axis.plot_line_graph(ts, hs, add_vertex_dots=False)
        cover2 = Rectangle(width=5, height=4.8, color=BLACK, fill_opacity=1).move_to(2.85 * RIGHT + 0.98 * DOWN)
        
        def plot_updater(plot:Axes):
            ts = system.get_times()
            hs = system.get_energies()
            new_plot = energy_plot_axis.plot_line_graph(
                ts, hs, add_vertex_dots=False
            )
            plot.become(new_plot)
        plot.add_updater(plot_updater)
        
        
        self.play(
            Write(energy_text),
            system.reference_time_tracker.animate.set_value(2)
        )
        self.add(
            energy_plot_axis,
            plot,
            cover2
        )
        self.play(FadeOut(cover2))
        
        
        self.wait(12)
        
        final_cover=FullScreenRectangle().set_opacity(1).set_color(BLACK).move_to(1.76 * DOWN)
        self.play(FadeIn(final_cover))
        self.wait(0.5)
        
class EularMethodFailure(Scene):
    def construct(self):
        
        tex_font_size = 36
        text_font_size = 32
        hamiltonian_tex = MathTex(
            r"H", "=", r"\sum_{j=1}^{N}\frac{\mathbf{P}_j}{2m_j}", "-", r"\sum_{j<k}\frac{Gm_jm_k}{\|\mathbf x_i-\mathbf x_j\|}",
            font_size=tex_font_size
        ).move_to(2.85 * UP)
        
        
        
        G = 4
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
        dummy = Dot(fill_opacity = 0).move_to(0.5 * DOWN + 3 * LEFT)
        ref_length_tracker = ValueTracker(0.5)
        N = 2
        system = HamiltonianSystem(
            2 * N,
            kinetic=kinetic,
            potential=potential,
            initialvalues=(q0, p0),
            method="eular",
            simulation_step_size=0.33
        )
        
        def updater(planet_or_moon:Circle, xtracker, ytracker):
            origin = dummy.get_center()
            ref_len = ref_length_tracker.get_value()
            x = xtracker.get_value()
            y = ytracker.get_value()
            planet_or_moon.move_to(origin + x * ref_len * RIGHT + y * ref_len * UP)

        earth_updater = functools.partial(
            updater,
            xtracker = system.qtrackers[0],
            ytracker = system.qtrackers[1]
        )
        moon_updater = functools.partial(
            updater,
            xtracker = system.qtrackers[2],
            ytracker = system.qtrackers[3]
        )
        earth.add_updater(
            earth_updater,
            call_updater=True
        )
        moon.add_updater(
            moon_updater,
            call_updater=True
        )
        system.freeze()
        self.add(hamiltonian_tex)
        exact_moon_orbit = Circle(radius=1.25, stroke_color=WHITE, fill_opacity=0, stroke_width=2).move_to(dummy)
        exact_earth_orbit = Circle(radius=0.125, stroke_color=WHITE, fill_opacity=0, stroke_width=2).move_to(dummy)
        
        moon_trace = TracedPath(moon.get_center, stroke_color=YELLOW, stroke_width=2)
        self.add(moon_trace)
        self.add(earth, moon)
        self.add(system)
        cover = Rectangle(width=5, height=5, color=BLACK, fill_opacity=1).move_to(dummy)
        # self.add(cover)

        eularmethod_text = Text(
            "Failure of Eular Method",
            font_size=text_font_size
        ).move_to(1.5 * UP + 3 * RIGHT)
        
        eularmethod_formula_x = MathTex(
            r"\Delta\mathbf x=\frac{\partial H}{\partial\mathbf p}\Delta t",
            font_size=tex_font_size
        ).next_to(eularmethod_text, direction=DOWN)
        eularmethod_formula_p = MathTex(
            r"\Delta\mathbf p=-\frac{\partial H}{\partial\mathbf x}\Delta t",
            font_size=tex_font_size
        ).next_to(eularmethod_formula_x, direction=DOWN)
        update_on_tangentspace1 = MathTex(
            r"(\Delta\mathbf x, \Delta\mathbf p)\cdot\nabla H=0",
            font_size=tex_font_size
        ).next_to(eularmethod_formula_p, direction=DOWN)
        update_on_tangentspace2 = MathTex(
            r"\mathbf Z(t)\in T_{\mathbf Z(t)}\text{(iso-H)}",
            font_size=tex_font_size
        ).next_to(update_on_tangentspace1, direction=DOWN)
        
        velocity_arrow = Arrow()
        def velocity_arrow_updater(arrow:Arrow):
            moon_pos = moon.get_center()
            velocity_vec = np.array([
                system.ptrackers[2].get_value(),
                system.ptrackers[3].get_value(),
                0
            ]) / m2 / 4
            arrow.become(Arrow(moon_pos, moon_pos+velocity_vec, color=GREEN, buff=0))
        velocity_arrow.add_updater(
            velocity_arrow_updater,
            call_updater=True
        )
        
        self.play(
            FadeIn(earth),
            FadeIn(moon),
            Write(eularmethod_text),
            Write(eularmethod_formula_x),
            Write(eularmethod_formula_p),
        )
        self.wait(2)
        
        self.play(
            Create(exact_moon_orbit),
            Create(exact_earth_orbit)
        )
        self.play(
            GrowArrow(velocity_arrow)
        )
        self.wait(2)
        
        def animate_step():
            old_moon_pos = moon.get_center()
            system.force_step()
            moon_updater(moon)
            earth_updater(earth)
            self.play(
                Flash(moon, line_length=0.06),
                Flash(old_moon_pos, reverse_rate_function = True, line_length=0.04, flash_radius=0)
            )
            
        animate_step()
        self.wait(2)
        self.play(
            Write(update_on_tangentspace1)
        )
        self.wait(1)
        self.play(
            Write(update_on_tangentspace2)
        )
        self.wait(2)
        
        delays = [0.65, 0.65, 0.08, 0.08, 0.08]
        
        for delay in delays:
            animate_step()
            self.wait(delay)
        
        self.wait(2)
        
        self.play(*(
            FadeOut(mobj) for mobj in self.mobjects
        ))
        self.wait(0.5)
        

class SympleticMethods(Scene):
    def construct(self):
        tex_font_size=36
        tex_font_size2=28
        text_font_size=32
        title = Text("Sympletic Integrator", font_size=text_font_size + 4).to_edge(UP)
        hamiltonian = MathTex(
            "H(q, p)", "=", "T(p)", "+", "V(q)",
            font_size=tex_font_size
        ).next_to(title, direction=DOWN)      
        new_var_and_op = MathTex(
            r"\mathbf{z}", "=", "(", r"\mathbf{x}", ",", r"\mathbf{p}", r")^\top,\quad",
            r"D_H\cdot", "=", r"\{\cdot, H\}",
            font_size=tex_font_size
        ).next_to(hamiltonian, direction=DOWN)  
        new_eom_and_sol = MathTex(
            r"\dot{\mathbf{z}}", "=", r"D_H\mathbf z", r",\quad", r"z(t+h)=\exp(hD_H)z(t)",
            font_size=tex_font_size
        ).next_to(new_var_and_op, direction=DOWN)
        expansion = MathTex(
            r"\exp(hD_H)=\exp[h(D_T+D_V)]=\prod_{j=1}^k\exp(c_j hD_T)\exp(d_j hD_V)+O(h^{k+1})",
            font_size=tex_font_size
        ).next_to(new_eom_and_sol, direction=DOWN)
        cdconstrains = Tex(
            r"with constrains: $\sum_{i=1}^k c_i=1,\quad\sum_{i=1}^k d_i=1$",
            font_size=tex_font_size
        ).next_to(expansion, direction=DOWN)
        taylor_approx = MathTex(
            r"\exp(X)\approx1+X",
            font_size=tex_font_size
        ).next_to(cdconstrains, direction=DOWN)
        symp_method_formula = MathTex(
            r"\mathbf p^{(i+1)}=\mathbf p^{(i)}-d_i\frac{\partial}{\partial\mathbf x}H(\mathbf q^{(i)})h", r", \quad",
            r"\mathbf q^{(i+1)}=\mathbf q^{(i)}-c_i\frac{\partial}{\partial\mathbf p}H(\mathbf p^{(i+1)})h",
            font_size=tex_font_size
        ).next_to(taylor_approx, direction=DOWN)
        
        p_updater_formula = MathTex(
            r"\mathbf p^{(i+1)}=\mathbf p^{(i)}-d_i\frac{\partial}{\partial\mathbf x}H(\mathbf q^{(i)})h",
            font_size=tex_font_size
        ).next_to(title, direction=DOWN).shift(3.5 * RIGHT)
        q_updater_formula = MathTex(
            r"\mathbf q^{(i+1)}=\mathbf q^{(i)}-c_i\frac{\partial}{\partial\mathbf p}H(\mathbf p^{(i+1)})h",
            font_size=tex_font_size
        ).next_to(p_updater_formula, direction=DOWN, aligned_edge=LEFT)  
        moved_updater_formula = VGroup(
            p_updater_formula, q_updater_formula
        )
        
        leapfrrg_text = Text("Leapfrog:", font_size=text_font_size).move_to(1.65 * UP)
        leapfrog_constants = VGroup(
            MathTex(r"c_1=1,\ c_2=0", font_size=tex_font_size2),
            MathTex(r"d_1=\frac{1}{2},\ d_2=\frac{1}{2}", font_size=tex_font_size2),
        ).arrange(DOWN).next_to(leapfrrg_text, DOWN)
        
        forest_ruth_text = Text("Forest-Ruth:", font_size=text_font_size).next_to(leapfrog_constants, DOWN)
        forest_ruth_constants = VGroup(
            MathTex(r"c_1=c_4=\frac{1}{2(2-2^{1/3})},\ c_2=c_3=\frac{1-2^{1/3}}{2(2-2^{1/3})}", font_size=tex_font_size2),
            MathTex(r"d_1=d_3=\frac{1}{2-2^{1/3}},\ d_2=-\frac{2^{1/3}}{2-2^{1/3}},\ d_4=0", font_size=tex_font_size2),
        ).arrange(DOWN).next_to(forest_ruth_text, DOWN)
        constants = VGroup(
            leapfrrg_text,
            leapfrog_constants,
            forest_ruth_text,
            forest_ruth_constants,
        )
        # symp_method_formula.next_to(title, direction=DOWN)
        # self.add(
        #     title,
        #     # hamiltonian,
        #     # new_var_and_op,
        #     # new_eom_and_sol,
        #     # expansion,
        #     # cdconstrains,
        #     # taylor_approx,
        #     symp_method_formula,
        #     # moved_updater_formula,
        #     leapfrrg_text,
        #     leapfrog_constants,
        #     forest_ruth_text,
        #     forest_ruth_constants
        # )
        
        self.wait(0.5)
        self.play(Write(title))
        self.wait(2)
        self.play(Write(hamiltonian))
        self.wait(2)
        self.play(Write(new_var_and_op))
        self.wait(2)
        self.play(Write(new_eom_and_sol))
        self.wait(2)
        self.play(Write(expansion))
        self.wait(2)
        self.play(Write(taylor_approx))
        self.wait(2)
        self.play(Write(symp_method_formula))
        self.wait(2)
        self.play(
            symp_method_formula.animate.next_to(title, direction=DOWN),
            FadeOut(hamiltonian),
            FadeOut(new_var_and_op),
            FadeOut(new_eom_and_sol),
            FadeOut(expansion),
            FadeOut(taylor_approx),
        )
        self.wait(2)
        self.play(
            LaggedStart(
                Write(leapfrrg_text),
                Write(leapfrog_constants),
                Write(forest_ruth_text),
                Write(forest_ruth_constants),
            )
        )
        self.wait(2)

        
        self.play(
            TransformMatchingTex(
                symp_method_formula, moved_updater_formula,
            ),
            constants.animate.shift(1.17 * DOWN + 3.5 * RIGHT)
        )
        
        self.wait(2)
        
    
class SympleticSim(Scene):
    def construct(self):
        tex_font_size=36
        tex_font_size2=28
        text_font_size=32
        title = Text("Sympletic Integrator", font_size=text_font_size + 4).to_edge(UP)
        leapfrrg_text = Text("Leapfrog:", font_size=text_font_size).move_to(1.65 * UP)
        leapfrog_constants = VGroup(
            MathTex(r"c_1=1,\ c_2=0", font_size=tex_font_size2),
            MathTex(r"d_1=\frac{1}{2},\ d_2=\frac{1}{2}", font_size=tex_font_size2),
        ).arrange(DOWN).next_to(leapfrrg_text, DOWN)
        p_updater_formula = MathTex(
            r"\mathbf p^{(i+1)}=\mathbf p^{(i)}-d_i\frac{\partial}{\partial\mathbf x}H(\mathbf q^{(i)})h",
            font_size=tex_font_size
        ).next_to(title, direction=DOWN).shift(3.5 * RIGHT)
        q_updater_formula = MathTex(
            r"\mathbf q^{(i+1)}=\mathbf q^{(i)}-c_i\frac{\partial}{\partial\mathbf p}H(\mathbf p^{(i+1)})h",
            font_size=tex_font_size
        ).next_to(p_updater_formula, direction=DOWN, aligned_edge=LEFT)  
        moved_updater_formula = VGroup(
            p_updater_formula, q_updater_formula
        )
        forest_ruth_text = Text("Forest-Ruth:", font_size=text_font_size).next_to(leapfrog_constants, DOWN)
        forest_ruth_constants = VGroup(
            MathTex(r"c_1=c_4=\frac{1}{2(2-2^{1/3})},\ c_2=c_3=\frac{1-2^{1/3}}{2(2-2^{1/3})}", font_size=tex_font_size2),
            MathTex(r"d_1=d_3=\frac{1}{2-2^{1/3}},\ d_2=-\frac{2^{1/3}}{2-2^{1/3}},\ d_4=0", font_size=tex_font_size2),
        ).arrange(DOWN).next_to(forest_ruth_text, DOWN)
        constants = VGroup(
            leapfrrg_text,
            leapfrog_constants,
            forest_ruth_text,
            forest_ruth_constants,
        ).shift(1.17 * DOWN + 3.5 * RIGHT)
        self.add(title, moved_updater_formula, constants)
        
        G = 4
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
        dummy = Dot(fill_opacity = 0).move_to(0.15 * UP + 3 * LEFT)
        ref_length_tracker = ValueTracker(0.75)
        N = 2
        system = HamiltonianSystem(
            2 * N,
            kinetic=kinetic,
            potential=potential,
            initialvalues=(q0, p0),
            method="forest-ruth",
            simulation_step_size=5e-5
        )
        
        def updater(planet_or_moon:Circle, xtracker, ytracker):
            origin = dummy.get_center()
            ref_len = ref_length_tracker.get_value()
            x = xtracker.get_value()
            y = ytracker.get_value()
            planet_or_moon.move_to(origin + x * ref_len * RIGHT + y * ref_len * UP)

        earth.add_updater(
            functools.partial(
                updater,
                xtracker = system.qtrackers[0],
                ytracker = system.qtrackers[1]
            ),
            call_updater=True
        )
        moon.add_updater(
            functools.partial(
                updater,
                xtracker = system.qtrackers[2],
                ytracker = system.qtrackers[3]
            ),
            call_updater=True
        )
        system.freeze()
        
        moon_trace = TracedPath(moon.get_center, dissipating_time=4)
        self.add(moon_trace)
        self.add(earth, moon)
        self.add(system)
        cover = Rectangle(width=5, height=5, color=BLACK, fill_opacity=1).move_to(dummy)
        self.add(cover)

        self.wait(0.5)
        system.unfreeze()
        self.play(FadeOut(cover))
        self.wait(5)
        
        acc4times_text = MathTex(r"\triangleright\triangleright\times 4", font_size=42).move_to(2.95 * DOWN + 3 * LEFT)
        self.play(
            system.reference_time_tracker.animate.set_value(4),
            FadeIn(acc4times_text)
        )
        self.wait(10)
        self.play(
            system.reference_time_tracker.animate.set_value(1),
            FadeOut(acc4times_text)
        )
        self.wait(3)
        fullcover = FullScreenRectangle().set_color(BLACK).set_opacity(1)
        self.play(FadeIn(fullcover))
        self.wait(0.5) 
        
        
        
# class OptimizationMethods(Scene):
#     def construct(self):
#         pass
    
# class TreeCode(Scene):
#     def construct(self):
#         pass
    
# class ParticleMesh(Scene):
#     def construct(self):
#         pass
    
# class Softening(Scene):
#     def construct(self):
#         pass
    
class Summary(Scene):
    def construct(self):
        width = 4.5
        height = width * 9 / 16
        
        
        text_summary = Text("Summary").to_edge(UP)
        rect_kwargs = {
            "width":width,
            "height":height,
            "fill_color":BLACK,
            "stroke_opacity":0.5
        }
        text_kwargs = {
            "font_size":26,
            "fill_opacity":0.5,
            "opacity":0
        }
        rects = []
        rects.append(Rectangle(stroke_color = GOLD, **rect_kwargs).move_to(3 * LEFT + 1.5 * UP))
        rects.append(Rectangle(stroke_color = GREEN, **rect_kwargs).move_to(3 * RIGHT + 1.5 * UP))
        rects.append(Rectangle(stroke_color = TEAL, **rect_kwargs).move_to(3 * LEFT + 1.8 * DOWN))
        rects.append(Rectangle(stroke_color = BLUE, **rect_kwargs).move_to(3 * RIGHT + 1.8 * DOWN))
        
        texts = []
        texts.append(Paragraph("Many body problems", **text_kwargs))
        texts.append(Paragraph("Equation of Motion of\nCelestial Mechanics", **text_kwargs))
        texts.append(Paragraph("Analytic Approaches", **text_kwargs))
        texts.append(Paragraph("Numerical Approaches\nand Symplectic Integrator", **text_kwargs))
        
        for text, rect in zip(texts, rects):
            text.next_to(rect, direction=DOWN, buff=0.05)
        
        self.play(Write(text_summary))
        self.play(LaggedStart(
            LaggedStart(
                *[Create(mob) for mob in rects],
                lag_ratio=0.4,
                run_time=3
            ),
            LaggedStart(
                *[Write(mob) for mob in texts],
                lag_ratio=0.4,
                run_time=3
            ),
            lag_ratio=0.25,
            run_time=4
        ))
        self.wait(1)
        
        for text, rect in zip(texts, rects):
            self.play(
                text.animate.set_opacity(1),
                rect.animate.set_opacity(1)
            )
            self.wait(3)
            self.play(
                text.animate.set_opacity(0.5),
                rect.animate.set_opacity(0.5)
            )
        
        self.play(LaggedStart(
            *[VGroup(text, rect).animate.set_opacity(1) for text, rect in zip(texts, rects)],
            lag_ratio=0.25,
            run_time=2
        ))
        
        self.wait(5)
        # fade out all
        self.play(*[FadeOut(mob) for mob in self.get_mobject_family_members()])
