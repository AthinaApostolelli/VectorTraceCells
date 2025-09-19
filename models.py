import jax
import brainpy as bp
import brainpy.math as bm
import numpy as np

class PC_cell(bp.DynamicalSystem):
    def __init__(
        self,
        num=100,
        tau=10.0,
        tauv=100.0,
        m0=10.0,
        k=1.0,
        a=0.08,
        sigma_x=0.05,
        A=10.0,
        J0=4.0,
        z_min=0,
        z_max=1,
    ):

        super(PC_cell, self).__init__()

        # Hyper-parameters
        self.num = num  # number of neurons at each dimension
        self.tau = tau  # synaptic time constant
        self.tauv = tauv  # time constant of firing rate adaptation
        self.m = tau / tauv * m0  # adaptation strength
        self.k = k  # degree of the rescaled inhibition
        self.a = a  # half-width of the range of excitatory connections
        self.sigma_x = sigma_x  # radial extent of tuning 
        self.A = A  # magnitude of the external input
        self.J0 = J0  # maximum connection value

        # Feature space
        self.z_range = z_max - z_min
        linspace_z = bm.linspace(z_min, z_max, num + 1)
        self.z = linspace_z[:-1]
        x, y = bm.meshgrid(self.z, self.z)  # x y index
        self.value_index = bm.stack([x.flatten(), y.flatten()]).T

        # Synaptic connections
        self.conn_mat = self.make_conn()
        
        # Initialize dynamical variables
        self.r = bm.Variable(bm.zeros((num, num)))  # firing rate of all PCs
        self.u = bm.Variable(bm.zeros((num, num)))  # presynaptic input of all PCs
        self.v = bm.Variable(bm.zeros((num, num)))  # firing rate adaptation of all PCs
        self.center = bm.Variable(bm.zeros(2))  # center of the bump
        self.loc_input = bm.Variable(bm.zeros((num, num)))  # location-dependent sensory input

        # Define the integrator
        self.integral = bp.odeint(method="exp_euler", f=self.derivative)

    @property
    def derivative(self):
        du = lambda u, t, Irec: (-u + Irec + self.total_input - self.v) / self.tau
        dv = lambda v, t: (-v + self.m * self.u) / self.tauv
        return bp.JointEq([du, dv])

    def dist(self, d):
        # Periodic wrap around - GC connectivity
        v_size = bm.asarray([self.z_range, self.z_range])
        return bm.where(d > v_size / 2, v_size - d, d)

    def make_conn(self):
        @jax.vmap
        def get_J(v):
            # d = self.dist(bm.abs(v - self.value_index)) # GC connectivity
            d = bm.abs(v - self.value_index) # 2D CAN connectivity
            d = bm.linalg.norm(d, axis=1)
            Jxx = (
                self.J0
                * bm.exp(-0.5 * bm.square(d / self.a))
                / (bm.sqrt(2 * bm.pi) * self.a)
            )
            return Jxx

        return get_J(self.value_index)

    def location_input(self, Animal_location, ThetaModulator):
        # return bump input (same dim as neuronal space) from a x-y location

        assert bm.size(Animal_location) == 2

        # d = self.dist(bm.abs(bm.asarray(Animal_location) - self.value_index)) # GC connectivity
        d = bm.abs(bm.asarray(Animal_location) - self.value_index) # 2D CAN connectivity
        d = bm.linalg.norm(d, axis=1)
        d = d.reshape((self.num, self.num))

        # Gaussian bump input
        loc_input = self.A * bm.exp(-0.25 * bm.square(d / self.sigma_x))

        # Theta modulation
        loc_input = loc_input * ThetaModulator

        return loc_input

    def update(self, Animal_location, ThetaModulator):

        # Total input
        self.loc_input = self.location_input(Animal_location, ThetaModulator)
        self.total_input = self.loc_input 

        # Recurrent connections
        Irec = bm.matmul(self.conn_mat, self.r.flatten()).reshape((self.num, self.num))

        # Update the system
        u, v = self.integral(self.u, self.v, None, Irec)

        self.u.value = bm.where(u > 0, u, 0)
        self.v.value = v
        r1 = bm.square(self.u)
        r2 = 1.0 + self.k * bm.sum(r1)
        self.r.value = r1 / r2
        
        # Get the center of the bump from self.r which is num x num matrix
        self.center[1], self.center[0] = bm.unravel_index(bm.argmax(self.r.value), [self.num, self.num])


class BV_cell(bp.DynamicalSystem):
    def __init__(
        self,
        num_r=10,
        num_a=36,
        tau=10.0,
        tau_v=100.0,
        mbar=75.0,
        A=1.0,
        beta=0.183, 
        sigma0=0.0122,
        sigma_a=0.2,
        J0=5.0,
        k=1,
        g = 1000,
        x_min=0.,
        x_max=1.,
        z_min=-bm.pi,
        z_max=bm.pi
    ):
        super(BV_cell, self).__init__()

        # dynamics parameters
        self.tau = tau  # synaptic time constant
        self.tau_v = tau_v  # time constant of the adaptation variable
        self.num_r = num_r  # number of excitatory neurons for radial dimension
        self.num_a = num_a  # number of excitatory neurons for angular dimension
        self.num = self.num_r * self.num_a 
        self.k = k  # degree of the rescaled inhibition
        self.A = A  # magnitude of the external input
        self.g = g
        self.J0 = J0/g  # maximum connection value
        self.m = mbar * tau / tau_v
        self.beta = beta  # controls tuning's radial extent with distance
        self.sigma0 = sigma0  # tuning's radial extent at 0 distance
        self.sigma_a = sigma_a  # tuning's angular extent

        # feature space - radial
        self.x_min = x_min
        self.x_max = x_max
        self.x_range = x_max - x_min
        linspace_x = bm.linspace(x_min, x_max, num_r + 1)  # The encoded feature values
        self.value_radial = linspace_x[0:-1]

        # feature space - angular
        self.z_min = z_min
        self.z_max = z_max
        self.z_range = z_max - z_min
        linspace_z = bm.linspace(z_min, z_max, num_a + 1)  # The encoded feature values
        self.value_angle = linspace_z[0:-1]

        # Initialize dynamical variables
        self.r = bm.Variable(bm.zeros(self.num))
        self.u = bm.Variable(bm.zeros(self.num))
        self.v = bm.Variable(bm.zeros(self.num))
        self.input = bm.Variable(bm.zeros(self.num))
        self.center_I = bm.Variable(bm.zeros(2))
        self.center = bm.Variable(bm.zeros(2))

        # Define the integrator
        self.integral = bp.odeint(method="exp_euler", f=self.derivative)

    def circle_period(self, d):
        d = bm.where(d > bm.pi, d - 2 * bm.pi, d)
        d = bm.where(d < -bm.pi, d + 2 * bm.pi, d)
        return d

    def dist(self, d):
        dis = self.circle_period(d)
        delta_x = dis[:, 0]
        delta_y = dis[:, 1]
        dis = bm.sqrt(delta_x ** 2 + delta_y ** 2)
        return dis
    
    def radial_input(self, Boundary_distance):

        d = bm.abs(bm.asarray(Boundary_distance) - self.value_radial)

        # Gaussian bump input
        self.sigma_r = (d / self.beta + 1) * self.sigma0
        radial_input = self.A * bm.exp(-0.25 * bm.square(d / self.sigma_r)) # TODO or 0.5?

        return radial_input
    
    def angular_input(self, Boundary_angle):
        
        d = self.circle_period(bm.abs(bm.asarray(Boundary_angle) - self.value_angle))
        
        # Gaussian bump input
        angular_input = self.A * bm.exp(-0.25 * bm.square(d / self.sigma_a)) # TODO or 0.5?

        return angular_input
    
    @property
    def derivative(self):
        du = (
            lambda u, t, input: (
                -u
                + input
                - self.v
            )
            / self.tau
        )
        dv = lambda v, t: (-v + self.m * self.u) / self.tau_v
        return bp.JointEq([du, dv])

    def reset_state(self):
        self.r.value = bm.Variable(bm.zeros(self.num))
        self.u.value = bm.Variable(bm.zeros(self.num))
        self.v.value = bm.Variable(bm.zeros(self.num))
        self.input.value = bm.Variable(bm.zeros(self.num))

    def update(self, Boundary_distance, Boundary_angle):
        
        # Total external input
        boundary_segments = 360  # Number of segments for boundary discretisation
        delta_theta = 2 * np.pi / boundary_segments

        # Compute all radial and angular inputs at once
        radial = bm.stack([self.radial_input(d) for d in Boundary_distance])    # (N, num_r)
        angular = bm.stack([self.angular_input(theta) for theta in Boundary_angle])  # (N, num_a)

        # Now compute outer products in one go
        inputs = radial[:, :, None] * angular[:, None, :]   # all outer products
        inputs = inputs * delta_theta

        # Sum over boundaries
        total_input = inputs.sum(axis=0).reshape(-1)   # flatten into (num_r * num_a,)
        self.input = total_input
        
        # Update neural state
        u, v = self.integral(self.u, self.v, bp.share["t"], self.input, bm.dt) 
        self.u.value = bm.where(u > 0, u, 0)
        self.v.value = v
        r1 = bm.square(self.u)
        r2 = 1.0 + self.k * bm.sum(r1)
        self.r.value = self.g * r1 / r2


class PC_cell_topdown(bp.DynamicalSystem):
    def __init__(
        self,
        num=100,
        tau=10.0,
        tauv=100.0,
        m0=10.0,
        k=1.0,
        a=0.08,
        sigma_x=0.05,
        A=10.0,
        J0=4.0,
        z_min=0,
        z_max=1,
        goal_a=0.08, 
        goal_A=1.,
        goal_J0=500.,
        goal_loc=None,
        topdown=True,
    ):

        super(PC_cell_topdown, self).__init__()

        # Hyper-parameters
        self.num = num  # number of neurons at each dimension
        self.tau = tau  # synaptic time constant
        self.tauv = tauv  # time constant of firing rate adaptation
        self.m = tau / tauv * m0  # adaptation strength
        self.k = k  # degree of the rescaled inhibition
        self.a = a  # half-width of the range of excitatory connections
        self.sigma_x = sigma_x  # radial extent of tuning 
        self.A = A  # magnitude of the external input
        self.J0 = J0  # maximum connection value
        self.goal_J0 = goal_J0  # maximum goal connection value
        self.goal_a = goal_a  # half-width of the range of top down connections from the reward cell
        self.goal_A = goal_A  # magnitude of the top down input
        self.goal_loc = goal_loc  # goal location
        self.topdown = topdown  # whether to turn on top down input

        # Feature space
        self.z_range = z_max - z_min
        linspace_z = bm.linspace(z_min, z_max, num + 1)
        self.z = linspace_z[:-1]
        x, y = bm.meshgrid(self.z, self.z)  # x y index
        self.value_index = bm.stack([x.flatten(), y.flatten()]).T

        # Synaptic connections
        self.conn_mat = self.make_conn()
        
        if goal_loc is not None:
            self.goal_loc = bm.array(goal_loc).reshape(1,2)
            self.gd_conn = self.make_gd_conn(self.goal_loc)  
            self.conn_mat = self.conn_mat + self.gd_conn

        # Define variables we want to update
        self.r = bm.Variable(bm.zeros((num, num)))  # firing rate of all PCs
        self.u = bm.Variable(bm.zeros((num, num)))  # presynaptic input of all PCs
        self.v = bm.Variable(bm.zeros((num, num)))  # firing rate adaptation of all PCs
        self.center = bm.Variable(bm.zeros(2))  # center of the bump
        self.loc_input = bm.Variable(bm.zeros((num, num)))  # location-dependent sensory input
        self.td_input = bm.Variable(bm.zeros((num, num))) # top down goal location input to the networks

        # Define the integrator
        self.integral = bp.odeint(method="exp_euler", f=self.derivative)

    @property
    def derivative(self):
        du = lambda u, t, Irec: (-u + Irec + self.loc_input - self.v) / self.tau
        dv = lambda v, t: (-v + self.m * self.u) / self.tauv
        return bp.JointEq([du, dv])

    def dist(self, d):
        # Periodic wrap around - GC connectivity
        v_size = bm.asarray([self.z_range, self.z_range])
        return bm.where(d > v_size / 2, v_size - d, d)

    def make_conn(self):
        @jax.vmap
        def get_J(v):
            # d = self.dist(bm.abs(v - self.value_index)) # GC connectivity
            d = bm.abs(v - self.value_index) # 2D CAN connectivity
            d = bm.linalg.norm(d, axis=1)
            # d = d.reshape((self.length, self.length))
            Jxx = (
                self.J0
                * bm.exp(-0.5 * bm.square(d / self.a))
                / (bm.sqrt(2 * bm.pi) * self.a)
            )
            return Jxx

        return get_J(self.value_index)

    def make_gd_conn(self, goal_loc):
        # Add a goal directed connection to the neurons at the goal location with a Gaussian profile
        @jax.vmap
        def get_J(v):
            d = self.dist(bm.abs(v - goal_loc))
            d = bm.linalg.norm(d, axis=1)
            Jxx = (
                self.goal_J0
                * bm.exp(-0.5 * bm.square(d / self.goal_a))
                / (bm.sqrt(2 * bm.pi) * self.goal_a)
            )
            return Jxx
        
        conn_vec = get_J(self.value_index)
        
        '''
        #asymmetric connection of a neuron
        #find the cloest index in self.value_grid to the goal location
        distances = bm.linalg.norm(self.goal_loc - self.value_index, axis=1)
        closest_index = bm.argmin(distances)
        
        #geterante a zeros matrix the same size as self.conn_mat, and out only the closest_index column as 
        goal_conn_mat = bm.zeros_like(self.conn_mat)
        goal_conn_mat[:,closest_index] = conn_vec.reshape(-1,)
        '''
        
        # asymmetric connection of multiple neurons
        # d = self.dist(bm.abs(self.goal_loc - self.value_index))
        d = bm.abs(self.goal_loc - self.value_index)
        distances = bm.linalg.norm(d, axis=1)
        # rank the distances from low to high and get the index
        closest_index = bm.argsort(distances)
        goal_conn_mat = bm.zeros_like(self.conn_mat)
        for i, index in enumerate(closest_index):
            if True:
                rank_dist = distances[index]
                alpha = 1 - rank_dist / (bm.max(distances) - bm.min(distances))
                goal_conn_mat[:,index] = conn_vec.reshape(-1,) * alpha 
        
        return goal_conn_mat

    def location_input(self, Animal_location, ThetaModulator):
        # return bump input (same dim as neuronal space) from a x-y location

        assert bm.size(Animal_location) == 2

        # d = self.dist(bm.abs(bm.asarray(Animal_location) - self.value_index)) # GC connectivity
        d = bm.abs(bm.asarray(Animal_location) - self.value_index) # 2D CAN connectivity
        d = bm.linalg.norm(d, axis=1)
        d = d.reshape((self.num, self.num))

        # Gaussian bump input
        loc_input = self.A * bm.exp(-0.25 * bm.square(d / self.sigma_x))

        # Theta modulation
        loc_input = loc_input * ThetaModulator

        return loc_input

    def topdown_input(self, Topdown_mod):
        # return bump input (same dim as neuronal space) from a x-y location
        assert bm.size(self.goal_loc) == 2

        # d = self.dist(bm.abs(bm.asarray(self.goal_loc) - self.value_index))
        d = bm.abs(bm.asarray(self.goal_loc) - self.value_index)
        d = bm.linalg.norm(d, axis=1)
        d = d.reshape((self.num, self.num))

        # Gaussian bump input
        td_input = self.goal_A * bm.exp(-0.25 * bm.square(d / self.goal_a))

        # further theta modulation
        td_input = td_input * Topdown_mod

        return td_input
    
    def update(self, Animal_location, ThetaModulator, Topdown_mod):

        # Total input
        self.loc_input = self.location_input(Animal_location, ThetaModulator)

        if self.topdown: # top down control
            self.td_input = self.topdown_input(Topdown_mod)
            self.total_input = self.loc_input + self.td_input
        else:
            self.total_input = self.loc_input
        
        self.td_input = self.topdown_input(Topdown_mod)
        
        # Recurrent connections
        Irec = bm.matmul(self.conn_mat, self.r.flatten()).reshape((self.num, self.num))

        # Update the system
        u, v = self.integral(self.u, self.v, None, Irec)

        self.u.value = bm.where(u > 0, u, 0)
        self.v.value = v
        r1 = bm.square(self.u)
        r2 = 1.0 + self.k * bm.sum(r1)
        self.r.value = r1 / r2
        
        # Get the center of the bump from self.r which is num x num matrix
        self.center[1], self.center[0] = bm.unravel_index(bm.argmax(self.r.value), [self.num, self.num])
