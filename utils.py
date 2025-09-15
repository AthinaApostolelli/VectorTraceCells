from scipy.ndimage import gaussian_filter
import numpy as np 
import ratinabox
from ratinabox.Environment import Environment
from ratinabox.Agent import Agent
from ratinabox.Neurons import *
from tqdm.notebook import tqdm  # gives time bar

def vector_intercepts(vector_list_a, vector_list_b):
    """
    Adapted from ratinabox/utils

    Each element of vector_list_a gives a line segment of the form [[x_a_0,y_a_0],[x_a_1,y_a_1]], or, 
    in vector notation [p_a_0,p_a_1] (same goes for vector vector_list_b). Thus
        vector_list_A.shape = (N_a,2,2)
        vector_list_B.shape = (N_b,2,2)
    where N_a is the number of vectors defined in vector_list_a

    Each line segment defines an (infinite) line, parameterised by line_a = p_a_0 + l_a.(p_a_1-p_a_0).
    
    We want to find the intersection between these lines in terms of the parameters l_a and l_b.
    If l_a and l_b are BOTH between 0 and 1 then the line segments intersect. 
    Thus the goal is to return an array, I, of shape
        I.shape = (N_a,N_b,2)
    where, if I[n_a,n_b][0] and I[n_a,n_b][1] are both between 0 and 1 then it means line segments 
    vector_list_a[n_a] and vector_list_b[n_b] intersect.

    To do this we consider solving the equation line_a = line_b. The solution to this is:
        l_a = dot((p_b_0 - p_a_0) , (p_b_1 - p_b_0)_p) / dot((p_a_1 - p_a_0) , (p_b_1 - p_b_0)_p)
        l_b = dot((p_a_0 - p_b_0) , (p_a_1 - p_a_0)_p) / dot((p_b_1 - p_b_0) , (p_a_1 - p_a_0)_p)
    where "_p" denotes the perpendicular (in two-D [x,y]_p = [-y,x]). Using notation
        l_a = dot(d0,sb_p) / dot(sa,sb_p)
        l_b = dot(-d0,sa_p) / dot(sb,sa_p)
    for
        d0 = p_b_0 - p_a_0
        sa = p_a_1 - p_a_0
        sb = p_b_1 - p_b_0
    We will calculate these first.

    """
    assert (vector_list_a.shape[-2:] == (2, 2)) and (
        vector_list_b.shape[-2:] == (2, 2)
    ), "vector_list_a and vector_list_b must be shape (_,2,2), _ is optional"
    vector_list_a = vector_list_a.reshape(-1, 2, 2)
    vector_list_b = vector_list_b.reshape(-1, 2, 2)
    vector_list_a = vector_list_a + np.random.normal(
        scale=1e-9, size=vector_list_a.shape
    )
    vector_list_b = vector_list_b + np.random.normal(
        scale=1e-9, size=vector_list_b.shape
    )

    N_a = vector_list_a.shape[0]
    N_b = vector_list_b.shape[0]

    d0 = np.expand_dims(vector_list_b[:, 0, :], axis=0) - np.expand_dims(
        vector_list_a[:, 0, :], axis=1
    )  # d0.shape = (N_a,N_b,2)
    sa = vector_list_a[:, 1, :] - vector_list_a[:, 0, :]  # sa.shape = (N_a,2)
    sb = vector_list_b[:, 1, :] - vector_list_b[:, 0, :]  # sb.shape = (N_b,2)
    sa_p = np.flip(sa.copy(), axis=1)
    sa_p[:, 0] = -sa_p[:, 0]  # sa_p.shape = (N_a,2)
    sb_p = np.flip(sb.copy(), axis=1)
    sb_p[:, 0] = -sb_p[:, 0]  # sb.shape = (N_b,2)

    """
    Now we can go ahead and solve for the line segments since d0 has shape (N_a,N_b,2) in order to 
    perform the dot product we must first reshape sa (etc.) by tiling to shape (N_a,N_b,2).
    """
    sa = np.tile(np.expand_dims(sa, axis=1), reps=(1, N_b, 1))  # sa.shape = (N_a,N_b,2)
    sb = np.tile(np.expand_dims(sb, axis=0), reps=(N_a, 1, 1))  # sb.shape = (N_a,N_b,2)
    sa_p = np.tile(
        np.expand_dims(sa_p, axis=1), reps=(1, N_b, 1)
    )  # sa.shape = (N_a,N_b,2)
    sb_p = np.tile(
        np.expand_dims(sb_p, axis=0), reps=(N_a, 1, 1)
    )  # sb.shape = (N_a,N_b,2)

    """
    The dot product can now be performed by broadcast multiplying the arrays then summing over the 
    last axis.
    """
    l_a = (d0 * sb_p).sum(axis=-1) / (sa * sb_p).sum(axis=-1)  # la.shape=(N_a,N_b)
    l_b = (-d0 * sa_p).sum(axis=-1) / (sb * sa_p).sum(axis=-1)  # la.shape=(N_a,N_b)

    intercepts = np.stack((l_a, l_b), axis=-1)
    
    return intercepts


def rotate(vector, theta):
    """
    Adapted from ratinabox/utils
    
    Rotates a vector shape (2,) anticlockwise by angle theta.
    Args:
        vector (array): the 2d vector
        theta (flaot): the rotation angle, radians
    """
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    vector_new = np.matmul(R, vector)
    return vector_new


def simulate_agent_in_environment(duration=300, dt=50e-3, speed_mean=1.0, start_pos=(0.5, 0.5), addwall=False, wall=np.array([[0.4, 0.2], [0.4, 0.6]])):
    """
    Simulate an agent in a square environment and return the trajectory and head direction.

    Parameters
    ----------
    duration : float
        Total simulation time in seconds.
    dt : float
        Time step in seconds.
    speed_mean : float
        Mean speed of the agent.
    start_pos : tuple
        Starting position (x, y) of the agent.

    Returns
    -------
    t : (n,) numpy array
        Time stamps of each sample.
    Position : (n, 2) numpy array
        Agent's x and y positions at each sample.
    HD_angle : (n,) numpy array
        Agent's allocentric head direction angles (in radians).
    """

    # 1. Initialise environment
    Env = Environment(params={"aspect": 1, "scale": 1})
    if addwall:
        Env.add_wall(wall)

    # 2. Add agent
    Ag = Agent(Env)
    Ag.pos = np.array(start_pos)
    Ag.speed_mean = speed_mean

    # 3. Simulate
    n_steps = int(duration / dt)
    for _ in tqdm(range(n_steps)):
        Ag.update(dt=dt)

    # 4. Collect history
    t = np.array(Ag.history['t'])
    Position = np.array(Ag.history['pos'])
    HeadDirection = np.array(Ag.history['head_direction'])
    Velocity = np.array(Ag.history['vel'])        # (n_steps, 2)
    Speed = np.linalg.norm(Velocity, axis=1) 

    # 5. Compute allocentric head direction angle
    complex_HD = HeadDirection[:, 0] + 1j * HeadDirection[:, 1]
    HD_angle = np.angle(complex_HD)
    
    HD_angle = (HD_angle + 2*np.pi) % (2 * np.pi)

    return t, Position, HD_angle, Speed, Env


def BVC_response(theta, r, phi_i, d_i, sigma_ang, beta, sigma0, addnoise=False, noise_level=0.1):
    """
    Computes the firing rate of a Boundary Vector Cell (BVC) in response to a boundary located
    at polar coordinates (r, theta), relative to the agent's current position.

    The response is modelled as a product of a radial and an angular Gaussian tuning function,
    following the formulation in Hartley et al. (2000, Hippocampus).

    Parameters
    ----------
    theta : float or np.ndarray
        Allocentric angle to the boundary (in radians).
    r : float or np.ndarray
        Distance to the boundary (in metres).
    phi_i : float
        Preferred direction of the BVC (in radians).
    d_i : float
        Preferred distance of the BVC (in metres).
    sigma_ang : float
        Angular tuning width (in radians).
    beta : float
        Scaling factor controlling the linear increase of radial tuning width with d_i.
    sigma0 : float
        Base radial tuning width at zero distance (in metres).
    addnoise : bool
        If True, multiplicative Gaussian noise is applied.
    noise_level : float
        Standard deviation of multiplicative noise (as a proportion of signal amplitude).

    Returns
    -------
    response : float or np.ndarray
        The BVC's firing rate at the specified (theta, r) location.
    """
    sigma_rad = (d_i / beta + 1) * sigma0
    radial_component = np.exp(-((r - d_i) ** 2) / (2 * sigma_rad ** 2)) / (np.sqrt(2 * np.pi) * sigma_rad)
    angular_diff = np.angle(np.exp(1j * (theta - phi_i)))
    angular_component = np.exp(-(angular_diff ** 2) / (2 * sigma_ang ** 2)) / (np.sqrt(2 * np.pi) * sigma_ang)

    response = radial_component * angular_component

    if addnoise:
        gain_noise = np.random.normal(loc=1.0, scale=noise_level, size=response.shape)
        response = response * gain_noise
        response = np.clip(response, 0, None)  # ensure non-negative

    return response


def compute_BVC_firing_rate(Position, boundary_angles, delta_theta, phi_i, d_i, sigma_ang, beta, sigma0, addnoise=False):
        
    # Firing rate calculation along trajectory
    firing_rates = []

    for pos in tqdm(Position):
        x, y = pos
        fr = 0

        # For each boundary angle, find boundary location and calculate contribution
        for theta in boundary_angles:
            
            # Cast ray from agent at angle theta, find distance to wall (simple box intersection)

            # Convert theta to vector
            dx = np.cos(theta)
            dy = np.sin(theta)

            # Calculate intersection distances
            distances = []
            if dx != 0:
                t1 = (0 - x) / dx
                t2 = (1 - x) / dx
                if t1 > 0:
                    distances.append(t1)
                if t2 > 0:
                    distances.append(t2)
            if dy != 0:
                t3 = (0 - y) / dy
                t4 = (1 - y) / dy
                if t3 > 0:
                    distances.append(t3)
                if t4 > 0:
                    distances.append(t4)

            if len(distances) == 0:
                continue

            r = min(distances)  # shortest distance to boundary

            # Evaluate BVC response
            g = BVC_response(theta, r, phi_i, d_i, sigma_ang, beta, sigma0, addnoise=addnoise, noise_level=5)

            # Contribution proportional to Δθ
            fr += g * delta_theta

        firing_rates.append(fr)

    firing_rates = np.array(firing_rates)

    return firing_rates


def PC_response(r, d_i, beta, sigma0, addnoise=False, noise_level=0.1):
    """
    Computes the firing rate of a Place Cell (PC) in response to the agent's current position.

    Parameters
    ----------
    r : float or np.ndarray
        Distance to the boundary (in metres).
    d_i : float
        Preferred location of the PC (in metres).
    beta : float
        Scaling factor controlling the linear increase of radial tuning width with d_i.
    sigma0 : float
        Base radial tuning width at zero distance (in metres).
    addnoise : bool
        If True, multiplicative Gaussian noise is applied.
    noise_level : float
        Standard deviation of multiplicative noise (as a proportion of signal amplitude).

    Returns
    -------
    response : float or np.ndarray
        The PC's firing rate at the specified (r) location.
    """
    sigma_rad_x = ((r[0] - d_i[0]) / beta + 1) * sigma0
    sigma_rad_y = ((r[1] - d_i[1]) / beta + 1) * sigma0

    radial_component_x = np.exp(-((r[0] - d_i[0]) ** 2) / (2 * sigma_rad_x ** 2)) / (np.sqrt(2 * np.pi) * sigma_rad_x **2)
    radial_component_y = np.exp(-((r[1] - d_i[1]) ** 2) / (2 * sigma_rad_y ** 2)) / (np.sqrt(2 * np.pi) * sigma_rad_y **2)
    
    response = radial_component_x * radial_component_y

    if addnoise:
        gain_noise = np.random.normal(loc=1.0, scale=noise_level, size=response.shape)
        response = response * gain_noise
        response = np.clip(response, 0, None)  # ensure non-negative

    return response


def compute_PC_firing_rate(Position, d_i, beta, sigma0, addnoise=False):
        
    # Firing rate calculation along trajectory
    firing_rates = []

    for pos in tqdm(Position):
        fr = 0

        # Evaluate BVC response
        g = PC_response(pos, d_i, beta, sigma0, addnoise=addnoise, noise_level=5)

        fr += g 

        firing_rates.append(fr)

    firing_rates = np.array(firing_rates)

    return firing_rates


def compute_firing_rate_map(positions, rate, timestamps, bin_size=0.02, smoothing="unsmoothed"):
    """
    Computes and optionally plots the firing rate map, taking into account variable dwell times
    and applies specified smoothing method.

    Parameters
    ----------
    positions : (n, 2) array
        X and Y positions.
    spikes : (n,) array
        Spike counts at each position sample.
    timestamps : (n,) array
        Time stamps of each position sample.
    bin_size : float
        Spatial bin size (same units as positions, e.g. cm or proportion).
    smoothing : str
        "unsmoothed", "gaussian".
    plot : bool
        If True, plot the firing rate map.

    Returns
    -------
    rate_map : 2D array
        Spatial firing rate map (bins_y, bins_x).
    dwell_times : 2D array
        Dwell time per bin.
    spike_counts : 2D array
        Spike counts per bin.
    extent : tuple
        (xmin, xmax, ymin, ymax) for plotting.
    """

    # Bin positions
    x = positions[:, 0]
    y = positions[:, 1]

    x_bins = np.arange(x.min(), x.max() + bin_size, bin_size)
    y_bins = np.arange(y.min(), y.max() + bin_size, bin_size)

    x_idx = np.digitize(x, x_bins) - 1
    y_idx = np.digitize(y, y_bins) - 1

    # Compute dwell times
    t_diff = np.diff(timestamps)
    median_diff = np.median(t_diff)
    t_diff = np.where(t_diff > median_diff, median_diff, t_diff)
    t_diff = np.insert(t_diff, 0, 0)  # Insert 0 at start to align with positions

    dwell_times = np.zeros((len(x_bins), len(y_bins)))
    spike_counts = np.zeros((len(x_bins), len(y_bins)))

    np.add.at(dwell_times, (x_idx, y_idx), t_diff)
    np.add.at(spike_counts, (x_idx, y_idx), rate)

    # Compute rate map
    with np.errstate(divide='ignore', invalid='ignore'):
        rate_map = spike_counts / dwell_times
        rate_map[dwell_times == 0] = np.nan

    if smoothing == "gaussian":
        rate_map_zero = np.nan_to_num(rate_map, nan=0.0)
        valid_mask = ~np.isnan(rate_map)

        smoothed_data = gaussian_filter(rate_map_zero, sigma=2, mode='constant', cval=0.0)
        smoothed_mask = gaussian_filter(valid_mask.astype(float), sigma=2, mode='constant', cval=0.0)

        with np.errstate(invalid='ignore', divide='ignore'):
            rate_map = smoothed_data / smoothed_mask
            rate_map[smoothed_mask == 0] = np.nan

    # Prepare extent for plotting
    extent = [x_bins[0], x_bins[-1], y_bins[0], y_bins[-1]]


    return rate_map, dwell_times, spike_counts, extent


def get_boundary_distance_angle(Environment, Position, boundary_segments=360):
    # Calculate distance to boundaries
    boundary_angles = np.linspace(-np.pi, np.pi, boundary_segments, endpoint=False)

    all_boundary_distances = np.empty((len(Position), len(boundary_angles)))
    all_boundary_angles = np.empty((len(Position), len(boundary_angles)))

    # Cast ray from agent at angle theta, find distance to wall (simple box intersection)
    for i, pos in enumerate(Position):
        x, y = pos

        # For each boundary angle, find boundary location and calculate contribution
        for j, theta in enumerate(boundary_angles):
            
            # Convert theta to vector
            dx = np.cos(theta)
            dy = np.sin(theta)

            # Calculate intersection distances
            distances = []
            if dx != 0:
                t1 = (0 - x) / dx
                t2 = (1 - x) / dx
                if t1 > 0:
                    distances.append(t1)
                if t2 > 0:
                    distances.append(t2)
            if dy != 0:
                t3 = (0 - y) / dy
                t4 = (1 - y) / dy
                if t3 > 0:
                    distances.append(t3)
                if t4 > 0:
                    distances.append(t4)

            if len(distances) == 0:
                continue

            r = min(distances)  # distance to nearest boundary

            all_boundary_distances[i,j] = r
            all_boundary_angles[i,j] = theta

    return np.array(all_boundary_distances), np.array(all_boundary_angles)


def boundary_vector_preference_function(x):
        """
        This is a random function needed to efficiently produce boundary vector cells. 
        x is any array of final dimension shape shape[-1]=2. As I use it here x has the form of the 
        output of utils.vector_intercepts. I.e. each point gives shape[-1]=2 lambda values (lam1,lam2) 
        for where a pair of line segments intercept. This function gives a preference for each pair. 
        
        Preference is -1 if lam1<0 (the collision occurs behind the first point) and if lam2>1 or lam2<0 
        (the collision occurs ahead of the first point but not on the second line segment). 
        If neither of these are true it's 1/x (i.e. it prefers collisions which are closest and the 
        preference decays monotonically with distance). 

        Args:
            x (array): shape=(any_shape...,2)

        Returns:
            the preferece values: shape=(any_shape)
        """
        assert x.shape[-1] == 2
        pref = np.piecewise(
            x=x,
            condlist=(
                x[..., 0] > 0,
                x[..., 0] < 0,
                x[..., 1] < 0,
                x[..., 1] > 1,
            ),
            funclist=(
                1 / x[x[..., 0] > 0],
                -1,
                -1,
                -1,
            ),
        )
        return pref[..., 0]


def get_boundary_distances(Position, Env, boundary_segments=360):
        
    N_pos = Position.shape[0]
    N_angles = boundary_segments
    N_walls = Env.walls.shape[0]

    # 1. Create direction lines
    delta_theta = (2 * np.pi / N_angles)

    test_direction = np.array([1, 0])
    test_directions = []
    test_angles = []

    # numerically discretise over 360 degrees
    n_test_angles = int(2 * np.pi / delta_theta)
    for i in range(n_test_angles):
        test_direction_ = rotate(test_direction, 2 * np.pi * i * delta_theta)
        test_directions.append(test_direction_)
        test_angles.append(2 * np.pi * i * delta_theta)

    test_directions = np.array(test_directions)
    test_angles = np.array(test_angles)

    direction_line_segments = np.tile(
                np.expand_dims(test_directions, axis=0), reps=(N_pos, 1, 1)
            )  # (N_pos,N_angles,2)

    # 2. Add direction line segments to each position line (basically rays out from each position)
    pos_line_segments = np.tile(
                np.expand_dims(np.expand_dims(Position, axis=1), axis=1), reps=(1, N_angles, 2, 1)
            )  # (N_pos,N_angles,2,2)

    pos_line_segments[:, :, 1, :] += direction_line_segments  # (N_pos,N_angles,2,2)
    pos_line_segments = pos_line_segments.reshape(-1, 2, 2)  # (N_pos x N_angles,2,2)

    # 3. Find intercepts of rays with walls
    intercepts = vector_intercepts(pos_line_segments, Env.walls)
    intercepts = intercepts.reshape((N_pos, N_angles, N_walls, 2))  # (N_pos,N_angles,N_walls,2)

    # The last two numbers are:
    # intercepts[..., 0] = l_ray → how far along the ray you must travel from the agent’s position to hit the wall
    # intercepts[..., 1] = l_wall → where along the wall segment the hit occurs 

    dist_to_walls = intercepts[:, :, :, 0]

    # 4. TODO Only consider the first boundary forward from a ray. The ones behind it are shaded by closer walls. 

    strongest_response_to_walls = boundary_vector_preference_function(intercepts)

    first_wall = np.expand_dims(np.argmax(strongest_response_to_walls, axis=-1), axis=-1)

    dist_to_first_wall = np.take_along_axis(
                dist_to_walls, first_wall, axis=-1
            ).reshape(
                (N_pos, N_angles)
            )  # (N_pos,N_angles)

    # Create an array of angles per position 
    boundary_angles = np.tile(np.expand_dims(test_angles, axis=0), reps=(N_pos, 1))

    return dist_to_first_wall, boundary_angles
