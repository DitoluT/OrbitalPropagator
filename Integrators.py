import numpy as np

## ------ ORBITAL PROPAGATOR ------ ##

MU_EARTH = 3.986004418e14  # m³/s²

# Right-Hand-Side for the orbital propagator
# Inputs:
# - pos[ 3 ]: 3D array holding X, Y, Z position [m]
# - vel[ 3 ]: 3D array holding V_X, V_Y, V_Z velocity [m/sec]
# Outputs:
# - rhs_pos[ 3 ]: RHS of position (Xdot, Ydot, Zdot) in [m/sec]
# - rhs_vel[ 3 ]: RHS of velocity (V_Xdot, V_Ydot, V_Zdot) in [m/sec^2]
def Orbital_RHS(pos, vel):
    rhs_pos = vel
    rhs_vel = - pos / (np.linalg.norm(pos) ** 3)
    return rhs_pos, rhs_vel

# Right-Hand-Side for the orbital propagator (dimensionalized version)
# Inputs:
# - pos[ 3 ]: 3D array holding X, Y, Z position [m]
# - vel[ 3 ]: 3D array holding V_X, V_Y, V_Z velocity [m/sec]
# Outputs:
# - rhs_pos[ 3 ]: RHS of position (Xdot, Ydot, Zdot) in [m/sec]
# - rhs_vel[ 3 ]: RHS of velocity (V_Xdot, V_Ydot, V_Zdot) in [m/sec^2]
def Orbital_RHS_Dimensionalized(pos, vel):
    rhs_pos = vel
    rhs_vel = - pos * (MU_EARTH) / (np.linalg.norm(pos) ** 3)
    return rhs_pos, rhs_vel

# Non-dimensionalization function for orbital mechanics
# Inputs:
# - pos[ 3, Npoints ]: Position array with shape (3, Npoints) [m]
# - vel[ 3, Npoints ]: Velocity array with shape (3, Npoints) [m/sec]
# - dt: Time step [sec]
# Outputs:
# - pos_nd[ 3, Npoints ]: Non-dimensionalized position array
# - vel_nd[ 3, Npoints ]: Non-dimensionalized velocity array
# - dt_nd: Non-dimensionalized time step
# - t_0: Characteristic time scale [sec]
# - r_0: Characteristic length scale [m]
def nondimensionalize(pos, vel, dt):
    # pos and vel have shape (3, Npoints)
    r_0 = np.linalg.norm(pos[:, 0])  # Norm of the first point
    print(r_0, pos[:, 0], pos.shape)
    
    t_0 = np.sqrt(r_0**3 / MU_EARTH)
    
    pos_nd = pos / r_0
    vel_nd = vel * t_0 / r_0
    dt_nd = dt / t_0
    
    return pos_nd, vel_nd, dt_nd, t_0, r_0

# Re-dimensionalization function for orbital mechanics
# Inputs:
# - pos[ 3, Npoints ]: Non-dimensionalized position array
# - vel[ 3, Npoints ]: Non-dimensionalized velocity array
# - t_0: Characteristic time scale [sec]
# - r_0: Characteristic length scale [m]
# Outputs:
# - pos_d[ 3, Npoints ]: Dimensionalized position array [m]
# - vel_d[ 3, Npoints ]: Dimensionalized velocity array [m/sec]
def dimensionalize(pos, vel, t_0, r_0):
    pos_d = pos * r_0
    vel_d = vel * r_0 / t_0
    return pos_d, vel_d

# Kepler orbital propagator using Euler integration method
# Inputs:
# - pos_init[ 3 ]: Initial position vector [m]
# - vel_init[ 3 ]: Initial velocity vector [m/sec]
# - t_initial: Initial time [sec]
# - t_final: Final time [sec]
# - Npoints: Number of time points for integration
# Outputs:
# - time[ Npoints ]: Time array [sec]
# - pos[ 3, Npoints ]: Position array [m]
# - vel[ 3, Npoints ]: Velocity array [m/sec]
def Kepler_Euler(pos_init, vel_init, t_initial, t_final, Npoints):
    dt = (t_final - t_initial) / (Npoints - 1.0)
    time = np.linspace(t_initial, t_final, Npoints)
    
    # Initialize arrays with shape (3, Npoints)
    pos = np.zeros((3, Npoints))
    vel = np.zeros((3, Npoints))
    pos[:, 0] = np.array(pos_init)
    vel[:, 0] = np.array(vel_init)

    pos, vel, dt_nd, t_0, r_0 = nondimensionalize(pos, vel, dt)

    for i in range(Npoints - 1):
        rhs_pos, rhs_vel = Orbital_RHS(pos[:, i], vel[:, i])
        pos[:, i + 1] = pos[:, i] + rhs_pos * dt_nd
        vel[:, i + 1] = vel[:, i] + rhs_vel * dt_nd

    pos, vel = dimensionalize(pos, vel, t_0, r_0)

    return time, pos, vel

# Kepler orbital propagator using Verlet integration method
# Inputs:
# - pos_init[ 3 ]: Initial position vector [m]
# - vel_init[ 3 ]: Initial velocity vector [m/sec]
# - t_initial: Initial time [sec]
# - t_final: Final time [sec]
# - Npoints: Number of time points for integration
# Outputs:
# - time[ Npoints ]: Time array [sec]
# - pos[ 3, Npoints ]: Position array [m]
# - vel[ 3, Npoints ]: Velocity array [m/sec]
def Kepler_Verlet(pos_init, vel_init, t_initial, t_final, Npoints):
    dt = (t_final - t_initial) / (Npoints - 1.0)
    time = np.linspace(t_initial, t_final, Npoints)

    # Initialize arrays with shape (3, Npoints)
    pos = np.zeros((3, Npoints))
    vel = np.zeros((3, Npoints))
    pos[:, 0] = np.array(pos_init)
    vel[:, 0] = np.array(vel_init)
    
    for i in range(Npoints - 1):
        rhs_pos, rhs_vel = Orbital_RHS_Dimensionalized(pos[:, i], vel[:, i])
        v_half = vel[:, i] + rhs_vel * dt / 2.0
        rhs_pos, rhs_vel = Orbital_RHS_Dimensionalized(pos[:, i], v_half)
        pos[:, i + 1] = pos[:, i] + rhs_pos * dt
        rhs_pos, rhs_vel = Orbital_RHS_Dimensionalized(pos[:, i + 1], v_half)
        vel[:, i + 1] = v_half + rhs_vel * dt / 2.0

    return time, pos, vel

# Kepler orbital propagator using Runge-Kutta 4th order integration method
# Inputs:
# - pos_init[ 3 ]: Initial position vector [m]
# - vel_init[ 3 ]: Initial velocity vector [m/sec]
# - t_initial: Initial time [sec]
# - t_final: Final time [sec]
# - Npoints: Number of time points for integration
# Outputs:
# - time[ Npoints ]: Time array [sec]
# - pos[ 3, Npoints ]: Position array [m]
# - vel[ 3, Npoints ]: Velocity array [m/sec]
def Kepler_RK4(pos_init, vel_init, t_initial, t_final, Npoints):
    time = np.linspace(t_initial, t_final, Npoints)
    dt = (t_final - t_initial) / (Npoints - 1.0)

    # Initialize arrays with shape (3, Npoints)
    pos = np.zeros((3, Npoints))
    vel = np.zeros((3, Npoints))
    pos[:, 0] = np.array(pos_init)
    vel[:, 0] = np.array(vel_init)

    # For non-dimensionalized comparison
    pos_1 = pos.copy()
    vel_1 = vel.copy()
    pos_1, vel_1, dt_nd, t_0, r_0 = nondimensionalize(pos_1, vel_1, dt)

    for i in range(Npoints - 1):
        # RK4 steps dimensionalized

        """
        k1_pos, k1_vel = Orbital_RHS_Dimensionalized(pos[:, i], vel[:, i])
        k2_pos, k2_vel = Orbital_RHS_Dimensionalized(pos[:, i] + 0.5 * dt * k1_pos, vel[:, i] + 0.5 * dt * k1_vel)
        k3_pos, k3_vel = Orbital_RHS_Dimensionalized(pos[:, i] + 0.5 * dt * k2_pos, vel[:, i] + 0.5 * dt * k2_vel)
        k4_pos, k4_vel = Orbital_RHS_Dimensionalized(pos[:, i] + dt * k3_pos, vel[:, i] + dt * k3_vel)

        pos[:, i + 1] = pos[:, i] + (dt / 6) * (k1_pos + 2 * k2_pos + 2 * k3_pos + k4_pos)
        vel[:, i + 1] = vel[:, i] + (dt / 6) * (k1_vel + 2 * k2_vel + 2 * k3_vel + k4_vel)

        print(f"Dimensionalized Step {i+1}/{Npoints-1}: Position = {pos[:, i + 1]}, Velocity = {vel[:, i + 1]}")
        """
        # RK4 steps NON-dimensionalized
        k1_pos, k1_vel = Orbital_RHS(pos_1[:, i], vel_1[:, i])
        k2_pos, k2_vel = Orbital_RHS(pos_1[:, i] + 0.5 * dt_nd * k1_pos, vel_1[:, i] + 0.5 * dt_nd * k1_vel)
        k3_pos, k3_vel = Orbital_RHS(pos_1[:, i] + 0.5 * dt_nd * k2_pos, vel_1[:, i] + 0.5 * dt_nd * k2_vel)
        k4_pos, k4_vel = Orbital_RHS(pos_1[:, i] + dt_nd * k3_pos, vel_1[:, i] + dt_nd * k3_vel)

        pos_1[:, i + 1] = pos_1[:, i] + (dt_nd / 6) * (k1_pos + 2 * k2_pos + 2 * k3_pos + k4_pos)
        vel_1[:, i + 1] = vel_1[:, i] + (dt_nd / 6) * (k1_vel + 2 * k2_vel + 2 * k3_vel + k4_vel)

        """
        # Dimensionalize for comparison
        pos_comp, vel_comp = dimensionalize(pos_1[:, i + 1:i + 2], vel_1[:, i + 1:i + 2], t_0, r_0)

        print(f"NON-Dimensionalized Step {i+1}/{Npoints-1}: Position = {pos_comp[:, 0]}, Velocity = {vel_comp[:, 0]}")
        print(f"Comparison Step {i+1}/{Npoints-1}: Pos diff = {np.max(np.abs(pos_comp[:, 0] - pos[:, i + 1]))}, Vel diff = {np.max(np.abs(vel_comp[:, 0] - vel[:, i + 1]))}")
        """

    # Dimensionalize the final result
    pos, vel = dimensionalize(pos_1, vel_1, t_0, r_0)

    return time, pos, vel