from Integrators import *
import matplotlib.pyplot as plt
from numpy import pi
import numpy as np
import pandas as pd
import csv

plt.rcParams['figure.figsize'] = [12, 8]  # Default figure size
plt.rcParams['figure.dpi'] = 100  # DPI setting
plt.rcParams['savefig.dpi'] = 150  # For saved figures



# Parse a .csv file with satellite time, position and velocity (PVT) for plotting
# Inputs:
# - filename of the .csv file where the results are saved
# NOTE: File structure is expected to be "Time (UTCG),x (km),y (km),z (km),vx (km/sec),vy (km/sec),vz (km/sec)"
# Output:
# It will return arrays of time, pos, vel as:
# - time[ Npoints ]: time in [sec] from simulation start
# - pos[ 3 ][ Npoints ]: 3D position in [km]
# - vel[ 3 ][ Npoints ]: 3D velocity in [km/s]
def parse_orbit_data(filename):
    fp = open(filename, "r")
    
    if fp.readable():
        data = csv.reader(fp)
        lst = []
        for line in data:
            lst.append(line)
        ndata = len(lst) - 1
        
        time = np.zeros(ndata)
        pos = np.zeros((3, ndata))
        vel = np.zeros((3, ndata))
        
        for i in range(0, ndata):
            time[i] = float(lst[i + 1][0])
            for j in range(0, 3):
                pos[j][i] = float(lst[i + 1][j + 1])
                vel[j][i] = float(lst[i + 1][j + 4])
    else:
        print("Unreadable data, something's wrong with the file " + filename)
    
    fp.close()
    return time, pos, vel

if __name__ == "__main__":

    # Read data from CSV file
    df = pd.read_csv('Satellite_PVT_GMAT.csv')
    
    time_initial = 0.0
    time_final = 86400.0
    Npoints = 24 * 360
    pos_initial = df[['x (km)', 'y (km)', 'z (km)']].iloc[0].values
    vel_initial = df[['vx (km/sec)', 'vy (km/sec)', 'vz (km/sec)']].iloc[0].values
    
    # Convert to numpy arrays
    pos_initial = np.array(pos_initial) * 1000
    print("Initial Position (m):", pos_initial)
    vel_initial = np.array(vel_initial) * 1000
    print("Initial Velocity (m/s):", vel_initial)

    # Run the Kepler RK4 integrator
    time, pos, vel = Kepler_RK4(pos_initial, vel_initial, time_initial, time_final, Npoints)

    # Plotting the results
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(pos[0]/1000, pos[1]/1000, pos[2]/1000, label='Kepler RK4 Trajectory', color='blue')
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')
    ax.set_title('3D Trajectory of the Satellite')
    ax.legend()
    plt.show()

    # 2D Plot example
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(time/3600, pos[0]/1000, label='X Position', color='blue')
    ax.plot(time/3600, pos[1]/1000, label='Y Position', color='orange')
    ax.plot(time/3600, pos[2]/1000, label='Z Position', color='green')
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Position (km)')
    ax.set_title('Satellite Position Over Time')
    ax.legend()
    ax.grid(True)
    plt.show()

    # ========== COMPARISON WITH REFERENCE DATA ==========
    print("\n" + "="*60)
    print("COMPARING INTEGRATION METHODS WITH REFERENCE DATA")
    print("="*60)
    
    # Parse reference data from CSV
    time_ref, pos_ref, vel_ref = parse_orbit_data('Satellite_PVT_GMAT.csv')
    
    # Convert reference data to SI units
    pos_ref_m = pos_ref * 1000  # km to m
    vel_ref_m = vel_ref * 1000  # km/s to m/s
    
    # Get initial conditions from reference data
    pos_0 = pos_ref_m[:, 0]  # Initial position [m]
    vel_0 = vel_ref_m[:, 0]  # Initial velocity [m/s]
    
    # Use same time parameters as reference
    t_initial_ref = time_ref[0]
    t_final_ref = time_ref[-1]
    Npoints_ref = len(time_ref)
    
    print(f"Reference data: {Npoints_ref} points from {t_initial_ref} to {t_final_ref} seconds")
    
    # Run all integration methods
    print("\nRunning integration methods...")
    
    # RK4 method
    print("Running RK4...")
    time_rk4, pos_rk4, vel_rk4 = Kepler_RK4(pos_0, vel_0, t_initial_ref, t_final_ref, Npoints_ref)
    
    # Euler method
    print("Running Euler...")
    time_euler, pos_euler, vel_euler = Kepler_Euler(pos_0, vel_0, t_initial_ref, t_final_ref, Npoints_ref)
    
    # Verlet method
    print("Running Verlet...")
    time_verlet, pos_verlet, vel_verlet = Kepler_Verlet(pos_0, vel_0, t_initial_ref, t_final_ref, Npoints_ref)
    
    # Calculate position errors for each method
    def calculate_position_error(pos_computed, pos_reference):
        """Calculate position error magnitude between computed and reference positions"""
        error = np.zeros(pos_computed.shape[1])
        for i in range(pos_computed.shape[1]):
            error[i] = np.linalg.norm(pos_computed[:, i] - pos_reference[:, i])
        return error
    
    # Calculate errors (convert to km for plotting)
    error_rk4 = calculate_position_error(pos_rk4, pos_ref_m) / 1000  # Convert to km
    error_euler = calculate_position_error(pos_euler, pos_ref_m) / 1000
    error_verlet = calculate_position_error(pos_verlet, pos_ref_m) / 1000
    
    # Print error statistics
    print(f"\nERROR STATISTICS (in km):")
    print(f"RK4    - Max: {np.max(error_rk4):.6f}, Mean: {np.mean(error_rk4):.6f}, Final: {error_rk4[-1]:.6f}")
    print(f"Euler  - Max: {np.max(error_euler):.6f}, Mean: {np.mean(error_euler):.6f}, Final: {error_euler[-1]:.6f}")
    print(f"Verlet - Max: {np.max(error_verlet):.6f}, Mean: {np.mean(error_verlet):.6f}, Final: {error_verlet[-1]:.6f}")
    
    # 3D Trajectory Comparison Plot
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot reference trajectory
    ax.plot(pos_ref[0], pos_ref[1], pos_ref[2], 'k-', linewidth=2, label='Reference (GMAT)', alpha=0.8)
    
    # Plot computed trajectories
    ax.plot(pos_rk4[0]/1000, pos_rk4[1]/1000, pos_rk4[2]/1000, 'r--', linewidth=1.5, label='RK4')
    ax.plot(pos_euler[0]/1000, pos_euler[1]/1000, pos_euler[2]/1000, 'b:', linewidth=1.5, label='Euler')
    ax.plot(pos_verlet[0]/1000, pos_verlet[1]/1000, pos_verlet[2]/1000, 'g-.', linewidth=1.5, label='Verlet')
    
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')
    ax.set_title('3D Orbital Trajectory Comparison')
    ax.legend()
    plt.show()
    
    # Position components comparison
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    components = ['X', 'Y', 'Z']
    colors_ref = ['black'] * 3
    colors_methods = [['red', 'blue', 'green']] * 3
    
    for i in range(3):
        # Reference data
        axes[i].plot(time_ref/3600, pos_ref[i], 'k-', linewidth=2, label='Reference (GMAT)', alpha=0.8)
        
        # Computed data
        axes[i].plot(time_rk4/3600, pos_rk4[i]/1000, 'r--', linewidth=1.5, label='RK4')
        axes[i].plot(time_euler/3600, pos_euler[i]/1000, 'b:', linewidth=1.5, label='Euler')
        axes[i].plot(time_verlet/3600, pos_verlet[i]/1000, 'g-.', linewidth=1.5, label='Verlet')
        
        axes[i].set_xlabel('Time (hours)')
        axes[i].set_ylabel(f'{components[i]} Position (km)')
        axes[i].set_title(f'{components[i]} Position vs Time')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Error plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Position error magnitude
    axes[0, 0].semilogy(time_ref/3600, error_rk4, 'r-', linewidth=2, label='RK4')
    axes[0, 0].semilogy(time_ref/3600, error_euler, 'b-', linewidth=2, label='Euler')
    axes[0, 0].semilogy(time_ref/3600, error_verlet, 'g-', linewidth=2, label='Verlet')
    axes[0, 0].set_xlabel('Time (hours)')
    axes[0, 0].set_ylabel('Position Error (km)')
    axes[0, 0].set_title('Position Error Magnitude (Log Scale)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Linear scale error plot
    axes[0, 1].plot(time_ref/3600, error_rk4, 'r-', linewidth=2, label='RK4')
    axes[0, 1].plot(time_ref/3600, error_euler, 'b-', linewidth=2, label='Euler')
    axes[0, 1].plot(time_ref/3600, error_verlet, 'g-', linewidth=2, label='Verlet')
    axes[0, 1].set_xlabel('Time (hours)')
    axes[0, 1].set_ylabel('Position Error (km)')
    axes[0, 1].set_title('Position Error Magnitude (Linear Scale)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Error growth rate
    error_rk4_rate = np.gradient(error_rk4, time_ref/3600)
    error_euler_rate = np.gradient(error_euler, time_ref/3600)
    error_verlet_rate = np.gradient(error_verlet, time_ref/3600)
    
    axes[1, 0].plot(time_ref/3600, error_rk4_rate, 'r-', linewidth=2, label='RK4')
    axes[1, 0].plot(time_ref/3600, error_euler_rate, 'b-', linewidth=2, label='Euler')
    axes[1, 0].plot(time_ref/3600, error_verlet_rate, 'g-', linewidth=2, label='Verlet')
    axes[1, 0].set_xlabel('Time (hours)')
    axes[1, 0].set_ylabel('Error Rate (km/hour)')
    axes[1, 0].set_title('Error Growth Rate')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Cumulative error
    cum_error_rk4 = np.cumsum(error_rk4)
    cum_error_euler = np.cumsum(error_euler)
    cum_error_verlet = np.cumsum(error_verlet)
    
    axes[1, 1].plot(time_ref/3600, cum_error_rk4, 'r-', linewidth=2, label='RK4')
    axes[1, 1].plot(time_ref/3600, cum_error_euler, 'b-', linewidth=2, label='Euler')
    axes[1, 1].plot(time_ref/3600, cum_error_verlet, 'g-', linewidth=2, label='Verlet')
    axes[1, 1].set_xlabel('Time (hours)')
    axes[1, 1].set_ylabel('Cumulative Error (km)')
    axes[1, 1].set_title('Cumulative Error')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Final summary plot - Error comparison at different time intervals
    time_intervals = [6, 12, 18, 24]  # hours
    methods = ['RK4', 'Euler', 'Verlet']
    errors_at_intervals = {method: [] for method in methods}
    
    for interval in time_intervals:
        idx = np.argmin(np.abs(time_ref/3600 - interval))
        errors_at_intervals['RK4'].append(error_rk4[idx])
        errors_at_intervals['Euler'].append(error_euler[idx])
        errors_at_intervals['Verlet'].append(error_verlet[idx])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(time_intervals))
    width = 0.25
    
    for i, method in enumerate(methods):
        offset = (i - 1) * width
        bars = ax.bar(x + offset, errors_at_intervals[method], width, label=method)
        
        # Add value labels on bars
        for bar, value in zip(bars, errors_at_intervals[method]):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(errors_at_intervals[method])*0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Position Error (km)')
    ax.set_title('Position Error Comparison at Different Time Intervals')
    ax.set_xticks(x)
    ax.set_xticklabels(time_intervals)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.show()
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)