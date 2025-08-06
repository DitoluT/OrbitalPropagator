from Integrators import * # Numerical integrators

import os # For file path handling

import matplotlib.pyplot as plt # For plotting results

import numpy as np # For numerical operations

import csv # For reading CSV files

import pandas as pd # For data manipulation


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
    Npoints = 24 * 360 + 1
    pos_initial = df[['x (km)', 'y (km)', 'z (km)']].iloc[0].values
    vel_initial = df[['vx (km/sec)', 'vy (km/sec)', 'vz (km/sec)']].iloc[0].values
    
    # Convert to numpy arrays
    pos_initial = np.array(pos_initial) * 1000
    print("Initial Position (m):", pos_initial)
    vel_initial = np.array(vel_initial) * 1000
    print("Initial Velocity (m/s):", vel_initial)

    print("RUNNING NUMERICAL METHODS...")
    print("Running Kepler RK4 using non-dimensionalized units")

    time_rk4_nd, pos_rk4_nd, vel_rk4_nd = Kepler_RK4(pos_initial, vel_initial, time_initial, time_final, Npoints)

    print("Running Kepler RK4 using dimensionalized units")

    time_rk4, pos_rk4, vel_rk4 = Kepler_RK4_Dimensionalized(pos_initial, vel_initial, time_initial, time_final, Npoints)

    print("Running Verlet method using non-dimensionalized units")

    time_verlet_nd, pos_verlet_nd, vel_verlet_nd = Kepler_Verlet(pos_initial, vel_initial, time_initial, time_final, Npoints)

    print("Running Verlet method using dimensionalized units")

    time_verlet, pos_verlet, vel_verlet = Kepler_Verlet_Dimensionalized(pos_initial, vel_initial, time_initial, time_final, Npoints)

    print("Running Euler method using non-dimensionalized units")

    time_euler_nd, pos_euler_nd, vel_euler_nd = Kepler_Euler(pos_initial, vel_initial, time_initial, time_final, Npoints)

    print("Running Euler method using dimensionalized units")

    time_euler, pos_euler, vel_euler = Kepler_Euler_Dimensionalized(pos_initial, vel_initial, time_initial, time_final, Npoints)

    print("RUNNING COMPLETED")

    print("PLOTTING RESULTS...")

    print("Plotting RK4 results in dimensionalized and non-dimensionalized units")

    # Plot orbital trajectories 3D RK4 and RK4 non-dimensionalized
    plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')
    
    # Plot RK4 results
    #ax.plot(pos_rk4[0], pos_rk4[1], pos_rk4[2], label='RK4 Dimensionalized', color='green')
    ax.plot(pos_rk4_nd[0], pos_rk4_nd[1], pos_rk4_nd[2], label='RK4 Non-Dimensionalized', color='red')
    ax.plot(df['x (km)'] * 1000, df['y (km)'] * 1000, df['z (km)'] * 1000, label='GMAT Data', color='cyan', linestyle='--')
    
    # Set labels and title
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Orbital Trajectories')
    ax.legend()
    
    plt.show()

    print("Plot error(%) comparison between RK4 and GMAT data:")

    # Calculate and plot percetange error between RK4 and GMAT data
    pos_gmat = df[['x (km)', 'y (km)', 'z (km)']].values.T * 1000  # Convert to meters
    
    pos_rk4_error = np.abs((pos_rk4 - pos_gmat) / pos_gmat) * 100
    pos_rk4_nd_error = np.abs((pos_rk4_nd - pos_gmat) / pos_gmat) * 100

    # Plotting the error
    plt.figure(figsize=(10, 6))
    plt.plot(time_rk4, pos_rk4_error[0], label='RK4 Dimensionalized Error (%)', color='green')
    #plt.plot(time_rk4_nd, pos_rk4_nd_error[0], label='RK4 Non-Dimensionalized Error (%)', color='red')
    plt.xlabel('Time (s)')
    plt.ylabel('Error (%)')
    plt.title('Percentage Error of RK4 Methods Compared to GMAT Data')
    plt.legend()
    plt.grid()
    plt.show()

    print("Plotting Verlet results in dimensionalized and non-dimensionalized units")

    # Plot orbital trajectories 3D Verlet and Verlet non-dimensionalized
    plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')

    # Plot Verlet results
    ax.plot(pos_verlet[0], pos_verlet[1], pos_verlet[2], label='Verlet Dimensionalized', color='blue')
    #ax.plot(pos_verlet_nd[0], pos_verlet_nd[1], pos_verlet_nd[2], label='Verlet Non-Dimensionalized', color='orange')
    ax.plot(df['x (km)'] * 1000, df['y (km)'] * 1000, df['z (km)'] * 1000, label='GMAT Data', color='cyan', linestyle='--')

    # Set labels and title
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Orbital Trajectories')
    ax.legend()
    
    plt.show()

    print("Plot error(%) comparison between Verlet and GMAT data:")

    # Calculate and plot percentage error between Verlet and GMAT data
    pos_verlet_error = np.abs((pos_verlet - pos_gmat) / pos_gmat) * 100
    pos_verlet_nd_error = np.abs((pos_verlet_nd - pos_gmat) / pos_gmat) * 100

    # Plotting the error
    plt.figure(figsize=(10, 6))
    plt.plot(time_verlet, pos_verlet_error[0], label='Verlet Dimensionalized Error (%)', color='blue')
    plt.plot(time_verlet_nd, pos_verlet_nd_error[0], label='Verlet Non-Dimensionalized Error (%)', color='orange')
    plt.xlabel('Time (s)')
    plt.ylabel('Error (%)')
    plt.title('Percentage Error of Verlet Methods Compared to GMAT Data')
    plt.legend()
    plt.grid()
    plt.show()

    print("Plotting Euler results in dimensionalized and non-dimensionalized units")

    # Plot orbital trajectories 3D Euler and Euler non-dimensionalized
    plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')

    # Plot Euler results
    ax.plot(pos_euler[0], pos_euler[1], pos_euler[2], label='Euler Dimensionalized', color='purple')
    ax.plot(pos_euler_nd[0], pos_euler_nd[1], pos_euler_nd[2], label='Euler Non-Dimensionalized', color='brown')
    ax.plot(df['x (km)'] * 1000, df['y (km)'] * 1000, df['z (km)'] * 1000, label='GMAT Data', color='cyan', linestyle='--')

    # Set labels and title
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Orbital Trajectories')
    ax.legend()
    
    plt.show()

    print("Plot error(%) comparison between Euler and GMAT data:")

    # Calculate and plot percentage error between Euler and GMAT data
    pos_euler_error = np.abs((pos_euler - pos_gmat) / pos_gmat) * 100
    pos_euler_nd_error = np.abs((pos_euler_nd - pos_gmat) / pos_gmat) * 100

    # Plotting the error
    plt.figure(figsize=(10, 6))
    plt.plot(time_euler, pos_euler_error[0], label='Euler Dimensionalized Error (%)', color='purple')
    plt.plot(time_euler_nd, pos_euler_nd_error[0], label='Euler Non-Dimensionalized Error (%)', color='brown')
    plt.xlabel('Time (s)')
    plt.ylabel('Error (%)')
    plt.title('Percentage Error of Euler Methods Compared to GMAT Data')
    plt.legend()
    plt.grid()
    plt.show()

    print("Plotting completed for all methods")
    print("PLOTTING COMPLETED")