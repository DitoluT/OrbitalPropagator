from Integrators import * # Numerical integrators

import os # For file path handling

import matplotlib.pyplot as plt # For plotting results

import numpy as np # For numerical operations

import csv # For reading CSV files

import pandas as pd # For data manipulation

CSV_FOLDER = os.path.join(os.getcwd(), '../csv') # Folder to save CSV files
IMG_FOLDER = os.path.join(os.getcwd(), '../images') # Folder to save images

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

    print("Saving results to CSV files...")

    # Save RK4 results
 
    df_rk4 = pd.DataFrame({
        'Time (s)': time_rk4,
        'x (km)': pos_rk4[0] / 1000,
        'y (km)': pos_rk4[1] / 1000,
        'z (km)': pos_rk4[2] / 1000,
        'vx (km/s)': vel_rk4[0] / 1000,
        'vy (km/s)': vel_rk4[1] / 1000,
        'vz (km/s)': vel_rk4[2] / 1000
    })

    df_rk4.to_csv(os.path.join(CSV_FOLDER, 'RK4_Dimensionalized.csv'), index=False)

    df_rk4_nd = pd.DataFrame({
        'Time (s)': time_rk4_nd,
        'x (km)': pos_rk4_nd[0] / 1000,
        'y (km)': pos_rk4_nd[1] / 1000,
        'z (km)': pos_rk4_nd[2] / 1000,
        'vx (km/s)': vel_rk4_nd[0] / 1000,
        'vy (km/s)': vel_rk4_nd[1] / 1000,
        'vz (km/s)': vel_rk4_nd[2] / 1000
    })

    df_rk4_nd.to_csv(os.path.join(CSV_FOLDER, 'RK4_Non_Dimensionalized.csv'), index=False)

    # Save Verlet results

    df_verlet = pd.DataFrame({
        'Time (s)': time_verlet,
        'x (km)': pos_verlet[0] / 1000,
        'y (km)': pos_verlet[1] / 1000,
        'z (km)': pos_verlet[2] / 1000,
        'vx (km/s)': vel_verlet[0] / 1000,
        'vy (km/s)': vel_verlet[1] / 1000,
        'vz (km/s)': vel_verlet[2] / 1000
    })

    df_verlet.to_csv(os.path.join(CSV_FOLDER, 'Verlet_Dimensionalized.csv'), index=False)

    df_verlet_nd = pd.DataFrame({
        'Time (s)': time_verlet_nd,
        'x (km)': pos_verlet_nd[0] / 1000,
        'y (km)': pos_verlet_nd[1] / 1000,
        'z (km)': pos_verlet_nd[2] / 1000,
        'vx (km/s)': vel_verlet_nd[0] / 1000,
        'vy (km/s)': vel_verlet_nd[1] / 1000,
        'vz (km/s)': vel_verlet_nd[2] / 1000
    })

    df_verlet_nd.to_csv(os.path.join(CSV_FOLDER, 'Verlet_Non_Dimensionalized.csv'), index=False)

    # Save Euler results

    df_euler = pd.DataFrame({
        'Time (s)': time_euler,
        'x (km)': pos_euler[0] / 1000,
        'y (km)': pos_euler[1] / 1000,
        'z (km)': pos_euler[2] / 1000,
        'vx (km/s)': vel_euler[0] / 1000,
        'vy (km/s)': vel_euler[1] / 1000,
        'vz (km/s)': vel_euler[2] / 1000
    })

    df_euler.to_csv(os.path.join(CSV_FOLDER, 'Euler_Dimensionalized.csv'), index=False)

    df_euler_nd = pd.DataFrame({
        'Time (s)': time_euler_nd,
        'x (km)': pos_euler_nd[0] / 1000,
        'y (km)': pos_euler_nd[1] / 1000,
        'z (km)': pos_euler_nd[2] / 1000,
        'vx (km/s)': vel_euler_nd[0] / 1000,
        'vy (km/s)': vel_euler_nd[1] / 1000,
        'vz (km/s)': vel_euler_nd[2] / 1000
    })

    df_euler_nd.to_csv(os.path.join(CSV_FOLDER, 'Euler_Non_Dimensionalized.csv'), index=False)

    print("Results saved to CSV files")

    # Plotting results

    print("PLOTTING RESULTS...")

    print("Plotting RK4 results in dimensionalized and non-dimensionalized units")

    # Plot orbital trajectories 3D RK4 and RK4 non-dimensionalized
    plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')
    ax.plot(pos_rk4[0], pos_rk4[1], pos_rk4[2], label='RK4 Dimensionalized', color='green')
    ax.plot(pos_rk4_nd[0], pos_rk4_nd[1], pos_rk4_nd[2], label='RK4 Non-Dimensionalized', color='red')
    ax.plot(df['x (km)'] * 1000, df['y (km)'] * 1000, df['z (km)'] * 1000, label='GMAT Data', color='cyan', linestyle='--')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Orbital Trajectories')
    ax.legend()
    plt.grid()
    plt.savefig(os.path.join(IMG_FOLDER, 'RK4_trajectories.png'))
    plt.show()
    

    print("Plot error(%) comparison between RK4 and GMAT data:")

    # Calculate and plot percetange error between RK4 and GMAT data
    pos_gmat = df[['x (km)', 'y (km)', 'z (km)']].values.T * 1000  # Convert to meters
    vel_gmat = df[['vx (km/sec)', 'vy (km/sec)', 'vz (km/sec)']].values.T * 1000  # Convert to meters/second
    
    pos_rk4_err = np.linalg.norm(pos_rk4 - pos_gmat, axis=0) / np.linalg.norm(pos_gmat, axis=0) * 100
    pos_rk4_nd_err = np.linalg.norm(pos_rk4_nd - pos_gmat, axis=0) / np.linalg.norm(pos_gmat, axis=0) * 100

    vel_rk4_err = np.linalg.norm(vel_rk4 - vel_gmat, axis=0) / np.linalg.norm(vel_gmat, axis=0) * 100
    vel_rk4_nd_err = np.linalg.norm(vel_rk4_nd - vel_gmat, axis=0) / np.linalg.norm(vel_gmat, axis=0) * 100

    # Plotting the error
    plt.figure(figsize=(10, 6))
    plt.plot(time_rk4, pos_rk4_err, label='RK4 Dimensionalized Error Position (%)', color='green')
    plt.plot(time_rk4_nd, pos_rk4_nd_err, label='RK4 Non-Dimensionalized Error Position(%)', color='red')
    plt.plot(time_rk4, vel_rk4_err, label='RK4 Dimensionalized Error Velocity (%)', color='darkgreen')
    plt.plot(time_rk4_nd, vel_rk4_nd_err, label='RK4 Non-Dimensionalized Error Velocity (%)', color='darkred')
    plt.xlabel('Time (s)')
    plt.ylabel('Error (%)')
    plt.title('Percentage Error of Position and Velocity RK4 Methods Compared to GMAT Data')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(IMG_FOLDER, 'RK4_error_comparison.png'))
    plt.show()

    # Errors comparison

    print("Plotting Verlet results in dimensionalized and non-dimensionalized units")

    # Plot orbital trajectories 3D Verlet and Verlet non-dimensionalized
    plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')

    # Plot Verlet results
    ax.plot(pos_verlet[0], pos_verlet[1], pos_verlet[2], label='Verlet Dimensionalized', color='blue')
    ax.plot(pos_verlet_nd[0], pos_verlet_nd[1], pos_verlet_nd[2], label='Verlet Non-Dimensionalized', color='orange')
    ax.plot(df['x (km)'] * 1000, df['y (km)'] * 1000, df['z (km)'] * 1000, label='GMAT Data', color='cyan', linestyle='--')

    # Set labels and title
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Orbital Trajectories')
    ax.legend()
    plt.savefig(os.path.join(IMG_FOLDER, 'Verlet_trajectories.png'))
    plt.show()

    print("Plot error(%) comparison between Verlet and GMAT data:")

    # Calculate and plot percentage error between Verlet and GMAT data
    pos_verlet_error = np.linalg.norm(pos_verlet - pos_gmat, axis=0) / np.linalg.norm(pos_gmat, axis=0) * 100
    pos_verlet_nd_error = np.linalg.norm(pos_verlet_nd - pos_gmat, axis=0) / np.linalg.norm(pos_gmat, axis=0) * 100

    vel_verlet_error = np.linalg.norm(vel_verlet - vel_gmat, axis=0) / np.linalg.norm(vel_gmat, axis=0) * 100
    vel_verlet_nd_error = np.linalg.norm(vel_verlet_nd - vel_gmat, axis=0) / np.linalg.norm(vel_gmat, axis=0) * 100

    # Plotting the error
    plt.figure(figsize=(10, 6))
    plt.plot(time_verlet, pos_verlet_error, label='Verlet Dimensionalized Error Position (%)', color='blue')
    plt.plot(time_verlet_nd, pos_verlet_nd_error, label='Verlet Non-Dimensionalized Error Position (%)', color='orange')
    plt.plot(time_verlet, vel_verlet_error, label='Verlet Dimensionalized Error Velocity (%)', color='darkblue')
    plt.plot(time_verlet_nd, vel_verlet_nd_error, label='Verlet Non-Dimensionalized Error Velocity (%)', color='darkorange')
    plt.xlabel('Time (s)')
    plt.ylabel('Error (%)')
    plt.title('Percentage Error of Verlet Methods Compared to GMAT Data')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(IMG_FOLDER, 'Verlet_error_comparison.png'))
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
    plt.savefig(os.path.join(IMG_FOLDER, 'Euler_trajectories.png'))
    plt.show()

    print("Plot error(%) comparison between Euler and GMAT data:")

    # Calculate and plot percentage error between Euler and GMAT data
    pos_euler_error = np.linalg.norm(pos_euler - pos_gmat, axis=0) / np.linalg.norm(pos_gmat, axis=0) * 100
    pos_euler_nd_error = np.linalg.norm(pos_euler_nd - pos_gmat, axis=0) / np.linalg.norm(pos_gmat, axis=0) * 100

    vel_euler_error = np.linalg.norm(vel_euler - vel_gmat, axis=0) / np.linalg.norm(vel_gmat, axis=0) * 100
    vel_euler_nd_error = np.linalg.norm(vel_euler_nd - vel_gmat, axis=0) / np.linalg.norm(vel_gmat, axis=0) * 100

    # Plotting the error
    plt.figure(figsize=(10, 6))
    plt.plot(time_euler, pos_euler_error, label='Euler Dimensionalized Error Position (%)', color='blue')
    plt.plot(time_euler_nd, pos_euler_nd_error, label='Euler Non-Dimensionalized Error Position (%)', color='orange')
    plt.plot(time_euler, vel_euler_error, label='Euler Dimensionalized Error Velocity (%)', color='darkblue')
    plt.plot(time_euler_nd, vel_euler_nd_error, label='Euler Non-Dimensionalized Error Velocity (%)', color='darkorange')
    plt.xlabel('Time (s)')
    plt.ylabel('Error (%)')
    plt.title('Percentage Error of Euler Methods Compared to GMAT Data')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(IMG_FOLDER, 'Euler_error_comparison.png'))
    plt.show()

    print("Plotting completed for all methods")
    print("Plotting logarithmic error scale for comparison between RK4, Verlet, and Euler methods")

    # Plotting logarithmic error scale for comparison between RK4, Verlet, and Euler methods
    plt.figure(figsize=(10, 6))
    plt.semilogy(time_rk4, pos_rk4_err, label='RK4 Dimensionalized Error Position (%)', color='green')
    plt.semilogy(time_rk4_nd, pos_rk4_nd_err, label='RK4 Non-Dimensionalized Error Position (%)', color='red')
    plt.semilogy(time_verlet, pos_verlet_error, label='Verlet Dimensionalized Error Position (%)', color='blue')
    plt.semilogy(time_verlet_nd, pos_verlet_nd_error, label='Verlet Non-Dimensionalized Error Position (%)', color='orange')
    plt.semilogy(time_euler, pos_euler_error, label='Euler Dimensionalized Error Position (%)', color='purple')
    plt.semilogy(time_euler_nd, pos_euler_nd_error, label='Euler Non-Dimensionalized Error Position (%)', color='brown')
    plt.xlabel('Time (s)')
    plt.ylabel('Logarithmic Error (%)')
    plt.title('Logarithmic Percentage Error of Position Methods Compared to GMAT Data')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(IMG_FOLDER, 'Logarithmic_Error_Comparison.png'))
    plt.show()

    print("PLOTTING COMPLETED")