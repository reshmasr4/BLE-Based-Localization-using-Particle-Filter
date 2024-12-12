import matplotlib.pyplot as plt

# Data for particle sizes and execution times
particle_sizes = [500, 800, 1000, 2000]
rssi_times = [6.1110, 6.1198, 6.1307, 6.1588]
coordinates_times = [37.60, 61.63, 74.17, 148.07]

# Create the figure and axis
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot for Coordinates Estimation
ax1.plot(particle_sizes, coordinates_times, marker='o', color='orange', label='Coordinates Estimation')
ax1.set_xlabel('Particle Size', fontsize=12)
ax1.set_ylabel('Execution Time (s) - Coordinates Estimation', fontsize=12, color='orange')
ax1.tick_params(axis='y', labelcolor='orange')

# Create a secondary y-axis for RSSI Proximity times
ax2 = ax1.twinx()
ax2.plot(particle_sizes, rssi_times, marker='o', color='blue', label='RSSI Proximity')
ax2.set_ylabel('Execution Time (s) - RSSI Proximity', fontsize=12, color='blue')
ax2.tick_params(axis='y', labelcolor='blue')

# Add a title
plt.title('Performance Comparison of Particle Size and Execution Time', fontsize=14)

# Show grid and plot
fig.tight_layout()
plt.grid(True)
plt.show()
