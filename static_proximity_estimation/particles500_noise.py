import os
import random
import numpy as np
import pygame
from math import exp, pi, sqrt, log10
import pandas as pd
import time

from matplotlib import pyplot as plt

# Directory to save frames
save_dir = "simulation_frames/Particles_500/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

frame_count = 0  # Frame counter

# Function to save the current frame
def save_frame(stage):
    global frame_count
    file_name = f"{save_dir}frame_{frame_count:04d}.png"
    pygame.image.save(screen, file_name)
    frame_count += 1
    print(f"Saved frame {frame_count} for stage {stage}")


# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Initialize Pygame for visualization
display_width = 800
display_height = 800
world_size = display_width

# Colors for visualization
blue = (0, 0, 255)
green = (0, 255, 0)
red = (255, 0, 0)
white = (255, 255, 255)
black = (0, 0, 0)

pygame.init()
screen = pygame.display.set_mode((display_width, display_height + 50))  # Extra space at the top
pygame.display.set_caption("Cow Localization Using RSSI and Grid Points")
clock = pygame.time.Clock()


# Flip the Y-coordinate to adjust for the inverted Y-axis
def flip_y(y, display_height):
    return display_height - y


def load_rssi_data(file_path):
    """
    Load RSSI data from the CSV file and store a list of RSSI values for each beacon.
    """
    rssi_data = {}
    df = pd.read_csv(file_path)

    if 'beaconId' in df.columns and 'rssi' in df.columns:
        for _, row in df.iterrows():
            try:
                beacon_id = int(row['beaconId'])
                rssi = float(row['rssi'])

                # Ignore RSSI values below -75 dBm
                if rssi >= -75:
                    if beacon_id not in rssi_data:
                        rssi_data[beacon_id] = []  # Initialize a list for each beacon
                    rssi_data[beacon_id].append(rssi)  # Append the RSSI value to the list
            except (ValueError, KeyError):
                print(f"Skipping malformed row: {row}")

    return rssi_data


def load_beacon_data(file_path):
    """
    Load beacon data from the given CSV file and create mappings.
    Returns:
        beacon_data: A dictionary mapping beaconId to (grid_x, grid_y) coordinates.
        beacon_id_to_number: A dictionary mapping beaconId to beacon_number.
    """
    beacon_data = {}
    beacon_id_to_number = {}

    df = pd.read_csv(file_path)
    if 'beaconId' in df.columns and 'beacon_number' in df.columns:
        for _, row in df.iterrows():
            beacon_id = int(row['beaconId'])
            beacon_number = int(row['beacon_number'])

            # Create a mapping from beaconId to beacon_number
            beacon_id_to_number[beacon_id] = beacon_number

            # Map beaconId to grid points if the beacon_number exists in grid_points
            if beacon_number in grid_points:
                grid_x, grid_y = grid_points[beacon_number]
                beacon_data[beacon_id] = (grid_x, grid_y)
            else:
                print(f"Beacon number {beacon_number} not found in grid_points.")
    else:
        print("Required columns 'beaconId' and 'beacon_number' not found in the CSV file.")

    return beacon_data, beacon_id_to_number


# Placeholder grid points based on your setup (adjust as needed)
grid_points = {
    1: (9, 13), 2: (9, 11), 3: (9, 9), 4: (9, 7),
    5: (7, 7), 6: (7, 9), 7: (7, 11), 8: (7, 14),
    18: (5.25, 15), 9: (5, 13), 10: (5, 11),
    11: (5, 9), 12: (5, 7),
    13: (3.5, 7), 14: (3.5, 9), 15: (3.5, 11), 16: (3.5, 13),
    17: (3.5, 15), 19: (7.5, 12.5)
}

# Scaling factor to match visualization dimensions
scale_factor = 50
scaled_grid_points = {key: (int(x * scale_factor), int(y * scale_factor)) for key, (x, y) in grid_points.items()}

# Calculate the range of the grid points
min_x = min(x for x, y in scaled_grid_points.values())
max_x = max(x for x, y in scaled_grid_points.values())
min_y = min(y for x, y in scaled_grid_points.values())
max_y = max(y for x, y in scaled_grid_points.values())

# Calculate the width and height of the plot
plot_width = max_x - min_x
plot_height = max_y - min_y

# Calculate the center offsets to move the plot to the center of the window horizontally and vertically
x_offset = (display_width - plot_width) // 2 - min_x
y_offset = ((display_height - plot_height) // 2 - min_y) + 300  # Adjust to move simulation down

# Define border padding
border_padding = 10



# Define Particle class
class Particle:
    def __init__(self):

        # Calculate the bounds for particle initialization based on beacon grid
        min_x = min(x for x, y in scaled_grid_points.values())
        max_x = max(x for x, y in scaled_grid_points.values())
        min_y = min(y for x, y in scaled_grid_points.values())
        max_y = max(y for x, y in scaled_grid_points.values())

        # Randomize particle's position within the beacon grid
        self.x = random.uniform(min_x, max_x)
        self.y = random.uniform(min_y, max_y)
        self.sense_noise = 0.3  # Reduced noise for better accuracy
    def set_noise(self, s_noise):
        self.sense_noise = s_noise

    def move(self):
        # Particles move randomly
        self.x += random.gauss(0, 2)
        self.y += random.gauss(0, 2)

        # Ensure particles stay within the grid boundaries
        self.x = max(min_x, min(self.x, max_x))
        self.y = max(min_y, min(self.y, max_y))

    def sense_signal_strength(self, beacons, rssi_data):
        """
        Calculate signal strength based on distance to beacons.
        """
        signal_strengths = {}
        n = 2  # Path-loss exponent
        A = -60  # Signal strength at 1 meter

        for beacon_id, (bx, by) in beacons.items():
            if beacon_id not in rssi_data:
                continue
            distance = sqrt((self.x - bx) ** 2 + (self.y - by) ** 2)
            if distance == 0:
                distance = 0.01
            signal_strengths[beacon_id] = A - 10 * n * log10(distance)
        return signal_strengths


def find_highest_proximity_rssi(rssi_data, window_size=5):
    """
    Find the beacon with the strongest average RSSI.
    """
    if not rssi_data:
        print("No RSSI data available.")
        return None, None

    # Compute the average of the last 'window_size' RSSI values
    beacon_averages = {
        beacon_id: np.mean(rssi_vals[-window_size:])
        for beacon_id, rssi_vals in rssi_data.items() if len(rssi_vals) >= window_size
    }

    # Find the beacon with the highest RSSI
    if beacon_averages:
        highest_proximity_beacon = max(beacon_averages, key=beacon_averages.get)
        highest_rssi = beacon_averages[highest_proximity_beacon]
        return highest_proximity_beacon, highest_rssi
    else:
        print("Not enough RSSI data to compute averages.")
        return None, None

# Initialize particles
num_particles = 500
particles = [Particle() for _ in range(num_particles)]
for particle in particles:
    particle.set_noise(0.3)  # Reducing measurement noise to focus on proximity
def visualize_particles(stage):
    screen.fill(white)

    # Draw legend in the top-right corner
    legend_x = display_width - 300  # X-coordinate for the legend (top-right)
    legend_y = 30  # Y-coordinate for the legend (top-right padding)

    font = pygame.font.Font(None, 20)
    font_small = pygame.font.Font(None, 24)
    pygame.draw.circle(screen, blue, (legend_x, legend_y + 20), 10)
    pygame.draw.circle(screen, green, (legend_x, legend_y + 50), 5)
    pygame.draw.circle(screen, red, (legend_x, legend_y + 80), 10)

    legend_blue = font.render("Beacon (Blue)", True, black)
    screen.blit(legend_blue, (legend_x + 20, legend_y + 10))

    legend_green = font.render("Particles (Green)", True, black)
    screen.blit(legend_green, (legend_x + 20, legend_y + 40))

    legend_red = font.render("Estimated Pos. (Red)", True, black)
    screen.blit(legend_red, (legend_x + 20, legend_y + 70))

    # Visualize each stage
    for beacon_id, (x, y) in scaled_grid_points.items():
        flipped_y = flip_y(y, display_height)

        # Map beaconId to beacon_number for labeling
        beacon_number = beacon_id_to_number.get(beacon_id, "Unknown")
        pygame.draw.circle(screen, blue, (int(x + x_offset), int(flipped_y + y_offset)), 10)
        font = pygame.font.Font(None, 24)
        label = font.render(f"B{beacon_id}", True, black)
        screen.blit(label, (x + x_offset + 10, flipped_y + y_offset - 10))

    # Draw particles with flipped Y-coordinates
    for particle in particles:
        flipped_y = flip_y(particle.y, display_height)
        pygame.draw.circle(screen, green, (int(particle.x + x_offset), int(flipped_y + y_offset)), 3)

    # Display stage
    label = font.render(f"Stage: {stage}", True, black)
    screen.blit(label, (150, 100))

    pygame.display.update()
    save_frame(stage)
    time.sleep(2)  # Add delay for each stage to visualize the process


# Load RSSI and beacon data
rssi_data = load_rssi_data("data/beacon_data_cow3at13_06to13_09.csv")
beacon_data, beacon_id_to_number = load_beacon_data("data/location_beacon.csv")

# Filter grid points to match loaded RSSI data
filtered_grid_points = {k: v for k, v in scaled_grid_points.items() if k in rssi_data.keys()}


# Step 1: Proximity Estimation Analysis with Noise Impact
noise_levels = [0.1, 0.5, 2.0, 5.0, 10.0, 20.0, 30.0]
incorrect_estimates = []

true_proximity_beacon_number = 7  # Beacon number
true_proximity_beacon_id = next(
    (beacon_id for beacon_id, number in beacon_id_to_number.items() if number == true_proximity_beacon_number), None
)

if true_proximity_beacon_id is None:
    raise ValueError(f"Beacon number {true_proximity_beacon_number} is not found in beacon_id_to_number mapping.")

# Analyze noise impact
for noise_level in noise_levels:
    print(f"\nTesting noise level: {noise_level}")

    # Add noise to the RSSI data
    rssi_data_with_noise = {}
    for beacon_id, rssi_list in rssi_data.items():
        noisy_rssi_list = [rssi + np.random.normal(0, noise_level) for rssi in rssi_list]
        rssi_data_with_noise[beacon_id] = noisy_rssi_list

    # Find the highest proximity beacon
    highest_proximity_beacon, highest_rssi = find_highest_proximity_rssi(rssi_data_with_noise)

    # Compare against the true proximity beacon ID
    is_incorrect = highest_proximity_beacon != true_proximity_beacon_id
    incorrect_estimates.append(is_incorrect)

    # Debug output
    print(f"Highest Proximity Beacon: {highest_proximity_beacon}, RSSI: {highest_rssi}")
    print(f"Is incorrect: {is_incorrect}")
# Step 2: Plot Noise Impact on Proximity Estimation
plt.figure(figsize=(8, 6))
plt.plot(noise_levels, incorrect_estimates, marker='o', color='blue', label='Incorrect Estimates')
plt.title('Noise Impact on Proximity Estimation')
plt.xlabel('Noise Level (dB)')
plt.ylabel('Incorrect Estimation (True=1, False=0)')
plt.grid(True)
plt.legend(loc='lower right')
plt.show()


def resample_particles(particles, weights):
    """Resample particles based on their weights."""
    new_particles = []
    indices = np.random.choice(len(particles), len(particles), p=weights)
    for i in indices:
        particle = Particle()
        particle.x = particles[i].x + random.gauss(0, 2)
        particle.y = particles[i].y + random.gauss(0, 2)
        new_particles.append(particle)
    return new_particles


def render_proximity_stage(stage, beaconId, beacon_number,cycle_time):
    """Render the final stage with estimated position and the closest beacon."""
    # Clear the screen to white
    screen.fill(white)

    # Draw legend in the top-right corner
    legend_x = display_width - 300  # X-coordinate for the legend (top-right)
    legend_y = 30  # Y-coordinate for the legend (top-right padding)

    font = pygame.font.Font(None, 20)
    font_small = pygame.font.Font(None, 24)
    pygame.draw.circle(screen, blue, (legend_x, legend_y + 20), 10)
    pygame.draw.circle(screen, green, (legend_x, legend_y + 50), 5)
    pygame.draw.circle(screen, red, (legend_x, legend_y + 80), 10)

    legend_blue = font.render("Beacon (Blue)", True, black)
    screen.blit(legend_blue, (legend_x + 20, legend_y + 10))

    legend_green = font.render("Particles (Green)", True, black)
    screen.blit(legend_green, (legend_x + 20, legend_y + 40))

    legend_red = font.render("Estimated Pos. (Red)", True, black)
    screen.blit(legend_red, (legend_x + 20, legend_y + 70))

    # Draw grid points (beacons) and their labels
    for beacon_id, (x, y) in scaled_grid_points.items():
        flipped_y = flip_y(y, display_height)
        # Draw each beacon as a blue circle
        pygame.draw.circle(screen, blue, (int(x + x_offset), int(flipped_y + y_offset)), 10)
        # Draw beacon label (small font)
        label = font.render(f"B{beacon_id}", True, black)
        screen.blit(label, (x + x_offset + 10, flipped_y + y_offset - 10))

        # Draw final particles in green, clustered near the proximity beacon
        if highest_proximity_beacon_number in grid_points:
            bx, by = grid_points[highest_proximity_beacon_number]
            bx_scaled, by_scaled = int(bx * scale_factor), int(by * scale_factor)

            # Generate clustered particles around the proximity beacon
            clustered_particles = []
            for _ in range(len(particles)):
                particle = Particle()
                particle.x = random.gauss(bx_scaled, 5)  # Cluster around the beacon with small spread
                particle.y = random.gauss(by_scaled, 5)  # Adjust spread as needed
                particle.x = max(min_x, min(particle.x, max_x))  # Keep within grid boundaries
                particle.y = max(min_y, min(particle.y, max_y))  # Keep within grid boundaries
                clustered_particles.append(particle)

            # Draw clustered particles
            for particle in clustered_particles:
                flipped_y = flip_y(particle.y, display_height)
                pygame.draw.circle(screen, green, (int(particle.x + x_offset), int(flipped_y + y_offset)), 3)

        # Highlight the proximity-based beacon based on the beacon_number
        if beacon_number in grid_points:
            bx, by = grid_points[beacon_number]
            bx_scaled, by_scaled = int(bx * scale_factor), int(by * scale_factor)
            flipped_y = flip_y(by_scaled, display_height)
            pygame.draw.circle(screen, red, (int(bx_scaled + x_offset), int(flipped_y + y_offset)), 10)

    # Display stage (Final Estimated Proximity)
    stage_label = font_small.render(f"Stage: {stage}", True, black)
    screen.blit(stage_label, (150, 100))

    # Display the closest beacon information using the small font
    closest_beacon_label = font_small.render(f"Closest Beacon: {beaconId}, B{beacon_number}", True, black)
    screen.blit(closest_beacon_label, (150, 140))

    # Display the time taken for the current cycle
    time_label = font_small.render(f"Cycle Time: {cycle_time:.4f} seconds", True, black)
    screen.blit(time_label, (150, 650))  # Display at (150, 180) coordinates

    # Display the number of particles
    particles_label = font_small.render(f"Number of Particles: {num_particles}", True, black)
    screen.blit(particles_label, (150, 70))  # Display below time label

    pygame.display.update()
    save_frame(stage)


# Start the visualization
visualize_particles("Initial Distribution")
import time

# Measure time taken for one localization cycle
start_time = time.time()
# Particle movement
for particle in particles:
    particle.move()
visualize_particles("Movement")

# Weight calculation
measured_signal_strengths = {beacon: np.mean(rssi) for beacon, rssi in rssi_data.items()}
weights = []
for particle in particles:
    predicted_signal_strengths = particle.sense_signal_strength(scaled_grid_points, rssi_data)
    # Calculate weight
    weight = sum(
        (measured_signal_strengths[beacon] - predicted_signal_strengths[beacon]) ** 2
        for beacon in measured_signal_strengths
        if beacon in predicted_signal_strengths
    )

    weights.append(1.0 / (weight + 1e-6))
weights = np.array(weights)
weights /= weights.sum()
visualize_particles("Weight Calculation")

# Resample particles
particles = resample_particles(particles, weights)
visualize_particles("Resampling")

# Find the highest proximity beacon
highest_proximity_beacon, highest_rssi = find_highest_proximity_rssi(rssi_data)

highest_proximity_beacon_number = beacon_id_to_number.get(highest_proximity_beacon, "Unknown")

print(f"Highest Proximity Beacon: {highest_proximity_beacon} (Number: {highest_proximity_beacon_number}), RSSI: {highest_rssi}")

# End the time measurement after the final stage has been rendered
end_time = time.time()
cycle_time = end_time - start_time  # Calculate the cycle time

render_proximity_stage("Proximity-Based Result", highest_proximity_beacon, highest_proximity_beacon_number,cycle_time)
# Save the frame for Stage 5
save_frame("Final Estimated Position")

# End the time measurement after the final stage has been rendered
end_time = time.time()
# Allow time to view the final result
time.sleep(2)
# Calculate the time taken for this cycle
cycle_time = end_time - start_time
# Store cycle time for performance evaluation
performance_data = {
    'particle_count': num_particles,
    'cycle_time': cycle_time
}
print(f"Time taken for one localization cycle: {cycle_time:.4f} seconds")
pygame.quit()

# Step 3: Visualization of the Correct Proximity Beacon (Optional)
highest_proximity_beacon, highest_rssi = find_highest_proximity_rssi(rssi_data)
highest_proximity_beacon_number = beacon_id_to_number.get(highest_proximity_beacon, "Unknown")

print(f"Highest Proximity Beacon: {highest_proximity_beacon} (Number: {highest_proximity_beacon_number}), RSSI: {highest_rssi}")
