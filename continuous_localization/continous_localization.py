import os
import random
import numpy as np
import pygame
from math import exp, sqrt, log10
import pandas as pd
from datetime import datetime, timedelta
import time

# Set up directories and constants
save_dir = "simulation_frames/moving_particles/"
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
screen = pygame.display.set_mode((display_width, display_height + 50))
pygame.display.set_caption("Cow Localization Using RSSI and Grid Points")
clock = pygame.time.Clock()

# Define grid points for beacons and scaling
grid_points = {
    1: (9, 13), 2: (9, 11), 3: (9, 9), 4: (9, 7),
    5: (7, 7), 6: (7, 9), 7: (7, 11), 8: (7, 14),
    18: (5.25, 15), 9: (5, 13), 10: (5, 11),
    11: (5, 9), 12: (5, 7),
    13: (3.5, 7), 14: (3.5, 9), 15: (3.5, 11), 16: (3.5, 13),
    17: (3.5, 15), 19: (7.5, 12.5)
}
scale_factor = 50
scaled_grid_points = {key: (int(x * scale_factor), int(y * scale_factor)) for key, (x, y) in grid_points.items()}

# Define boundary variables based on grid points
min_x = min(x for x, y in scaled_grid_points.values())
max_x = max(x for x, y in scaled_grid_points.values())
min_y = min(y for x, y in scaled_grid_points.values())
max_y = max(y for x, y in scaled_grid_points.values())

# Set a consistent random seed for reproducibility
random.seed(42)
np.random.seed(42)


# Helper functions
def flip_y(y, display_height):
    return display_height - y


def load_rssi_data(file_path):
    rssi_data = []
    df = pd.read_csv(file_path)
    if 'beaconId' in df.columns and 'rssi' in df.columns and 'timestamp' in df.columns:
        for _, row in df.iterrows():
            try:
                beacon_id = int(row['beaconId'])
                rssi = float(row['rssi'])
                timestamp = pd.to_datetime(row['timestamp'])
                # Adjusted to include all RSSI values between -65 and 0 dBm
                if -75 <= rssi <= 0:
                    rssi_data.append((timestamp, beacon_id, rssi))
            except (ValueError, KeyError):
                print(f"Skipping malformed row: {row}")
    return rssi_data



def load_beacon_data(file_path):
    beacon_data = {}
    beacon_id_to_number = {}
    df = pd.read_csv(file_path)
    if 'beaconId' in df.columns and 'beacon_number' in df.columns:
        for _, row in df.iterrows():
            beacon_id = int(row['beaconId'])
            beacon_number = int(row['beacon_number'])
            beacon_id_to_number[beacon_id] = beacon_number  # Add the mapping
            if beacon_number in grid_points:
                grid_x, grid_y = grid_points[beacon_number]
                beacon_data[beacon_id] = (grid_x * scale_factor, grid_y * scale_factor)
    return beacon_data, beacon_id_to_number

class Particle:
    def __init__(self):
        self.x = random.uniform(min_x, max_x)
        self.y = random.uniform(min_y, max_y)
        self.sense_noise = 0.5  # Increased noise for exploratory movement

    def set_noise(self, s_noise):
        self.sense_noise = s_noise

    def move(self):
        # More movement noise to allow particles to explore multiple beacons
        self.x += random.gauss(0, 5)
        self.y += random.gauss(0, 5)
        self.x = max(min_x, min(self.x, max_x))
        self.y = max(min_y, min(self.y, max_y))

    def sense_signal_strength(self, beacons, rssi_data):
        signal_strengths = {}
        n = 2
        A = -60
        for beacon_id, (bx, by) in beacons.items():
            if beacon_id not in rssi_data:
                continue
            distance = sqrt((self.x - bx) ** 2 + (self.y - by) ** 2)
            distance = max(distance, 0.01)  # Avoid zero distance
            predicted_signal_strength = A - (10 * n * log10(distance))
            predicted_signal_strength += random.gauss(0.0, self.sense_noise)
            signal_strengths[beacon_id] = predicted_signal_strength
        return signal_strengths


def find_top_beacons(measured_signal_strengths, top_n=4):
    sorted_beacons = sorted(measured_signal_strengths.items(), key=lambda x: x[1], reverse=True)
    return [beacon_id for beacon_id, _ in sorted_beacons[:top_n]]


def measurement_prob_with_multiple_beacons(particle, measured_signal_strengths, top_beacons, beacons, rssi_data):
    predicted_signal_strengths = particle.sense_signal_strength(beacons, rssi_data)
    error = 1.0
    for beacon_id in top_beacons:
        if beacon_id not in predicted_signal_strengths:
            continue
        predicted_strength = predicted_signal_strengths[beacon_id]
        actual_strength = measured_signal_strengths[beacon_id]
        weight = exp(-0.5 * abs(actual_strength - predicted_strength))
        error *= weight
    return error


def estimate_position(particles, weights):
    x_estimate = sum(p.x * w for p, w in zip(particles, weights))
    y_estimate = sum(p.y * w for p, w in zip(particles, weights))
    return x_estimate, y_estimate


def resample_particles_final_stage(particles, weights, estimated_x, estimated_y):
    new_particles = []
    num_particles = len(particles)
    indices = np.random.choice(range(num_particles), size=num_particles, p=weights)
    for i in indices:
        new_particle = Particle()
        new_particle.x = random.gauss(estimated_x, 5)
        new_particle.y = random.gauss(estimated_y + 10, 5)
        new_particle.x = max(min_x, min(new_particle.x, max_x))
        new_particle.y = max(min_y, min(new_particle.y, max_y))
        new_particles.append(new_particle)
    return new_particles


# Initialize a list to store the cow's path
cow_path = []

def render_final_stage(estimated_x, estimated_y, particles, cycle_time=None, stage="Final Stage", final_delay=5):
    # Clear the screen
    screen.fill(white)

    # Calculate grid placement offsets
    plot_width = max_x - min_x
    plot_height = max_y - min_y
    x_offset = (display_width - plot_width) // 2 - min_x  # Center the grid horizontally
    y_offset = ((display_height - plot_height) // 2 - min_y) + 250  # Center vertically with a margin

    # Draw grid points (beacons) and labels
    font = pygame.font.Font(None, 18)
    for beacon_id, (x, y) in scaled_grid_points.items():
        flipped_y = flip_y(y, display_height)  # Adjust for Pygame's inverted Y-axis
        pygame.draw.circle(screen, blue, (int(x + x_offset), int(flipped_y + y_offset)), 10)  # Draw beacon
        label = font.render(f"B{beacon_id}", True, black)  # Beacon ID label
        screen.blit(label, (int(x + x_offset + 10), int(flipped_y + y_offset - 10)))  # Offset label from beacon

    # Draw particles
    for particle in particles:
        flipped_y = flip_y(particle.y, display_height)  # Adjust for Y-axis flip
        pygame.draw.circle(screen, green, (int(particle.x + x_offset), int(flipped_y + y_offset)), 3)

    # Draw estimated position
    estimated_flipped_y = flip_y(estimated_y, display_height)
    pygame.draw.circle(screen, red, (int(estimated_x + x_offset), int(estimated_flipped_y + y_offset)), 10)

    # Draw the cow's movement path
    if len(cow_path) > 1:
        for i in range(len(cow_path) - 1):
            start_x, start_y = cow_path[i]
            end_x, end_y = cow_path[i + 1]
            flipped_start_y = flip_y(start_y, display_height)
            flipped_end_y = flip_y(end_y, display_height)
            pygame.draw.line(
                screen,
                red,  # Line color (same as estimated position)
                (int(start_x + x_offset), int(flipped_start_y + y_offset)),
                (int(end_x + x_offset), int(flipped_end_y + y_offset)),
                2,  # Line thickness
            )

    # Draw legend in the top-right corner
    legend_x = display_width - 250  # Adjust legend position horizontally
    legend_y = 50  # Adjust legend position vertically
    font_legend = pygame.font.Font(None, 24)
    font_small = pygame.font.Font(None, 22)

    pygame.draw.circle(screen, blue, (legend_x, legend_y + 20), 10)  # Blue circle for beacons
    pygame.draw.circle(screen, green, (legend_x, legend_y + 50), 5)  # Green circle for particles
    pygame.draw.circle(screen, red, (legend_x, legend_y + 80), 10)  # Red circle for estimated position

    legend_beacon = font_small.render("Beacon (Blue)", True, black)
    screen.blit(legend_beacon, (legend_x + 20, legend_y + 10))

    legend_particle = font_small.render("Particles (Green)", True, black)
    screen.blit(legend_particle, (legend_x + 20, legend_y + 40))

    legend_estimated = font_small.render("Estimated Pos. (Red)", True, black)
    screen.blit(legend_estimated, (legend_x + 20, legend_y + 70))

    # Display the stage name
    stage_label = font_legend.render(f"Stage: {stage}", True, black)
    screen.blit(stage_label, (150, 70))

    # Display cycle time if provided
    if cycle_time is not None:
        cycle_time_label = font_legend.render(f"Cycle Time: {cycle_time:.2f} seconds", True, black)
        screen.blit(cycle_time_label, (150, 100))

    # Update display
    pygame.display.update()

    # Save the frame
    save_frame(stage)

    # Add delay for visualization
    if stage == "Total Simulation Time":
        time.sleep(final_delay)  # Longer delay for the final stage
    else:
        time.sleep(0.5)  # Short delay for intermediate stages


def run_localization_cycle(rssi_data, beacon_data, beacon_id_to_number, interval=timedelta(minutes=10)):
    global cow_path
    last_position = None
    last_timestamp = None
    timestamps = sorted(set(entry[0] for entry in rssi_data))

    # Start timing the entire simulation
    simulation_start_time = time.time()

    for timestamp in timestamps:
        if last_timestamp is None or timestamp >= last_timestamp + interval:
            # Filter relevant RSSI data for the current timestamp and threshold
            relevant_rssi = {entry[1]: entry[2] for entry in rssi_data if entry[0] == timestamp and entry[2] >= -75}

            # Find top beacons
            top_beacons = find_top_beacons(relevant_rssi)

            # Initialize and move particles
            particles = [Particle() for _ in range(500)]
            for particle in particles:
                particle.set_noise(0.5)
                particle.move()

            # Calculate weights
            weights = np.array([
                measurement_prob_with_multiple_beacons(particle, relevant_rssi, top_beacons, beacon_data, relevant_rssi)
                for particle in particles
            ])
            if weights.sum() == 0:
                weights.fill(1 / len(weights))
            else:
                weights /= weights.sum()

            # Estimate the position
            estimated_x, estimated_y = estimate_position(particles, weights)
            particles = resample_particles_final_stage(particles, weights, estimated_x, estimated_y)

            # Append the estimated position to the cow's path
            cow_path.append((estimated_x, estimated_y))

            # Smoothly transition to the new position
            if last_position:
                for i in range(10):
                    interpolated_x = last_position[0] + (estimated_x - last_position[0]) * (i / 10)
                    interpolated_y = last_position[1] + (estimated_y - last_position[1]) * (i / 10)
                    render_final_stage(interpolated_x, interpolated_y, particles, None, stage="Localization")
            else:
                render_final_stage(estimated_x, estimated_y, particles, None, stage="Localization")

            # Update last position and timestamp
            last_position = (estimated_x, estimated_y)
            last_timestamp = timestamp

    # End timing the entire simulation
    simulation_end_time = time.time()
    total_simulation_time = simulation_end_time - simulation_start_time

    # Render final stage showing total simulation time
    render_final_stage(
        estimated_x,
        estimated_y,
        particles,
        cycle_time=total_simulation_time,
        stage="Final Stage",
        final_delay=10  # Stay for 10 seconds
    )

    # Debug: Print total simulation time
    print(f"Total Simulation Time: {total_simulation_time:.2f} seconds")


# Load RSSI and beacon data
rssi_data = load_rssi_data('data/beacon_data_cow3_13pmto13_56pm.csv')
beacon_data, beacon_id_to_number = load_beacon_data('data/location_beacon.csv')

# Run localization simulation with intervals
run_localization_cycle(rssi_data, beacon_data, beacon_id_to_number, interval=timedelta(minutes=10))

pygame.quit()
