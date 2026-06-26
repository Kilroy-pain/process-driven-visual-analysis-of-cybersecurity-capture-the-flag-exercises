import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from datetime import datetime

# Simulated data generation for Capture the Flag (CTF) exercises
def generate_dummy_ctf_data(num_teams=5, num_events=100):
    np.random.seed(42)
    teams = np.random.choice(range(num_teams), num_events)
    timestamps = np.array([datetime.now().timestamp() + i * 60 for i in range(num_events)])
    scores = np.random.randint(0, 100, num_events)
    event_types = np.random.choice(['attack', 'defense', 'solve'], num_events)
    return teams, timestamps, scores, event_types

# Process mining: Temporal clustering of events
def temporal_clustering(timestamps, num_clusters=3):
    timestamps = np.array(timestamps).reshape(-1, 1)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(timestamps)
    return clusters, kmeans.cluster_centers_

# Visualization: Multivariate network and temporal view
def visualize_ctf_data(teams, timestamps, scores, event_types, clusters, cluster_centers):
    plt.figure(figsize=(12, 6))

    # Temporal view
    plt.subplot(1, 2, 1)
    plt.scatter(timestamps, scores, c=clusters, cmap='viridis', alpha=0.7)
    for center in cluster_centers:
        plt.axvline(center, color='red', linestyle='--', label='Cluster Center')
    plt.xlabel('Timestamp')
    plt.ylabel('Score')
    plt.title('Temporal View of Events')
    plt.legend()

    # Multivariate network visualization
    plt.subplot(1, 2, 2)
    unique_teams = np.unique(teams)
    for team in unique_teams:
        team_indices = np.where(teams == team)
        plt.scatter(scores[team_indices], team_indices[0], label=f'Team {team}', alpha=0.7)
    plt.xlabel('Score')
    plt.ylabel('Event Index')
    plt.title('Multivariate Network Visualization')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Main block to demonstrate functionality
if __name__ == '__main__':
    # Generate dummy data
    num_teams = 5
    num_events = 100
    teams, timestamps, scores, event_types = generate_dummy_ctf_data(num_teams, num_events)

    # Perform temporal clustering
    num_clusters = 3
    clusters, cluster_centers = temporal_clustering(timestamps, num_clusters)

    # Visualize the data
    visualize_ctf_data(teams, timestamps, scores, event_types, clusters, cluster_centers)