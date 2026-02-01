
import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_generation.synthetic_bot import SyntheticBotGenerator
from data_generation.synthetic_human import SyntheticHumanGenerator
from feature_extraction.mouse_features import MouseFeatureExtractor

def debug_bot():
    print("="*60)
    print("DEBUGGING BOT GENERATOR")
    print("="*60)
    
    bot_gen = SyntheticBotGenerator(bot_id=1, verbose=True)
    traj = bot_gen.generate_bot_mouse_trajectory(bot_sophistication='simple', num_points=200)
    
    print("\nBot Trajectory Head:")
    print(traj.head())
    print("\nBot Trajectory Tail:")
    print(traj.tail())
    
    # Check if x/y change
    dx = np.diff(traj['x'])
    dy = np.diff(traj['y'])
    dist = np.sqrt(dx**2 + dy**2)
    print(f"\nStep distances summary: Min={dist.min():.4f}, Max={dist.max():.4f}, Sum={dist.sum():.4f}")
    
    euclidean = np.sqrt((traj['x'].iloc[-1] - traj['x'].iloc[0])**2 + (traj['y'].iloc[-1] - traj['y'].iloc[0])**2)
    print(f"Euclidean Distance: {euclidean:.4f}")
    
    extractor = MouseFeatureExtractor(debug=True)
    features = extractor.extract_features(traj)
    
    print("\nExtracted Bot Features:")
    print(f"  velocity_std: {features['velocity_std']:.4f}")
    print(f"  path_efficiency: {features['path_efficiency']:.4f} (Euclidean / PathLength)")
    print(f"  path_length: {features['path_length']:.4f}")
    print(f"  duration: {features['duration']:.4f}")

def debug_human_randomness():
    print("\n" + "="*60)
    print("DEBUGGING HUMAN RANDOMNESS")
    print("="*60)
    
    user_id = 5
    print(f"Generating two trajectories for User {user_id}...")
    
    gen = SyntheticHumanGenerator(user_id=user_id, verbose=False)
    
    # Trajectory 1
    traj1 = gen.generate_mouse_trajectory(num_points=100)
    
    # Trajectory 2
    traj2 = gen.generate_mouse_trajectory(num_points=100)
    
    print("\nTrajectory 1 Head:")
    print(traj1.head(3))
    print("\nTrajectory 2 Head:")
    print(traj2.head(3))
    
    if traj1.equals(traj2):
        print("\n[FAIL] Trajectories are IDENTICAL!")
    else:
        print("\n[PASS] Trajectories are different.")
        
    # Check start points
    p1 = (traj1['x'].iloc[0], traj1['y'].iloc[0])
    p2 = (traj2['x'].iloc[0], traj2['y'].iloc[0])
    print(f"Start Point 1: {p1}")
    print(f"Start Point 2: {p2}")

if __name__ == "__main__":
    debug_bot()
    debug_human_randomness()
