# """Test the verification system with FIXED user identity."""
# import sys
# import os
# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# import numpy as np
# import pandas as pd
# from inference.verify_user import HumanVerificationSystem
# from inference.reward_integration import (
#     RewardVerificationIntegration,
#     extract_embedding_from_interaction,
#     extract_features_from_interaction
# )
# from data_generation.synthetic_human import SyntheticHumanGenerator
# from data_generation.synthetic_bot import SyntheticBotGenerator

# print("Initializing verification system...")
# verifier = HumanVerificationSystem(
#     model_path='training/checkpoints/best_model.pth',
#     scaler_path='training/checkpoints/scaler.pkl'
# )

# reward_system = RewardVerificationIntegration(verifier)

# print("\nCreating user database (10 distinct users)...")
# database_embeddings = []
# for user_id in range(10):
#     # CRITICAL FIX: Each user gets unique profile
#     human_gen = SyntheticHumanGenerator(user_id=user_id, verbose=False)
#     traj = human_gen.generate_mouse_trajectory()
#     embedding = extract_embedding_from_interaction(traj, None, verifier)
#     database_embeddings.append(embedding)

# database_embeddings = np.array(database_embeddings)
# print(f"Database: {len(database_embeddings)} users")

# # TEST 1: NEW genuine human (user_id=999, not in database)
# print("\n" + "="*70)
# print("TEST 1: New Genuine Human User (ID=999)")
# print("="*70)
# new_human = SyntheticHumanGenerator(user_id=999, verbose=False)
# new_human_traj = new_human.generate_mouse_trajectory()
# new_human_embedding = extract_embedding_from_interaction(new_human_traj, None, verifier)
# new_human_features = extract_features_from_interaction(new_human_traj)

# result = reward_system.verify_and_allocate_reward(
#     user_id="new_user_999",
#     user_embedding=new_human_embedding,
#     user_features=new_human_features,
#     database_embeddings=database_embeddings,
#     base_reward=100.0
# )

# print(reward_system.get_user_summary(result))
# print(f"Explanation: {result['explanation']}")

# # TEST 2: Bot user
# print("\n" + "="*70)
# print("TEST 2: Bot User (Simple)")
# print("="*70)
# bot_gen = SyntheticBotGenerator(bot_id=1, verbose=False)
# bot_traj = bot_gen.generate_bot_mouse_trajectory(bot_sophistication='simple')
# bot_embedding = extract_embedding_from_interaction(bot_traj, None, verifier)
# bot_features = extract_features_from_interaction(bot_traj)

# result = reward_system.verify_and_allocate_reward(
#     user_id="bot_user_456",
#     user_embedding=bot_embedding,
#     user_features=bot_features,
#     database_embeddings=database_embeddings,
#     base_reward=100.0
# )

# print(reward_system.get_user_summary(result))
# print(f"Explanation: {result['explanation']}")
# if result.get('bot_detected'):
#     print(f"Bot reasons: {', '.join(result['bot_reasons'][:3])}")

# # TEST 3: Multi-account (SAME user as in database)
# print("\n" + "="*70)
# print("TEST 3: Multi-Account Detection (User ID=5, already in database)")
# print("="*70)
# # Use user_id=5 (which is in the database)
# same_user = SyntheticHumanGenerator(user_id=5, verbose=False)
# multi_traj = same_user.generate_mouse_trajectory()  # Different trajectory, same user
# multi_embedding = extract_embedding_from_interaction(multi_traj, None, verifier)
# multi_features = extract_features_from_interaction(multi_traj)

# result = reward_system.verify_and_allocate_reward(
#     user_id="multi_account_789",
#     user_embedding=multi_embedding,
#     user_features=multi_features,
#     database_embeddings=database_embeddings,
#     base_reward=100.0
# )

# print(reward_system.get_user_summary(result))
# print(f"Explanation: {result['explanation']}")

# print("\n" + "="*70)
# print("TESTING COMPLETE!")
# print("="*70)
"""
Test the verification system with PROPERLY DIVERSE users.
FIXED: Ensures database and test users are truly different.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from inference.verify_user import HumanVerificationSystem
from inference.reward_integration import (
    RewardVerificationIntegration,
    extract_embedding_from_interaction,
    extract_features_from_interaction
)
from data_generation.synthetic_human import SyntheticHumanGenerator
from data_generation.synthetic_bot import SyntheticBotGenerator
import re

print("Initializing verification system...")
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'training', 'checkpoints', 'best_model.pth')
scaler_path = os.path.join(base_dir, 'training', 'checkpoints', 'scaler.pkl')

verifier = HumanVerificationSystem(
    model_path=model_path,
    scaler_path=scaler_path
)

reward_system = RewardVerificationIntegration(verifier)

# Check for Real Data
real_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'balabit_raw')
if not os.path.exists(real_data_dir):
    alt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'Mouse-Dynamics-Challenge-master')
    if os.path.exists(alt_dir):
        real_data_dir = alt_dir
    else:
        real_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

print(f"Searching for data in: {real_data_dir}")
import glob
# Use os.walk for robustness and to match dataset_real.py logic
csv_files = []
for root, dirs, files in os.walk(real_data_dir):
    for file in files:
        if file.startswith("session_") or file.endswith(".csv"):
             if "public_labels" not in file:
                 csv_files.append(os.path.join(root, file))

real_data_available = len(csv_files) > 0
print(f"Found {len(csv_files)} potential session files.")

if real_data_available:
    print("\nUSING REAL WORLD DATA (BALABIT)")
    from training.dataset_real import load_balabit_dataset
    print("Loading real data...")
    # Load data without feature extraction (raw df) ?? 
    # Actually load_balabit_dataset returns features. We need raw trajectories for 'extract_embedding_from_interaction' which takes a DF.
    # We will implement a helper here or modify load_balabit_dataset?
    # Let's just load the CSVs directly here for testing.
    
    database_embeddings = []
    database_user_ids = []
    
    user_files = {}
    for f in csv_files:
        if 'public_labels' in f: continue
        
        # Try filename first
        match = re.search(r'user(\d+)', os.path.basename(f))
        
        # Try parent directory if filename fails
        if not match:
             parent_dir = os.path.basename(os.path.dirname(f))
             match = re.search(r'user(\d+)', parent_dir)
             
        if match:
            uid = int(match.group(1))
            if uid not in user_files: user_files[uid] = []
            user_files[uid].append(f)
            
    # Select first 7 users for database (leaving some for testing)
    all_users = sorted(list(user_files.keys()))
    selected_users = all_users[:7] if len(all_users) >= 7 else all_users
    print(f"Selected users for database: {selected_users} (Total available: {len(all_users)})")
    
    from training.dataset_real import load_balabit_session
    
    for user_id in selected_users:
        if len(user_files[user_id]) == 0: continue
        
        # Use first session as 'enrollment'
        session_file = user_files[user_id][0]
        # print(f"Enrollment file for user {user_id}: {session_file}") 
        
        try:
            traj = load_balabit_session(session_file)
            if traj is not None and len(traj) > 50:
                 embedding = extract_embedding_from_interaction(traj, None, verifier)
                 database_embeddings.append(embedding)
                 database_user_ids.append(user_id)
        except Exception as e:
            print(f"Error enrolling user {user_id}: {e}")
else:
    print("\nUSING SYNTHETIC DATA (FALLBACK)")
    # CRITICAL: Create database with SPECIFIC user IDs (0-6)
    print("\nCreating user database (7 distinct users with IDs 0-6)...")
    database_embeddings = []
    database_user_ids = list(range(7))

    for user_id in database_user_ids:
        # Create unique generator for THIS user
        human_gen = SyntheticHumanGenerator(user_id=user_id, verbose=False)
        traj = human_gen.generate_mouse_trajectory()
        embedding = extract_embedding_from_interaction(traj, None, verifier)
        database_embeddings.append(embedding)
        
        if user_id < 3:  # Debug first 3 users
            print(f"  User {user_id}: embedding norm = {np.linalg.norm(embedding):.4f}")

database_embeddings = np.array(database_embeddings)
print(f"Database: {len(database_embeddings)} users (IDs: {database_user_ids})")

# Verify database diversity
print("\nDatabase diversity check:")
for i in range(min(3, len(database_embeddings))):
    for j in range(i+1, min(3, len(database_embeddings))):
        sim = np.dot(database_embeddings[i], database_embeddings[j])
        sim_01 = (sim + 1) / 2
        print(f"  User {i} vs User {j}: similarity = {sim_01:.4f}")

# TEST 1: NEW genuine human (user_id=999, NOT in database)
print("\n" + "="*70)
print("TEST 1: New Genuine Human User")
print("="*70)

new_human_traj = None
if real_data_available:
    # Pick a user NOT in database (e.g. 8th user)
    if len(all_users) > 7:
        new_user_id = all_users[7]
        print(f"Using Real User {new_user_id} (not in DB)")
        session_file = user_files[new_user_id][0]
        new_human_traj = load_balabit_session(session_file)
    else:
        print("Not enough real users for Test 1. Using synthetic.")

if new_human_traj is None:
    print("Using Synthetic User 999")
    new_human = SyntheticHumanGenerator(user_id=999, verbose=True)
    new_human_traj = new_human.generate_mouse_trajectory()

new_human_embedding = extract_embedding_from_interaction(new_human_traj, None, verifier)
new_human_features = extract_features_from_interaction(new_human_traj)

print(f"New user embedding norm: {np.linalg.norm(new_human_embedding):.4f}")

# Check similarity to each database user
print("Similarities to database users:")
for i in range(min(5, len(database_embeddings))):
    sim = np.dot(new_human_embedding, database_embeddings[i])
    sim_01 = (sim + 1) / 2
    print(f"  vs User {i}: {sim_01:.4f}")

result = reward_system.verify_and_allocate_reward(
    user_id="new_user_999",
    user_embedding=new_human_embedding,
    user_features=new_human_features,
    database_embeddings=database_embeddings,
    base_reward=100.0
)

print("\n" + reward_system.get_user_summary(result))
print(f"Explanation: {result['explanation']}")

# TEST 2: Bot user
print("\n" + "="*70)
print("TEST 2: Bot User (Simple Bot)")
print("="*70)
bot_gen = SyntheticBotGenerator(bot_id=1, verbose=False)
bot_traj = bot_gen.generate_bot_mouse_trajectory(bot_sophistication='simple')
bot_embedding = extract_embedding_from_interaction(bot_traj, None, verifier)
bot_features = extract_features_from_interaction(bot_traj)

# DEBUG: Print bot features
print(f"Bot Features Debug: VelStd={bot_features.get('velocity_std', 0.0):.2f}, Eff={bot_features.get('path_efficiency', 0.0):.4f}")

print(f"Bot embedding norm: {np.linalg.norm(bot_embedding):.4f}")

result = reward_system.verify_and_allocate_reward(
    user_id="bot_user_456",
    user_embedding=bot_embedding,
    user_features=bot_features,
    database_embeddings=database_embeddings,
    base_reward=100.0
)

print(reward_system.get_user_summary(result))
print(f"Explanation: {result['explanation']}")
if result.get('bot_detected'):
    print(f"Bot reasons: {', '.join(result['bot_reasons'][:3])}")

# TEST 3: Multi-account (SAME user as in database)
print("\n" + "="*70)
print("TEST 3: Multi-Account Detection (User ALREADY in database)")
print("="*70)

multi_traj = None
target_db_idx = 5 

if real_data_available:
    # Use User at index 5 of database
    if len(database_user_ids) > target_db_idx:
        target_uid = database_user_ids[target_db_idx]
        print(f"Using Real User {target_uid} (from DB)")
        # Try to find a DIFFERENT session
        sessions = user_files[target_uid]
        if len(sessions) > 1:
            print("Using 2nd session")
            multi_traj = load_balabit_session(sessions[1])
        else:
            print("User has only 1 session. Using synthetic.")

if multi_traj is None:
    print("Using Synthetic User 5")
    same_user = SyntheticHumanGenerator(user_id=5, verbose=True)  # Same as database user 5
    multi_traj = same_user.generate_mouse_trajectory()  # Different trajectory, same user

multi_embedding = extract_embedding_from_interaction(multi_traj, None, verifier)
multi_features = extract_features_from_interaction(multi_traj)

print(f"Multi-account embedding norm: {np.linalg.norm(multi_embedding):.4f}")

# Check similarity to User 5 specifically
sim_to_user5 = np.dot(multi_embedding, database_embeddings[5])
sim_to_user5_01 = (sim_to_user5 + 1) / 2
print(f"Similarity to original User 5: {sim_to_user5_01:.4f}")

result = reward_system.verify_and_allocate_reward(
    user_id="multi_account_789",
    user_embedding=multi_embedding,
    user_features=multi_features,
    database_embeddings=database_embeddings,
    base_reward=100.0
)

print("\n" + reward_system.get_user_summary(result))
print(f"Explanation: {result['explanation']}")

# ADDITIONAL TEST: Check if model learned user identity
print("\n" + "="*70)
print("ADDITIONAL: Same User, Different Trajectories")
print("="*70)
print("Testing if User 5's two trajectories have high similarity...")

user5_gen = SyntheticHumanGenerator(user_id=5, verbose=False)
traj1 = user5_gen.generate_mouse_trajectory()
traj2 = user5_gen.generate_mouse_trajectory()

emb1 = extract_embedding_from_interaction(traj1, None, verifier)
emb2 = extract_embedding_from_interaction(traj2, None, verifier)

sim = np.dot(emb1, emb2)
sim_01 = (sim + 1) / 2
print(f"User 5 - Trajectory 1 vs Trajectory 2: {sim_01:.4f}")
print(f"Expected: > 0.85 (same user)")

print("\n" + "="*70)
print("TESTING COMPLETE!")
print("="*70)
print("\nInterpretation Guide:")
print("- TEST 1: Should show sim ~0.3-0.6 (different users)")
print("- TEST 2: Should detect bot (low velocity variance)")
print("- TEST 3: Should show sim ~0.85-0.95 (same user, different session)")
print("="*70)
