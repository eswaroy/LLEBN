# # Create this file: debug_model.py

# import torch
# import numpy as np
# import pickle
# from models.siamese_network import SiameseNetwork
# from models.model_config import ModelConfig

# print("Loading model...")
# checkpoint = torch.load('training/checkpoints/best_model.pth', map_location='cpu', weights_only=False)

# print("\n" + "="*70)
# print("MODEL CHECKPOINT INFO")
# print("="*70)
# print(f"Epoch: {checkpoint.get('epoch', 'N/A')}")
# print(f"Train Loss: {checkpoint.get('train_loss', 'N/A'):.6f}")
# print(f"Val Loss: {checkpoint.get('val_loss', 'N/A'):.6f}")

# # Load scaler
# with open('training/checkpoints/scaler.pkl', 'rb') as f:
#     scaler = pickle.load(f)

# print(f"\nScaler type: {type(scaler).__name__}")
# print(f"Input features: {scaler.n_features_in_}")

# # Load model
# config = ModelConfig()
# model = SiameseNetwork(
#     input_dim=scaler.n_features_in_,
#     embedding_dim=config.EMBEDDING_DIM,
#     dropout=0.0
# )
# model.load_state_dict(checkpoint['model_state_dict'])
# model.eval()

# print("\n" + "="*70)
# print("MODEL TEST WITH RANDOM INPUTS")
# print("="*70)

# # Test with 5 random inputs
# test_features = np.random.randn(5, scaler.n_features_in_).astype(np.float32)
# test_features_scaled = scaler.transform(test_features)
# test_tensor = torch.FloatTensor(test_features_scaled)

# with torch.no_grad():
#     embeddings = model(test_tensor)

# print(f"Embeddings shape: {embeddings.shape}")
# print(f"\nEmbedding norms:")
# for i, emb in enumerate(embeddings):
#     norm = torch.norm(emb).item()
#     print(f"  Sample {i+1}: {norm:.6f}")

# print(f"\nPairwise cosine similarities:")
# for i in range(5):
#     for j in range(i+1, 5):
#         sim = torch.dot(embeddings[i], embeddings[j]).item()
#         sim_normalized = (sim + 1) / 2  # Map to [0, 1]
#         print(f"  Sample {i+1} vs {j+1}: {sim_normalized:.6f}")

# print("\n" + "="*70)
# print("EMBEDDING STATISTICS")
# print("="*70)
# print(f"Mean: {embeddings.mean(dim=0).abs().mean().item():.6f}")
# print(f"Std: {embeddings.std(dim=0).mean().item():.6f}")
# print(f"Min: {embeddings.min().item():.6f}")
# print(f"Max: {embeddings.max().item():.6f}")

# # Check if model is just outputting zeros
# if embeddings.abs().max().item() < 0.01:
#     print("\n❌ ERROR: Model outputs near-zero embeddings!")
#     print("The model did not train properly.")
# else:
#     print("\n✓ Model appears to be producing valid embeddings")
# Save as: debug_diversity.py

import numpy as np
from data_generation.synthetic_human import SyntheticHumanGenerator
from feature_extraction.mouse_features import MouseFeatureExtractor

print("Testing user diversity in synthetic data...\n")

extractor = MouseFeatureExtractor(debug=False)

# Generate 5 users
user_ids = [0, 1, 2, 999, 1000]
features_list = []

for user_id in user_ids:
    gen = SyntheticHumanGenerator(user_id=user_id, verbose=True)
    traj = gen.generate_mouse_trajectory()
    features = extractor.extract_features(traj)
    features_list.append(features)
    
    print(f"\nUser {user_id} features:")
    print(f"  velocity_mean: {features['velocity_mean']:.1f}")
    print(f"  velocity_std: {features['velocity_std']:.1f}")
    print(f"  num_pauses: {features['num_pauses']:.0f}")
    print(f"  path_efficiency: {features['path_efficiency']:.3f}")

print("\n" + "="*70)
print("FEATURE DIVERSITY CHECK")
print("="*70)

# Check if features are actually different
for key in ['velocity_mean', 'velocity_std', 'num_pauses']:
    values = [f[key] for f in features_list]
    print(f"{key}: min={min(values):.1f}, max={max(values):.1f}, std={np.std(values):.1f}")
