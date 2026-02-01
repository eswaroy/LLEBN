# Save as: check_embeddings.py

import torch
import numpy as np
import pickle
from models.siamese_network import SiameseNetwork
from data_generation.synthetic_human import SyntheticHumanGenerator
from feature_extraction.mouse_features import MouseFeatureExtractor

print("="*70)
print("EMBEDDING COLLAPSE DIAGNOSTIC")
print("="*70)

# Load model
checkpoint = torch.load('training/checkpoints/best_model.pth', map_location='cpu', weights_only=False)
with open('training/checkpoints/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

model = SiameseNetwork(input_dim=scaler.n_features_in_, embedding_dim=128, dropout=0.0)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"\nModel trained to epoch: {checkpoint.get('epoch', 'N/A')}")
print(f"Final val loss: {checkpoint.get('val_loss', 'N/A'):.6f}")

# Generate 5 DIFFERENT users
print("\nGenerating 5 different users...")
extractor = MouseFeatureExtractor(debug=False)
embeddings_list = []

for user_id in [100, 200, 300, 400, 500]:
    gen = SyntheticHumanGenerator(user_id=user_id, verbose=False)
    traj = gen.generate_mouse_trajectory()
    features = extractor.extract_features(traj)
    feature_vec = np.array(list(features.values())).reshape(1, -1)
    feature_vec = scaler.transform(feature_vec)
    
    with torch.no_grad():
        emb = model(torch.FloatTensor(feature_vec))
    
    embeddings_list.append(emb.numpy()[0])
    print(f"User {user_id}: embedding norm = {np.linalg.norm(emb.numpy()[0]):.6f}")

embeddings = np.array(embeddings_list)

print("\n" + "="*70)
print("PAIRWISE COSINE SIMILARITIES")
print("="*70)
for i in range(5):
    for j in range(i+1, 5):
        sim = np.dot(embeddings[i], embeddings[j])
        sim_01 = (sim + 1) / 2
        print(f"User {[100,200,300,400,500][i]} vs User {[100,200,300,400,500][j]}: {sim_01:.6f}")

print("\n" + "="*70)
print("EMBEDDING STATISTICS")
print("="*70)
print(f"Mean absolute value: {np.abs(embeddings).mean():.6f}")
print(f"Std deviation: {embeddings.std():.6f}")
print(f"Min value: {embeddings.min():.6f}")
print(f"Max value: {embeddings.max():.6f}")

# Check if all embeddings are identical
std_per_dim = embeddings.std(axis=0)
print(f"\nStd per dimension (should be > 0.01):")
print(f"  Mean: {std_per_dim.mean():.6f}")
print(f"  Max: {std_per_dim.max():.6f}")
print(f"  Dims with std < 0.001: {(std_per_dim < 0.001).sum()} / 128")

if std_per_dim.max() < 0.01:
    print("\n❌ CRITICAL: All embeddings are IDENTICAL!")
    print("The model collapsed during training.")
    print("\nMost likely cause:")
    print("1. Model outputs are all zeros before normalization")
    print("2. Batch normalization is in eval mode during training")
    print("3. Learning rate too high causing instability")
else:
    print("\n✓ Embeddings show some diversity")

# Check actual output before normalization
print("\n" + "="*70)
print("RAW EMBEDDINGS (before L2 norm)")
print("="*70)

# Temporarily remove normalization
class TestModel(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.encoder = base_model.encoder if hasattr(base_model, 'encoder') else base_model
    
    def forward(self, x):
        # Get embeddings WITHOUT L2 normalization
        for i, layer in enumerate(self.encoder.children()):
            x = layer(x)
        return x

test_model = TestModel(model)
test_model.eval()

user_id = 100
gen = SyntheticHumanGenerator(user_id=user_id, verbose=False)
traj = gen.generate_mouse_trajectory()
features = extractor.extract_features(traj)
feature_vec = np.array(list(features.values())).reshape(1, -1)
feature_vec = scaler.transform(feature_vec)

with torch.no_grad():
    raw_emb = test_model(torch.FloatTensor(feature_vec))

print(f"Raw embedding (before norm):")
print(f"  Shape: {raw_emb.shape}")
print(f"  Mean: {raw_emb.mean().item():.6f}")
print(f"  Std: {raw_emb.std().item():.6f}")
print(f"  Norm: {torch.norm(raw_emb).item():.6f}")
print(f"  Sample values: {raw_emb[0, :5].numpy()}")

if torch.norm(raw_emb).item() < 0.1:
    print("\n❌ RAW embeddings are near ZERO!")
    print("Model is outputting zeros → L2 norm makes them all identical")
