import smplx
import torch
import os

def test_load_smplx():
    model_dir = "data/models/smpl"
    print(f"Loading SMPL-X from {model_dir}...")
    try:
        model = smplx.create(model_dir, model_type='smplx', gender='neutral', ext='pkl')
        print("✅ SMPL-X model loaded successfully!")
        print(f"Number of vertices: {model.v_template.shape[0]}")
    except Exception as e:
        print(f"❌ Failed to load SMPL-X: {e}")

if __name__ == "__main__":
    test_load_smplx()
