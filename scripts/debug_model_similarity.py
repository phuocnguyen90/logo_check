
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from logo_similarity.embeddings.efficientnet import EfficientNetEmbedder
from logo_similarity.config import paths, settings
import matplotlib.pyplot as plt
import os

def debug_similarity():
    # 1. Load Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model = EfficientNetEmbedder().to(device)
    
    # Try to load latest checkpoint
    ckpt_path = paths.CHECKPOINTS_DIR / "latest.pth"
    if not ckpt_path.exists():
        ckpt_path = paths.CHECKPOINTS_DIR / "best_model.pth"
    
    if ckpt_path.exists():
        print(f"Loading checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        try:
            state_dict = checkpoint['model_state_dict']
            # Clean generic MoCo prefixes if present
            clean_state = {}
            for k, v in state_dict.items():
                if k.startswith('encoder_q.'):
                    clean_state[k.replace('encoder_q.', '')] = v
                else:
                    clean_state[k] = v
            model.load_state_dict(clean_state, strict=False)
            print("Checkpoint loaded successfully.")
        except Exception as e:
            print(f"Failed to load state dict: {e}")
    else:
        print("No checkpoint found! Using random initialization (Baseline).")

    model.eval()
    
    # 2. Helper to get embedding
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    def get_emb(img_path):
        img = Image.open(img_path).convert('RGB')
        t = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model(t)
            return F.normalize(emb, dim=1)

    # 3. Pick an image
    # Get first image from raw dataset
    imgs = sorted(list((paths.RAW_DATASET_DIR / "images").glob("*.jpg")))
    if not imgs:
         imgs = sorted(list((paths.RAW_DATASET_DIR).glob("*.jpg")))
         
    if not imgs:
        print("No images found to test.")
        return

    img_a_path = imgs[0]
    img_b_path = imgs[1] # Different image
    
    print(f"Image A: {img_a_path.name}")
    print(f"Image B: {img_b_path.name}")

    # 4. Compute Similarities
    # A vs A (Self)
    emb_a = get_emb(img_a_path)
    score_self = torch.matmul(emb_a, emb_a.T).item()
    
    # A vs A_augmented (Simulate Instance Invariance)
    # Augment: CenterCrop + ColorJitter
    aug_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(180), # aggressive crop
        transforms.ColorJitter(brightness=0.5),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_a_pil = Image.open(img_a_path).convert('RGB')
    t_aug = aug_transform(img_a_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        emb_a_aug = F.normalize(model(t_aug), dim=1)
        
    score_aug = torch.matmul(emb_a, emb_a_aug.T).item()
    
    # A vs B (Negative)
    emb_b = get_emb(img_b_path)
    score_neg = torch.matmul(emb_a, emb_b.T).item()
    
    print("-" * 30)
    print(f"Similarity Scores (Dot Product):")
    print(f"Self (A vs A):      {score_self:.4f} (Control: Should be 1.0)")
    print(f"Augmented (A vs A'): {score_aug:.4f} (Goal: High, e.g. > 0.8)")
    print(f"Negative (A vs B):   {score_neg:.4f} (Goal: Low, e.g. < 0.3)")
    print("-" * 30)
    
    if score_aug > 0.8 and score_neg < 0.5:
        print("✅ Model has learned INSTANCE INVARIANCE.")
        print("It recognizes the same image even if modified.")
    elif score_aug < 0.5:
        print("❌ Model failed Instance Invariance.")
        print("It thinks the augmented version is a different image.")
    elif score_neg > 0.8:
        print("⚠️ Model Collapse?")
        print("It thinks different images are the same.")
    else:
        print("UNKNOWN state. Analysis needed.")

if __name__ == "__main__":
    debug_similarity()
