import torch
import numpy as np

def inspect_checkpoint(path):
    # Allow unpickling numpy scalars
    try:
        import torch.serialization
        torch.serialization.add_safe_globals([
            np.core.multiarray.scalar,
            np._core.multiarray.scalar if hasattr(np, '_core') else np.core.multiarray.scalar,
            np.float64, np.int64
        ])
    except:
        pass
        
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    print(f"Keys in checkpoint: {checkpoint.keys()}")
    
    state_dict = checkpoint['model_state_dict']
    keys = sorted(state_dict.keys())
    print(f"\nNumber of keys in state_dict: {len(keys)}")
    print("\nFirst 10 keys:")
    for k in keys[:10]:
        print(f"  {k}")
        
    print("\nLast 10 keys:")
    for k in keys[-10:]:
        print(f"  {k}")
        
    # Check for encoder_q vs model
    prefixes = set()
    for k in keys:
        prefixes.add(k.split('.')[0])
    print(f"\nRoot prefixes: {prefixes}")

if __name__ == "__main__":
    import sys
    inspect_checkpoint(sys.argv[1])
