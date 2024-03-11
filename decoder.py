import torch
from assets.templates import LitModel, ffhq256_autoenc
import argparse
import os
from sentence_transformers import SentenceTransformer
import torchvision.utils as vutils
import random

def apply_noise(xT, ber):
    """
    Applies random noise to the tensor based on the Bit Error Rate (BER).

    Args:
    - xT: The tensor to which noise will be applied.
    - ber: The Bit Error Rate, specifying the noise level.
    """
    lgr, lgb, lgg = xT[0].max(2).values.max(1).values
    slr, slb, slg = xT[0].min(2).values.min(1).values
    
    # Apply random noise based on BER
    for j in range(256):
        for k in range(256):
            if random.uniform(0, 1) <= ber:
                xT[0][:, j, k] = torch.tensor([random.uniform(slr, lgr), random.uniform(slb, lgb), random.uniform(slg, lgg)])
    return xT

def decode_and_save(xT_path, text, save_path, ber=0):
    """
    Decodes an encoded tensor xT into an image, applying optional BER noise, and saves the result.

    Args:
    - xT_path: Path to the encoded tensor xT.
    - text: Text description for conditioning the decoding.
    - save_path: Path where the decoded image will be saved.
    - ber: Optional Bit Error Rate to simulate noise.
    """
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    # Load configuration and model
    conf = ffhq256_autoenc()
    model = LitModel(conf)
    state = torch.load(f'checkpoints/{conf.name}/last.ckpt', map_location='cpu')
    model.load_state_dict(state['state_dict'], strict=False)
    model.ema_model.eval()
    model.ema_model.to(device)

    # Load xT tensor
    xT = torch.load(xT_path).to(device)

    # Process text
    modelem = SentenceTransformer('clip-ViT-L-14')
    text_emb = modelem.encode([text])
    text_emb = torch.from_numpy(text_emb).to(device)
    cond = text_emb[:, :512].reshape(1, 512)

    # Apply optional BER noise
    if ber > 0:
        xT = apply_noise(xT, ber)

    # Decode
    pred = model.render(xT, cond, T=250)
    pred = (pred - pred.min()) / (pred.max() - pred.min())

    # Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    vutils.save_image(pred, save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Decode xT tensor to an image with optional BER noise.')
    parser.add_argument('--xT_path', type=str, help='Path to the encoded tensor xT.')
    parser.add_argument('--text', type=str, help='Text description for conditioning the decoding.')
    parser.add_argument('--save_path', type=str, help='Path to save the decoded image.')
    parser.add_argument('--ber', type=float, default=0, help='Optional Bit Error Rate to simulate noise.')

    args = parser.parse_args()

    decode_and_save(args.xT_path, args.text, args.save_path, args.ber)
