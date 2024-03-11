import torch
from assets.templates import LitModel, ffhq256_autoenc
from sentence_transformers import SentenceTransformer
from PIL import Image
import torchvision.transforms as transforms
import argparse
import os


def save_tensor_as_image(tensor, path):
    """
    Saves a tensor as an image.

    Args:
    - tensor: A torch.Tensor to be saved as an image.pip
    - path: Path where the image will be saved.
    """
    from torchvision.utils import save_image
    save_image(tensor, path)


def save_tensor(tensor, path):
    """
    Saves a tensor in .pt format.

    Args:
    - tensor: A torch.Tensor to be saved.
    - path: Path where the tensor will be saved.
    """
    torch.save(tensor, path)


def encode_image_and_text(image_path, text, save_path, save_as_image=True):
    """
    Encodes an image and text into a tensor xT, and optionally saves it.

    Args:
    - image_path: Path to the input image.
    - text: Text description to condition the encoding.
    - save_path: Path where the encoded tensor or image will be saved.
    - save_as_image: If True, saves the tensor as an image; otherwise, saves as a .pt file.
    """

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Load configuration and model
    conf = ffhq256_autoenc()
    model = LitModel(conf)
    state = torch.load(f'checkpoints/{conf.name}/last.ckpt', map_location='cpu')
    model.load_state_dict(state['state_dict'], strict=False)
    model.ema_model.eval()
    model.ema_model.to(device)

    # Process image
    img = Image.open(image_path).convert('RGB').resize((256, 256))
    img_tensor = transforms.ToTensor()(img).unsqueeze_(0).to(device)

    # Process text
    modelem = SentenceTransformer('clip-ViT-L-14')
    text_emb = modelem.encode([text])
    text_emb = torch.from_numpy(text_emb).to(device)
    cond = text_emb[:, :512].reshape(1, 512)

    # Encode
    xT = model.encode_stochastic(img_tensor, cond, T=250)

    # Save
    if save_as_image:
        save_tensor_as_image(xT, save_path)
    else:
        save_tensor(xT, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Encode image and text to xT tensor.')
    parser.add_argument('--image_path', default='D:\Github\LASER\imgs\download.jpg', type=str, help='Path to the input image.')
    parser.add_argument('--text', default='A man', type=str, help='Text description for conditioning.')
    parser.add_argument('--save_path', default='D:\Github\LASER\solution.pt', type=str, help='Path to save the encoded tensor or image.')
    parser.add_argument('--save_as_image', default=False, action='store_true', help='Save the encoded tensor as an image.')
    args = parser.parse_args()

    # Make sure the save directory exists
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    encode_image_and_text(args.image_path, args.text, args.save_path, args.save_as_image)
