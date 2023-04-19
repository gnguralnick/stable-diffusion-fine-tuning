# Description: Compute Frechet Inception Distance between two sets of images.
# Note: ChatGPT was used to assist in the creation of this script.

import os
import argparse
import warnings
from scipy import linalg
import torch
import numpy as np
import torchvision.transforms as transforms
from load_images import load_images_from_directory

from torchvision.models import inception_v3
from torchvision.models.inception import Inception_V3_Weights


def compute_fid(images1, images2):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load the Inception-v3 model
    model = inception_v3(transform_input=False, weights=Inception_V3_Weights.IMAGENET1K_V1).to(device)
    model = model.eval()

    # Preprocess images
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    images1 = torch.stack([transform(image) for image in images1]).to(device)
    images2 = torch.stack([transform(image) for image in images2]).to(device)

    # Compute image features
    with torch.no_grad():
        features1 = model(images1)
        features2 = model(images2)

    # Compute the mean and covariance of the features
    mu1, mu2 = features1.mean(0), features2.mean(0)
    sigma1, sigma2 = torch_cov(features1), torch_cov(features2)

    sigma1_np, sigma2_np = sigma1.cpu().numpy(), sigma2.cpu().numpy()

    covmean = linalg.sqrtm(np.dot(sigma1_np, sigma2_np))
    if np.iscomplexobj(covmean):
       covmean = covmean.real

    # Compute the FID
    dist = np.linalg.norm(mu1.cpu().numpy() - mu2.cpu().numpy()) ** 2 + \
           np.trace(sigma1_np + sigma2_np - 2 * covmean)

    return dist


def torch_cov(mat):
    mat = mat - mat.mean(0)
    return 1 / (mat.shape[0] - 1) * mat.T @ mat


def main(target_images_dir, generated_images_dir):
    target_images = load_images_from_directory(target_images_dir)

    generated_images = load_images_from_directory(generated_images_dir)

    fid = compute_fid(target_images, generated_images)

    return fid


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute Frechet Inception Distance between two sets of images.")
    parser.add_argument("--target_images_dir", type=str, help="Path to the directory containing target images.")
    parser.add_argument("--generated_images_dir", type=str,
                        help="Path to the directory containing generated images.")
    parser.add_argument("--method_name", type=str, help="Name of the method used for generating images.")
    parser.add_argument("--eval_output_dir", required=False, type=str,
                        help="Path to the directory where evaluation results will be saved.")

    args = parser.parse_args()
    fid = main(args.target_images_dir, args.generated_images_dir)

    target_images_dir_name = args.target_images_dir.split('/')[-1]
    generated_images_dir_name = args.generated_images_dir.split('/')[-1]

    if args.eval_output_dir is None:
        eval_output_dir = f'../../evaluation-results/fid/{target_images_dir_name}/{generated_images_dir_name}'

    os.makedirs(args.eval_output_dir, exist_ok=True)
    output_file = os.path.join(args.eval_output_dir, f"{args.method_name}.txt")

    with open(output_file, 'w') as f:
        f.write(f"Frechet Inception Distance: {fid}\n")
