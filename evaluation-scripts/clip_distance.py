# Description: Compute average CLIP cosine similarity between two sets of images.
# Note: ChatGPT was used to assist in the creation of this script.

import os
import argparse
import torch
import torchvision.transforms as transforms
from transformers import CLIPModel
from torch.nn.functional import cosine_similarity
from load_images import load_images_from_directory


def compute_clip_cosine_similarity(target_images, generated_images):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load the CLIP model and processor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

    # Preprocess images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    target_images = torch.stack([transform(image) for image in target_images]).to(device)
    generated_images = torch.stack([transform(image) for image in generated_images]).to(device)

    # Compute image features
    with torch.no_grad():
        target_features = model.get_image_features(target_images)
        generated_features = model.get_image_features(generated_images)

    # Compute average pairwise cosine similarity
    cosine_similarities = cosine_similarity(target_features.unsqueeze(1), generated_features.unsqueeze(0))

    avg_cosine_similarity = cosine_similarities.mean().item()

    return avg_cosine_similarity


def main(target_images_dir, generated_images_dir):
    target_images = load_images_from_directory(target_images_dir)

    generated_images = load_images_from_directory(generated_images_dir)

    avg_cosine_similarity = compute_clip_cosine_similarity(target_images, generated_images)

    return avg_cosine_similarity




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute average CLIP cosine similarity between two sets of images.")
    parser.add_argument("--target_images_dir", type=str, help="Path to the directory containing target images.")
    parser.add_argument("--generated_images_dir", type=str, help="Path to the directory containing generated images.")
    parser.add_argument("--method_name", type=str, help="Name of the method used for generating images.")
    parser.add_argument("--eval_output_dir", required=False, type=str,
                        help="Path to the directory where evaluation results will be saved.")

    args = parser.parse_args()
    avg_cosine_similarity = main(args.target_images_dir, args.generated_images_dir)

    target_images_dir_name = args.target_images_dir.split('/')[-1]

    generated_images_dir_name = args.generated_images_dir.split('/')[-1]

    if args.eval_output_dir is None:
        eval_output_dir = f'../evaluation-results/clip-distance/{target_images_dir_name}/{generated_images_dir_name}'

    os.makedirs(args.eval_output_dir, exist_ok=True)
    output_file = os.path.join(args.eval_output_dir, f"{args.method_name}.txt")

    with open(output_file, 'w') as f:
        f.write(f"Average CLIP Cosine Similarity: {avg_cosine_similarity}\n")
