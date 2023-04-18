import argparse
import os

from diffusers import StableDiffusionPipeline
import torch
from numpy import random

HYPERPARAMETERS = {
    "num_inference_steps": 50,
    "guidance_scale": 7.5,
    "num_generations": 10
}

edit_prompts = {
    "forest": "A photo of a <placeholder> in a forest",
    "city": "A photo of a <placeholder> in a city",
    "desert": "A photo of a <placeholder> in a desert",
    "painting": "An oil painting of a <placeholder>",
    "field": "A photo of a <placeholder> in a field",
    "elmo": "Elmo holding a <placeholder>",
    "kitchen": "A photo of a <placeholder> in a kitchen",
    "bedroom": "A photo of a <placeholder> in a bedroom",
    "forest_painting": "A painting of a <placeholder> in a forest",
}

BASIC_PROMPT = "A photo of a <placeholder>"

DEFAULT_MODEL_NAME = "runwayml/stable-diffusion-v1-5"


def run_inference(generated_images_dir, placeholder_token, hyperparameters,
                  method, target_name, model_path=DEFAULT_MODEL_NAME, learned_embeddings_path=None,
                  checkpoint_steps=None):
    print(f"Running inference for {target_name} from method {method}")
    print(f"Model path: {model_path}")
    print(f"Learned embeddings path: {learned_embeddings_path}")
    print(f"Checkpoint steps: {checkpoint_steps}")

    model_id = model_path
    if torch.cuda.is_available():
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
    else:
        pipe = StableDiffusionPipeline.from_pretrained(model_id)

    if method == "textual-inversion":
        weight_name = f"learned_embeds-steps-{checkpoint_steps}.bin" if checkpoint_steps else "learned_embeds.bin"
        pipe.load_textual_inversion(learned_embeddings_path,
                                    weight_name=weight_name,
                                    local_files_only=True)

    subdir = generated_images_dir + f"/{method}"
    if not os.path.exists(subdir):
        os.makedirs(subdir)
    subdir += f"/{target_name}/"
    if not os.path.exists(subdir):
        os.makedirs(subdir)
    subdir_basic = subdir + "basic/"
    subdirs_complex = [subdir + f"{edit_prompt}/" for edit_prompt in edit_prompts]
    if not os.path.exists(subdir_basic):
        os.makedirs(subdir_basic)
    for subdir_complex in subdirs_complex:
        if not os.path.exists(subdir_complex):
            os.makedirs(subdir_complex)

    print(f"Saving images to {subdir}...")

    for i in range(hyperparameters['num_generations']):
        image = pipe(BASIC_PROMPT.replace("<placeholder>", placeholder_token),
                     num_inference_steps=hyperparameters['num_inference_steps'],
                     guidance_scale=hyperparameters['guidance_scale']).images[0]

        image.save(f"{subdir_basic}/image_{i}{f'-checkpoint-{checkpoint_steps}' if checkpoint_steps else ''}.png")

        for j in range(len(edit_prompts)):
            image = pipe(edit_prompts[j].replace("<placeholder>", placeholder_token),
                         num_inference_steps=hyperparameters['num_inference_steps'],
                         guidance_scale=hyperparameters['guidance_scale']).images[0]

            image.save(
                f"{subdirs_complex[j]}/image_{i}{f'-checkpoint-{checkpoint_steps}' if checkpoint_steps else ''}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_output_path", type=str, required=False, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--generated_images_dir", type=str, required=True)
    parser.add_argument("--placeholder_token", type=str, required=False, default="<*>")
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--target_name", type=str, required=True)
    parser.add_argument("--learned_embeddings_path", type=str, default=None)
    parser.add_argument("--checkpoint_steps", type=int, default=None)
    parser.add_argument("--complex_prompt", action="store_true")
    args = parser.parse_args()

    run_inference(args.generated_images_dir, args.placeholder_token, HYPERPARAMETERS,
                  args.method, args.target_name, args.model_output_path, args.learned_embeddings_path,
                  args.checkpoint_steps)
