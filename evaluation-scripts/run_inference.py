import argparse

from diffusers import StableDiffusionPipeline
import torch
from numpy import random

HYPERPARAMETERS = {
    "num_inference_steps": 50,
    "guidance_scale": 7.5,
    "num_generations": 10
}

edit_prompts = [
    "A photo of a <placeholder> in a forest",
    "A photo of a <placeholder> in a city",
    "A photo of a <placeholder> in a desert",
    "An oil painting of a <placeholder>",
    "A photo of a <placeholder> in a field",
    "Elmo holding a <placeholder>",
    "A photo of a <placeholder> in a kitchen",
    "A photo of a <placeholder> in a bedroom",
    "A painting of a <placeholder> in a forest",
]

BASIC_PROMPT = "A photo of a <placeholder>"


def run_inference(model_output_path, generated_images_dir, placeholder_token, hyperparameters,
                  method, learned_embeddings_path=None, complex_prompt=False):
    model_id = model_output_path
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

    if method == "textual-inversion":
        pipe.load_textual_inversion(learned_embeddings_path,
                                    weight_name=f"learned_embeds.bin",
                                    local_files_only=True)

    if not complex_prompt:
        prompt = BASIC_PROMPT.replace("<placeholder>", placeholder_token)
        prompts = [prompt] * hyperparameters['num_generations']
    else:
        full_prompts = [edit_prompt.replace("<placeholder>", placeholder_token) for edit_prompt in edit_prompts]
        prompts = random.sample(full_prompts, hyperparameters['num_generations'])

    for i in range(len(prompts)):
        prompt = prompts[i]
        image = pipe(prompt,
                     num_inference_steps=hyperparameters['num_inference_steps'],
                     guidance_scale=hyperparameters['guidance_scale']).images[0]

        image.save(f"{generated_images_dir}/image_{i}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_output_path", type=str, required=True)
    parser.add_argument("--generated_images_dir", type=str, required=True)
    parser.add_argument("--placeholder_token", type=str, required=True)
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--learned_embeddings_path", type=str, default=None)
    parser.add_argument("--complex_prompt", action="store_true")
    args = parser.parse_args()

    run_inference(args.model_output_path, args.generated_images_dir, args.placeholder_token, HYPERPARAMETERS,
                  args.method, args.learned_embeddings_path, args.complex_prompt)
