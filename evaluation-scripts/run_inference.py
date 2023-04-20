import argparse
import os
from collections import defaultdict

from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch

from safetensors.torch import load_file

HYPERPARAMETERS = {
    "num_inference_steps": 50,
    "guidance_scale": 7.5,
    "num_generations": 5
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


def load_lora_weights(pipeline, checkpoint_path, multiplier, device, dtype):
    LORA_PREFIX_UNET = "lora_unet"
    LORA_PREFIX_TEXT_ENCODER = "lora_te"
    # load LoRA weight from .safetensors
    state_dict = load_file(checkpoint_path, device=device)

    updates = defaultdict(dict)
    for key, value in state_dict.items():
        # it is suggested to print out the key, it usually will be something like below
        # "lora_te_text_model_encoder_layers_0_self_attn_k_proj.lora_down.weight"

        layer, elem = key.split('.', 1)
        updates[layer][elem] = value

    # directly update weight in diffusers model
    for layer, elems in updates.items():

        if "text" in layer:
            layer_infos = layer.split(LORA_PREFIX_TEXT_ENCODER + "_")[-1].split("_")
            curr_layer = pipeline.text_encoder
        else:
            layer_infos = layer.split(LORA_PREFIX_UNET + "_")[-1].split("_")
            curr_layer = pipeline.unet

        # find the target layer
        temp_name = layer_infos.pop(0)
        while len(layer_infos) > -1:
            try:
                curr_layer = curr_layer.__getattr__(temp_name)
                if len(layer_infos) > 0:
                    temp_name = layer_infos.pop(0)
                elif len(layer_infos) == 0:
                    break
            except Exception:
                if len(temp_name) > 0:
                    temp_name += "_" + layer_infos.pop(0)
                else:
                    temp_name = layer_infos.pop(0)

        # get elements for this layer
        weight_up = elems['lora_up.weight'].to(dtype)
        weight_down = elems['lora_down.weight'].to(dtype)
        alpha = elems['alpha']
        if alpha:
            alpha = alpha.item() / weight_up.shape[1]
        else:
            alpha = 1.0

        # update weight
        if len(weight_up.shape) == 4:
            curr_layer.weight.data += multiplier * alpha * torch.mm(weight_up.squeeze(3).squeeze(2),
                                                                    weight_down.squeeze(3).squeeze(2)).unsqueeze(
                2).unsqueeze(3)
        else:
            curr_layer.weight.data += multiplier * alpha * torch.mm(weight_up, weight_down)

    return pipeline


def run_inference(generated_images_dir, method, target_name,
                  placeholder_token="<*>", hyperparameters=None, model_path=DEFAULT_MODEL_NAME, learned_embeddings_path=None,
                  checkpoint_steps=None):
    if hyperparameters is None:
        hyperparameters = HYPERPARAMETERS
    print(f"Running inference for {target_name} from method {method}")
    print(f"Model path: {model_path}")
    print(f"Learned embeddings path: {learned_embeddings_path}")
    print(f"Checkpoint steps: {checkpoint_steps}")

    model_id = model_path
    if torch.cuda.is_available():
        pipe = StableDiffusionPipeline.from_pretrained(DEFAULT_MODEL_NAME, torch_dtype=torch.float16).to("cuda")
    else:
        pipe = StableDiffusionPipeline.from_pretrained(model_id)

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    if method == "textual-inversion":
        weight_name = f"learned_embeds-steps-{checkpoint_steps}.bin" if checkpoint_steps else "learned_embeds.bin"
        pipe.load_textual_inversion(learned_embeddings_path,
                                    weight_name=weight_name,
                                    local_files_only=True)
    elif method == "lora":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        # weight_name = f"{model_path}/"
        # pipe.unet.load_attn_procs(model_path, use_safetensors=True)
        pipe = load_lora_weights(pipe, model_path, 1.0, "cuda", torch.float16)

    subdir = generated_images_dir + f"/{method}"
    if checkpoint_steps:
        subdir += f"-step-{checkpoint_steps}"
    if not os.path.exists(subdir):
        os.makedirs(subdir)
    subdir += f"/{target_name}/"
    if not os.path.exists(subdir):
        os.makedirs(subdir)
    subdir_basic = os.path.join(subdir, "basic")
    subdirs_complex = {edit_prompt: os.path.join(subdir, f"{edit_prompt}") for edit_prompt in edit_prompts.keys()}
    if not os.path.exists(subdir_basic):
        os.makedirs(subdir_basic)
    for subdir_complex in subdirs_complex:
        if not os.path.exists(subdirs_complex[subdir_complex]):
            os.makedirs(subdirs_complex[subdir_complex])

    print(f"Saving images to {subdir}...")

    for i in range(hyperparameters['num_generations']):
        image = pipe(BASIC_PROMPT.replace("<placeholder>", placeholder_token),
                     num_inference_steps=hyperparameters['num_inference_steps'],
                     guidance_scale=hyperparameters['guidance_scale']).images[0]

        image.save(os.path.join(subdir_basic, f"image_{i}.png"))

        for edit_prompt in edit_prompts:
            image = pipe(edit_prompts[edit_prompt].replace("<placeholder>", placeholder_token),
                         num_inference_steps=hyperparameters['num_inference_steps'],
                         guidance_scale=hyperparameters['guidance_scale']).images[0]

            image.save(
                os.path.join(subdirs_complex[edit_prompt], f"image_{i}.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_output_path", type=str, required=False, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--generated_images_dir", type=str, required=True)
    parser.add_argument("--placeholder_token", type=str, required=False, default="<*>")
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--target_name", type=str, required=True)
    parser.add_argument("--learned_embeddings_path", type=str, default=None)
    parser.add_argument("--checkpoint_steps", type=int, default=None)
    args = parser.parse_args()

    run_inference(args.generated_images_dir, args.method, args.target_name,
                  args.placeholder_token, HYPERPARAMETERS, args.model_output_path, args.learned_embeddings_path,
                  args.checkpoint_steps)
