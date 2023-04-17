import os
import sys

from diffusers import StableDiffusionPipeline
import torch
import argparse

HYPERPARAMETERS = {
    "resolution": 512,
    "train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "max_train_steps": 3000,
    "learning_rate": 5.0e-4,
    "lr_scheduler": "constant",
    "lr_warmup_steps": 0,
    "num_inference_steps": 50,
    "guidance_scale": 7.5,
    "seed": 42
}

DEFAULT_MODEL_NAME = "runwayml/stable-diffusion-v1-5"


def run_training(target_images_dir, model_output_dir, model_name, placeholder_token, initializer_token,
                 hyperparameters):
    """Run the textual inversion training."""
    os.system(f"accelerate launch textual-inversion.py \
        --pretrained_model_name_or_path={model_name} \
        --train_data_dir={target_images_dir} \
        --learnable_property=\"object\" \
        --placeholder_token=\"{placeholder_token}\" --initializer_token=\"{initializer_token}\" \
        --resolution={hyperparameters['resolution']} \
        --train_batch_size={hyperparameters['train_batch_size']} \
        --gradient_accumulation_steps={hyperparameters['gradient_accumulation_steps']} \
        --max_train_steps={hyperparameters['max_train_steps']} \
        --learning_rate={hyperparameters['learning_rate']} --scale_lr \
        --lr_scheduler=\"{hyperparameters['lr_scheduler']}\" \
        --lr_warmup_steps={hyperparameters['lr_warmup_steps']} \
        --seed={hyperparameters['seed']} \
        --output_dir={model_output_dir}")
    # return whether the training was successful
    return os.path.exists(f"{model_output_dir}/checkpoint-{hyperparameters['max_train_steps']}")


def run_inference(model_output_path, generated_images_dir, placeholder_token, hyperparameters):
    model_id = model_output_path
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

    prompt = f"A photo of a {placeholder_token}"

    images = pipe(prompt,
                  num_inference_steps=hyperparameters['num_inference_steps'],
                  guidance_scale=hyperparameters['guidance_scale'])

    for i, image in enumerate(images):
        image.save(f"{generated_images_dir}/image_{i}.png")


def main(target_images_dir, initializer_token="object", model_output_dir=None, model_name=DEFAULT_MODEL_NAME,
         placeholder_token="<*>", generated_images_dir=None, train=False):
    target_images_dir_name = args.target_images_dir.split("/")[-1]

    if model_output_dir is None:
        model_output_dir = f"../../fine-tuned-models/textual-inversion/{target_images_dir_name}"
    if generated_images_dir is None:
        generated_images_dir = f"../../generated-images/textual-inversion/{target_images_dir_name}"

    if train:
        model_training_successful = run_training(target_images_dir, model_output_dir, model_name,
                                         placeholder_token, initializer_token, HYPERPARAMETERS)
    else:
        model_training_successful = True
    if model_training_successful:
        model_output_path = f"{model_output_dir}/checkpoint-{HYPERPARAMETERS['max_train_steps']}"
    else:
        sys.exit("Model training failed. Exiting.")
    run_inference(model_output_path, generated_images_dir, placeholder_token, HYPERPARAMETERS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_images_dir", type=str, required=True)
    parser.add_argument("--model_output_dir", type=str, required=False)
    parser.add_argument("--model_name", type=str, required=False, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--placeholder_token", type=str, required=False, default="<*>")
    parser.add_argument("--initializer_token", type=str, required=False, default="object")
    parser.add_argument("--generated_images_dir", type=str, required=False)
    parser.add_argument("--train", type=bool, required=False, default=True)
    args = parser.parse_args()

    main(args.target_images_dir, args.initializer_token, args.model_output_dir, args.model_name, args.placeholder_token,
         args.generated_images_dir, args.train)
