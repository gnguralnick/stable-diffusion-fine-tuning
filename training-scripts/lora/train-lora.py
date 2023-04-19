import argparse
import os
import sys

from diffusers import StableDiffusionPipeline
import torch
import shutil


HYPERPARAMETERS = {
    "resolution": 512,
    "train_batch_size": 2,
    "gradient_accumulation_steps": 1,
    "max_train_steps": 400,
    "learning_rate": 5e-6,
    "lr_scheduler": "constant",
    "lr_warmup_steps": 0,
    "num_inference_steps": 50,
    "guidance_scale": 7.5,
    "seed": 1337,
    "checkpointing_steps": 100,
    "validation_epochs": 50,
    "num_generations": 10
}

DEFAULT_MODEL_NAME = "runwayml/stable-diffusion-v1-5"


def run_training(target_images_dir, model_output_dir, model_name,
                 instance_prompt, hyperparameters):
    """Run the LoRA-based model training."""

    os.system(f"accelerate launch LoRA.py \
        --pretrained_model_name_or_path={model_name}  \
        --instance_data_dir= {target_images_dir} \
        --output_dir={model_output_dir} \
        --instance_prompt=\"{instance_prompt}\" \
        --resolution={hyperparameters['resolution']} \
        --train_batch_size={hyperparameters['train_batch_size']} \
        --gradient_accumulation_steps={hyperparameters['gradient_accumulation_steps']} \
        --learning_rate={hyperparameters['learning_rate']} \
        --lr_scheduler=\"{hyperparameters['lr_scheduler']}\" \
        --lr_warmup_steps={hyperparameters['lr_warmup_steps']} \
        --max_train_steps={hyperparameters['max_train_steps']} \
        --seed={hyperparameters['seed']}")

    return os.path.exists(f"{model_output_dir}")


def run_inference(model_output_path, generated_images_dir, prompt,
                  hyperparameters):
    model_id = model_output_path
    pipe = StableDiffusionPipeline.from_pretrained(model_id,torch_dtype=torch.float16).to("cuda")

    for i in range(hyperparameters['num_generations']):
        image = pipe(prompt, num_inference_steps=hyperparameters[
            'num_inference_steps']).images[0]
        image.save(f"{generated_images_dir}/image_{i}.png")


def main(target_images_dir, model_output_dir=None,
         model_name=DEFAULT_MODEL_NAME,
         instance_prompt="<*>", inference_prompt="<*>",
         generated_images_dir=None, train=True, train_log="training.log"):
    target_images_dir_name = args.target_images_dir.split("/")[-1]

    if model_output_dir is None:
        model_output_dir = f"../../fine-tuned-models/LoRA/{target_images_dir_name}"
    if generated_images_dir is None:
        generated_images_dir = f"../../generated-images/LoRA/{target_images_dir_name}"

    shutil.rmtree(generated_images_dir, ignore_errors=True)
    os.makedirs(generated_images_dir, exist_ok=True)

    with open(train_log, "w") as f:
        f.write(
            f"Running LoRA training for the target images in {target_images_dir}.")

    if train:
        model_training_successful = run_training(target_images_dir,
                                                 model_output_dir, model_name,
                                                 instance_prompt,
                                                 HYPERPARAMETERS)
    else:
        model_training_successful = True

    if not model_training_successful:
        sys.exit("Model training failed. Exiting.")

    with open(train_log, "a") as f:
        f.write("Training completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_images_dir", type=str, required=True)
    # parser.add_argument("--logfile", type=str, required=False, default="training.log")
    # parser.add_argument("--model_output_dir", type=str, required=False)
    # parser.add_argument("--model_name", type=str, required=False, default=DEFAULT_MODEL_NAME)
    # parser.add_argument("--instance_prompt", type=str, required=False, default="<*>")
    # parser.add_argument("--inference_prompt", type=str, required=False, default="<*>")
    # parser.add_argument("--generated_images_dir", type=str, required=False)
    # parser.add_argument("--train", type=bool, required=False, default=True)
    args = parser.parse_args()

    instance_prompt = "a photo of backpack"
    inference_prompt = "a photo of backpack"
    #lst = ["backpack", "bottle", "car", "cat","clock", "duck", "llama", "teapot", "wallet", "wolf"]
    main(target_images_dir=args.target_images_dir, instance_prompt=instance_prompt, inference_prompt=inference_prompt, train=True)
