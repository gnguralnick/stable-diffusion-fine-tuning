import argparse

from run_inference import run_inference
import os


def run_all_textual_inversion_inference(checkpoint_steps=None):
    generated_dir = "../generated-images"
    target_complex_dir = "../target-complex-images"

    target_dir = "../target-images"
    target_names = os.listdir(target_dir)
    for target_name in target_names:
        run_inference(generated_dir, method="textual-inversion", target_name=target_name,
                      learned_embeddings_path=f"../fine-tuned-models/textual-inversion/{target_name}",
                      checkpoint_steps=checkpoint_steps)
        run_inference(target_complex_dir, method="textual-inversion", target_name=target_name,
                      learned_embeddings_path=f"../fine-tuned-models/textual-inversion/{target_name}",
                      checkpoint_steps=checkpoint_steps, placeholder_token="")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_steps", type=int, required=False, default=None)

    args = parser.parse_args()

    run_all_textual_inversion_inference(checkpoint_steps=args.checkpoint_steps)
