from train_textual_inversion import main
import os
from os import getcwd
import argparse

if __name__ == "__main__":
    # call main for every subdirectory of ../../target-images
    # main(target_images_dir, initializer_token="object", model_output_dir=None, model_name=DEFAULT_MODEL_NAME,
    #      placeholder_token="<*>", generated_images_dir=None, train=True, train_log="training.log")

    parser = argparse.ArgumentParser()
    parser.add_argument("--target_images_dir", type=str, required=False, default="../../target-images")

    args = parser.parse_args()

    for subdir in os.listdir("../../target-images"):
        main(f"{args.target_images_dir}/{subdir}", train_log=f"{getcwd()}/training-logs/{subdir}.log")
