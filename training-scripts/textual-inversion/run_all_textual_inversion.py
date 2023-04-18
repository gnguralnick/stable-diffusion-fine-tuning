from train_textual_inversion import main
import os
from os import getcwd

if __name__ == "__main__":
    # call main for every subdirectory of ../../target-images
    # main(target_images_dir, initializer_token="object", model_output_dir=None, model_name=DEFAULT_MODEL_NAME,
    #      placeholder_token="<*>", generated_images_dir=None, train=True, train_log="training.log")

    for subdir in os.listdir("../../target-images"):
        main(f"../../target-images/{subdir}", train_log=f"{getcwd()}/training-logs/{subdir}.log")
