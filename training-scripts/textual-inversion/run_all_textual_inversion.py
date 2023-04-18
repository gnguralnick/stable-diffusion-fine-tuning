from train_textual_inversion import main
from os import getcwd
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--target_images_dirs", type=str, required=True, nargs='+')

    args = parser.parse_args()

    print("Running training for the following target images directories:")
    for target_dir in args.target_images_dirs:
        print('\t' + target_dir)

    for target_dir in args.target_images_dirs:
        target_dir_name = target_dir.split("/")[-1]
        main(target_dir, train_log=f"{getcwd()}/training-logs/{target_dir_name}.log")
