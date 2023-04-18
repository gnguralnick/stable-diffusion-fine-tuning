import argparse
import os

from run_inference import edit_prompts
import clip_distance
import fid


def run_eval(method_name, checkpoint_steps):

    results_file = open(f"../evaluation-results/{method_name}-{checkpoint_steps}.txt", "w")
    results_file.write("prompt-type,target-name,avg-fid,avg-clip")

    # compute FID and CLIP scores for basic prompt
    basic_prompt_dir = f"../generated-images/{method_name}/basic"
    basic_target_dir = "../target-images"
    basic_target_names = os.listdir(basic_target_dir)
    overall_fid = 0
    overall_clip = 0
    for target_name in basic_target_names:
        target_dir = os.path.join(basic_target_dir, target_name)
        avg_fid = fid.main(target_dir, basic_prompt_dir)
        avg_clip = clip_distance.main(target_dir, basic_prompt_dir)
        results_file.write(f"basic,{target_name},{avg_fid},{avg_clip}")

        overall_fid += avg_fid
        overall_clip += avg_clip

    overall_fid /= len(basic_target_names)
    overall_clip /= len(basic_target_names)
    results_file.write(f"basic,overall,{overall_fid},{overall_clip}")  # overall image similarity scores

    # compute FID and CLIP scores for edit prompts
    overall_fid = 0
    overall_clip = 0
    for prompt in edit_prompts:
        prompt_dir = f"../generated-images/{method_name}/{prompt}"
        prompt_target_dir = f"../target-complex-images/{prompt}"
        prompt_target_names = os.listdir(prompt_target_dir)
        prompt_overall_fid = 0
        prompt_overall_clip = 0
        for target_name in prompt_target_names:
            target_dir = os.path.join(prompt_target_dir, target_name)
            avg_fid = fid.main(target_dir, prompt_dir)
            avg_clip = clip_distance.main(target_dir, prompt_dir)
            results_file.write(f"{prompt},{target_name},{avg_fid},{avg_clip}")

            prompt_overall_fid += avg_fid
            prompt_overall_clip += avg_clip

        prompt_overall_fid /= len(prompt_target_names)
        prompt_overall_clip /= len(prompt_target_names)
        results_file.write(f"{prompt},overall,{prompt_overall_fid},{prompt_overall_clip}")

        overall_fid += prompt_overall_fid
        overall_clip += prompt_overall_clip

    overall_fid /= len(edit_prompts)
    overall_clip /= len(edit_prompts)
    results_file.write(f"overall,overall,{overall_fid},{overall_clip}")  # overall text similarity scores

    results_file.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Compute evaluation metrics for a given method and checkpoint step.")
    parser.add_argument("--method_name", type=str, help="Name of the method used for generating images.")
    parser.add_argument("--checkpoint_steps", type=int, help="Number of steps used for training the model.")

    args = parser.parse_args()

    run_eval(args.method_name, args.checkpoint_steps)
