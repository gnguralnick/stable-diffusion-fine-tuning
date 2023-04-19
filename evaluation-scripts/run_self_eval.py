import os

from run_inference import edit_prompts
import clip_distance
import fid


def run_self_eval(method_name, checkpoint_steps=None):
    # run eval where we compare each set of target images to itself
    # and where we compare each set of complex target images to itself
    # this is to provide a baseline for the evaluation

    if checkpoint_steps is not None:
        results_file = open(f"../evaluation-results/{method_name}-self-step-{checkpoint_steps}.txt", "w")
    else:
        results_file = open(f"../evaluation-results/{method_name}-self.txt", "w")

    results_file.write("prompt-type,target-name,avg-fid,avg-clip\n")

    print(f"Running self evaluation for {method_name} with checkpoint steps {checkpoint_steps}...")

    target_dir = "../target-images"
    target_names = os.listdir(target_dir)

    overall_fid = 0
    overall_clip = 0

    for target_name in target_names:
        full_target_dir = os.path.join(target_dir, target_name)
        avg_fid = fid.main(full_target_dir, full_target_dir)
        avg_clip = clip_distance.main(full_target_dir, full_target_dir)
        results_file.write(f"basic,{target_name},{avg_fid},{avg_clip}\n")

        overall_fid += avg_fid
        overall_clip += avg_clip

    overall_fid /= len(target_names)
    overall_clip /= len(target_names)
    results_file.write(f"basic,overall,{overall_fid},{overall_clip}\n")  # overall image similarity scores

    print("Finished basic prompt evaluation.")

    # compute FID and CLIP scores for edit prompts
    overall_fid = 0
    overall_clip = 0
    for prompt in edit_prompts:
        prompt_overall_fid = 0
        prompt_overall_clip = 0
        for target_name in target_names:
            prompt_target_dir = f"../target-complex-images/{method_name}"
            if checkpoint_steps:
                prompt_target_dir += f"-step-{checkpoint_steps}"
            prompt_target_dir = os.path.join(prompt_target_dir, target_name, prompt)
            avg_fid = fid.main(prompt_target_dir, prompt_target_dir)
            avg_clip = clip_distance.main(prompt_target_dir, prompt_target_dir)
            results_file.write(f"{prompt},{target_name},{avg_fid},{avg_clip}\n")

            prompt_overall_fid += avg_fid
            prompt_overall_clip += avg_clip

        prompt_overall_fid /= len(target_names)
        prompt_overall_clip /= len(target_names)
        results_file.write(f"{prompt},overall,{prompt_overall_fid},{prompt_overall_clip}\n")

        overall_fid += prompt_overall_fid
        overall_clip += prompt_overall_clip

    overall_fid /= len(edit_prompts)
    overall_clip /= len(edit_prompts)

    results_file.write(f"edit,overall,{overall_fid},{overall_clip}\n")  # overall image similarity scores

    print("Finished edit prompt evaluation.")

    results_file.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_steps", type=int, required=False, default=None)
    parser.add_argument("--method_name", type=str, required=True)

    args = parser.parse_args()

    run_self_eval(args.method_name, checkpoint_steps=args.checkpoint_steps)
