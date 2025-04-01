import sys, os, argparse, yaml, shutil

sys.path.append("../")

import utils.gpt as gpt
from utils.util import save, get_prompt, save_svg

import torch
import clip
import ImageReward as RM
import glob
from PIL import Image


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, help="concept to be generated")
    parser.add_argument("--output_path", type=str, help="top folder name to save the results")
    parser.add_argument("--output_folder", type=str, help="folder name to save the results")
    parser.add_argument("--system_path", type=str, default="..")
    parser.add_argument("--viewbox", type=int, default=512)
    parser.add_argument("--refine_iter", type=int, default=2)
    parser.add_argument("--model", type=str, default="claude-3-5-sonnet-20240620")
    parser.add_argument("--reward_model", type=str, default="ImageReward")
    parser.add_argument("--prompt", type=str, default="")
    args = parser.parse_args()
    if args.prompt is None or args.prompt == "":
        args.prompt = get_prompt(args.target)
    # Set up output directories
    args.root_dir = f"{args.output_path}/{args.output_folder}"
    args.output_folder = f"{args.root_dir}/stage_1"
    args.svg_dir = f"{args.output_folder}/svg_logs"
    args.png_dir = f"{args.output_folder}/png_logs"
    args.msg_dir = f"{args.output_folder}/raw_logs"
    for dir in [args.output_folder, args.svg_dir, args.png_dir, args.msg_dir]:
        os.makedirs(dir, exist_ok=True)

    args.prompts_file = "prompts"

    # Save config
    with open(f'{args.output_folder}/config.yaml', 'w') as f:
        yaml.dump(args.__dict__, f)
    # Save prompts file

    shutil.copyfile(f"{args.system_path}/{args.prompts_file}.yaml", f"{args.output_folder}/prompts.yaml")

    return args


def select_best_svg(cfg, model_name='ImageReward'):
    assert model_name in ['ImageReward', 'CLIP'], "Only `ImageReward` and `CLIP` are supported"

    # Load appropriate model
    if model_name == 'ImageReward':
        model = RM.load("ImageReward-v1.0")
    else:  # CLIP
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)

    png_files = sorted(glob.glob(f"{cfg.png_dir}/*.png"))
    prompt = cfg.prompt

    # Get ranking based on selected model
    with torch.no_grad():
        if model_name == 'ImageReward':
            ranking, _ = model.inference_rank(prompt, png_files)
            best_index = ranking[0] - 1  # ImageReward uses 1-based indexing
        else:  # CLIP
            device = next(model.parameters()).device  # Get device from model

            # Process images and text
            images = torch.cat([preprocess(Image.open(png_file)).unsqueeze(0)
                                for png_file in png_files]).to(device)
            text = clip.tokenize([prompt]).to(device)

            # Compute normalized features and similarity
            image_features = model.encode_image(images)
            text_features = model.encode_text(text)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            # Get ranking
            similarity = (100.0 * image_features @ text_features.T).squeeze()
            best_index = similarity.argmax().item()

    # Copy the best SVG to the root directory
    best_svg = f"{cfg.target}_{best_index}.svg"
    print(f"The best SVG is: {best_svg}")
    shutil.copy(f"{cfg.svg_dir}/{best_svg}", f"{cfg.root_dir}/{cfg.target}_template.svg")


def main(cfg):
    session = gpt.Session(model=cfg.model, prompts_file=f"{cfg.system_path}/{cfg.prompts_file}")
    msg_path = lambda i: f"{cfg.msg_dir}/{cfg.target}_raw{i}"

    # Task 1: Expand the Text Prompt
    expanded_text_prompt = session.send("expand_text_prompt", {"text_prompt": cfg.prompt},
                                        file_path=f"{cfg.msg_dir}/{cfg.target}_prompt")
    save(f"{cfg.msg_dir}/{cfg.target}_prompt", expanded_text_prompt)

    # Task 2: Generate SVG Code
    svg_code = session.send("write_svg_code", file_path=msg_path(0))
    save(msg_path(0), svg_code)
    save_svg(cfg, svg_code, f"{cfg.target}_0")
    svg_path = f"{cfg.svg_dir}/{cfg.target}_0.svg"
    png_path = f"{cfg.png_dir}/{cfg.target}_0.png"

    # Task 3: Iterate Improvement
    for i in range(1, cfg.refine_iter + 1):
        svg_code = session.send("svg_refine", images=[png_path], file_path=msg_path(i))
        save(msg_path(i), svg_code)
        save_svg(cfg, svg_code, f"{cfg.target}_{i}")
        svg_path = f"{cfg.svg_dir}/{cfg.target}_{i}.svg"
        png_path = f"{cfg.png_dir}/{cfg.target}_{i}.png"

    # Automatically select the best SVG
    print("======== Selecting the best SVG using ImageReward or CLIP ========")
    select_best_svg(cfg, model_name=cfg.reward_model)
    print("Done!")


if __name__ == '__main__':
    cfg = parse_arguments()
    main(cfg)
