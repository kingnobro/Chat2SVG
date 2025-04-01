import sys

sys.path.append("../")

import os
import shutil
import argparse
from typing import List, Tuple
from PIL import ImageFilter
from rasterio import features
from pathlib import Path
from diffusers.utils import make_image_grid

import cv2
import numpy as np
from skimage import measure
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import torch
from transformers import set_seed
from diffusers import StableDiffusionXLControlNetImg2ImgPipeline, ControlNetModel, EulerAncestralDiscreteScheduler
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

from svglib.svg import SVG, SVGPath
from svglib.geom import Bbox
from utils.util import get_prompt


def get_dominant_color(pixels, n_colors=5):
    # Reshape the pixels to be a list of RGB values
    pixels = pixels.reshape(-1, 3)

    n_colors = min(n_colors, len(pixels))
    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
    kmeans.fit(pixels)

    # Get the colors and their counts
    colors = kmeans.cluster_centers_.astype(int)
    counts = np.bincount(kmeans.labels_)

    # Return the most common color
    dominant_color = tuple(colors[np.argmax(counts)])
    return dominant_color


def vectorize_and_add_masks(svg: SVG, image: np.ndarray, masks: List[dict]) -> SVG:
    # Sort masks by area in descending order
    sorted_masks = sorted(masks, key=lambda x: x['area'], reverse=True)

    for i, mask in enumerate(sorted_masks):
        binary_mask = mask['segmentation'].astype(np.uint8)
        contours = measure.find_contours(binary_mask)

        # Subtract all smaller masks from the current mask
        for smaller_mask in sorted_masks[i + 1:]:
            binary_mask = np.logical_and(binary_mask, np.logical_not(smaller_mask['segmentation']))

        # Convert back to uint8
        binary_mask = binary_mask.astype(np.uint8)
        pixels = image[binary_mask.astype(bool)]
        color = get_dominant_color(pixels)
        color_hex = '#{:02x}{:02x}{:02x}'.format(*color)

        for contour in contours:
            epsilon = 0.002 * cv2.arcLength(contour.astype(np.float32), True)
            approx = cv2.approxPolyDP(contour.astype(np.float32), epsilon, True)

            path_data = []
            for j in range(len(approx)):
                y, x = approx[j][0]
                if j == 0:
                    path_data.append(f'M {x:.2f} {y:.2f}')
                else:
                    path_data.append(f'L {x:.2f} {y:.2f}')
            path_data.append('Z')  # Close the path

            svg_path_group = SVGPath.from_str(
                ' '.join(path_data),
                fill=True,
                color=color_hex,
                add_closing=True,
            )
            svg.add_path_group(svg_path_group)

    return svg


def show_masks(image: np.ndarray, masks: List[dict], output_file: Path, add_bbox: bool = False) -> None:
    if not masks:
        print("No masks found")
        return
    sorted_masks = sorted(masks, key=lambda x: x['area'], reverse=True)

    mask_dir = output_file.with_name(f"{output_file.stem}_masks")
    if mask_dir.exists():
        shutil.rmtree(mask_dir)
    mask_dir.mkdir()

    fig, ax = plt.subplots(figsize=(20, 20))
    ax.imshow(image)

    for i, ann in enumerate(sorted_masks):
        color = np.concatenate([np.random.random(3), [0.7]])
        mask_image = np.zeros((*image.shape[:2], 4), dtype=np.float32)
        mask_image[ann['segmentation']] = color

        mask_fig, mask_ax = plt.subplots(figsize=(10, 10))
        mask_ax.imshow(mask_image)
        mask_ax.axis('off')

        stats_text = (
            f"Area: {ann['area']:.2f}\n"
            f"Predicted IoU: {ann['predicted_iou']:.3f}\n"
            f"Stability Score: {ann['stability_score']:.3f}"
        )
        mask_ax.text(10, 45, stats_text, fontsize=12, color='white',
                     bbox=dict(facecolor='black', alpha=0.5))

        plt.savefig(mask_dir / f"mask_{i:03d}.png", bbox_inches='tight', pad_inches=0)
        plt.close(mask_fig)

        ax.imshow(mask_image)

        if add_bbox:
            bbox = ann['bbox']
            rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3],
                                 fill=False, edgecolor=color[:3], linewidth=2.6, linestyle='--')
            ax.add_patch(rect)

    ax.axis('off')
    plt.savefig(output_file, bbox_inches='tight', pad_inches=0)
    plt.close()


def initialize_sam(checkpoint: str, config: dict, model_type: str, device: str) -> SamAutomaticMaskGenerator:
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device=device)
    return SamAutomaticMaskGenerator(sam, **config)


def segment_image(img_path: Path, mask_generator: SamAutomaticMaskGenerator, output_file: Path) -> None:
    image = cv2.imread(str(img_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    masks = mask_generator.generate(image)
    show_masks(image, masks, output_file)
    return masks, image


def select_masks(cfg, masks_template: List[dict], masks_target: List[dict]) -> List[dict]:
    """
    Detect masks in target that are not in init and meet certain criteria.
    """
    # Sort masks by area in descending order
    masks_template = sorted(masks_template, key=lambda x: x['area'], reverse=True)
    masks_target = sorted(masks_target, key=lambda x: x['area'], reverse=True)

    # Set thresholds
    thresh_area_ub = cfg.thresh_area_ub_ratio * masks_target[1]['area']
    thresh_area_lb = cfg.thresh_area_lb
    thresh_iou = cfg.thresh_iou

    masks_added = []

    for i, mask_target in enumerate(masks_target):
        # Skip masks that are too large or too small
        if not thresh_area_lb <= mask_target['area'] <= thresh_area_ub:
            continue

        # Check if the mask is already in masks_template
        if any(calculate_iou(mask_template['segmentation'], mask_target['segmentation']) > thresh_iou
               for mask_template in masks_template):
            continue

        # Calculate the visible area of the mask
        visible_area = mask_target['segmentation'].copy()
        for smaller_mask in masks_target[i + 1:]:
            visible_area = np.logical_and(visible_area, np.logical_not(smaller_mask['segmentation']))

        # Calculate the exposed area ratio
        visible_area_ratio = visible_area.sum() / mask_target['segmentation'].sum()

        if visible_area_ratio > cfg.thresh_visible_area_ratio:
            masks_added.append(mask_target)
        else:
            print(f"Mask {i}, area: {visible_area.sum()}, visible area ratio: {visible_area_ratio:.2f}, "
                  f"is blocked by smaller masks")

    return masks_added


def calculate_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Calculate Intersection over Union (IoU) for two masks."""
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    return intersection.sum() / union.sum()


def sam_add_paths(cfg, svg: SVG, target_image_path: str, final_svg_path: str,
                  mask_generator_coarse: SamAutomaticMaskGenerator,
                  mask_generator_fine: SamAutomaticMaskGenerator) -> None:
    template_image_path = f"{cfg.root_folder}/{cfg.target}_template.png"
    output_dir = Path(cfg.sam_folder)

    masks_template, image_template = segment_image(template_image_path, mask_generator_coarse,
                                                   output_dir / f"segmented_template.png")
    print("masks_template:", len(masks_template))

    masks_target, image_target = segment_image(target_image_path, mask_generator_fine,
                                               output_dir / f"segmented_target.png")
    print("masks_target:", len(masks_target))

    masks_added = select_masks(cfg, masks_template, masks_target)
    print("masks_added:", len(masks_added))

    # save masks_added to output_dir
    image = cv2.imread(str(target_image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    show_masks(image, masks_added, output_dir / f"masks_added.png", add_bbox=True)

    # vectorize masks_added
    svg_with_new_path = vectorize_and_add_masks(svg, image_target, masks_added)
    svg_with_new_path.line_to_bezier().drop_z().filter_duplicates().filter_consecutives().filter_empty()  # convert to cubic bezier
    svg_with_new_path.save_svg(final_svg_path, coordinate_precision=3)


def remove_invisible_paths(svg: SVG):
    """Remove paths that are completely blocked by other paths"""
    canvas_size = int(svg.viewbox.size.x)
    mask_size = canvas_size

    # Get mask of each path group
    # mask: mask_size x mask_size, uint8, 0 for background, 1 for foreground
    group_masks = []
    for i, group in enumerate(svg.svg_path_groups):
        group_shapely = group.to_shapely()

        # Create a mask for the current group
        mask = np.zeros((mask_size, mask_size), dtype=np.uint8)

        # Rasterize the shapely geometry onto the mask
        features.rasterize(
            [(group_shapely, 1)],
            out=mask,
            all_touched=True,
            dtype=np.uint8
        )

        group_masks.append(mask)

    new_path_groups = []
    for i, group in enumerate(svg.svg_path_groups):
        group_mask = group_masks[i]

        unblocked_mask = group_mask.copy()
        for j in range(i + 1, len(group_masks)):
            unblocked_mask = np.logical_and(unblocked_mask, np.logical_not(group_masks[j]))

        # This group is completely blocked, skip it
        if np.sum(unblocked_mask) < 5:
            continue

        new_path_groups.append(group)

    # Update the SVG object with the new path groups
    svg.svg_path_groups = new_path_groups
    return svg


def clean_data(cfg, svg_template_path, svg_cleaned_path):
    try:
        # picosvg
        os.system(f'picosvg {svg_template_path} > {svg_cleaned_path}')

        # svglib
        svg = SVG.load_svg(svg_cleaned_path)
        svg.to_path().simplify_arcs()
        remove_invisible_paths(svg)
        svg.line_to_bezier().split_paths()
        svg.drop_z().filter_duplicates().filter_consecutives().filter_empty()
        svg.save_svg(svg_cleaned_path, coordinate_precision=6)
        return svg
    except Exception as e:
        print(f"Error during clean_data() for {cfg.target}: {str(e)}")
        raise e


def load_model(model_id, controlnet_id=None, clip_skip=2):
    if torch.backends.mps.is_available():
        # app silicon use mps
        device = torch.device("mps")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    controlnet = ControlNetModel.from_pretrained(
        controlnet_id, torch_dtype=torch.float16, use_safetensors=True
    ).to(device)
    pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_single_file(
        model_id, controlnet=controlnet, torch_dtype=torch.float16
    ).to(device)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    clip_layers = pipe.text_encoder.text_model.encoder.layers
    if clip_skip > 0:
        pipe.text_encoder.text_model.encoder.layers = clip_layers[:-clip_skip]
    if not torch.backends.mps.is_available():
        pipe.enable_model_cpu_offload()
    return pipe


def prompt_template(cfg):
    prompt = "children's coloring book image of {}, lineal color, simple drawing, plain, minimalistic, best quality, flat 2d illustration, cartoon style".format(
        cfg.prompt)

    negative_prompt = "black outline, black strok, NSFW, logo, text, blurry, low quality, bad anatomy, sketches, lowres, normal quality, monochrome, grayscale, worstquality, signature, watermark, cropped, bad proportions, out of focus, usemame, Multiple people, bad body, long body, long neck, deformed, mutated, mutation, ugly, disfigured, poorly drawn face, skin blemishes, skin spots, acnes, missing limb, malformed limbs, floating limbs, disconnected limbs, extra limb, extra arms, mutated hands, poorly drawn hands, malformed hands, mutated hands and fingers, bad hands, missing fingers, fused fingers, too many fingers, extra legs, bad feet, cross-eyed"

    return prompt, negative_prompt


def detail_enhancement(cfg, svg_cleaned_path, target_image_path, index=1):
    target = cfg.target
    output_folder = cfg.output_folder
    root_folder = cfg.root_folder

    num_inference_steps = cfg.num_inference_steps
    num_images_per_prompt = cfg.num_images_per_prompt
    controlnet_conditioning_scale = cfg.controlnet_conditioning_scale
    guidance_scale = cfg.guidance_scale
    clip_skip = cfg.clip_skip
    strength = cfg.strength
    blur_radius = cfg.blur_radius

    pipe = load_model(
        model_id=cfg.diffusion_model_id,
        controlnet_id=cfg.controlnet_id,
        clip_skip=clip_skip,
    )

    try:
        svg = SVG.load_svg(svg_cleaned_path)

        svg_512 = svg.copy()
        svg_512.normalize(Bbox(512))  # resize to 512x512
        svg_512.save_svg(f"{output_folder}/{target}_512.svg")
        original_image = svg.draw(do_display=False, return_png=True, background_color="white")
        control_image = original_image.filter(ImageFilter.GaussianBlur(blur_radius))
        control_image.save(f"{output_folder}/template_blurred.png")

        prompt, negative_prompt = prompt_template(cfg)

        output_images = pipe(
            prompt,
            negative_prompt=negative_prompt,
            image=original_image,
            control_image=control_image,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=num_images_per_prompt,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            strength=strength,  # img2img
        ).images

        for i, output_sd in enumerate(output_images):
            output_sd.save(f"{output_folder}/target_{i + 1}.png")
            # save the target image
            if i + 1 == index:
                output_sd.save(target_image_path)

        output_grid = make_image_grid([original_image, control_image, *output_images], rows=2, cols=3)
        output_grid.save(f"{output_folder}/target_grid.png")
        original_image.save(f"{root_folder}/{target}_template.png")

        del pipe

    except Exception as e:
        print(f"Error during detail_enhancement() for {target}: {str(e)}")
        raise e


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, help="object to be enhanced")
    parser.add_argument("--output_path", type=str, help="top folder name to save the results")
    parser.add_argument("--output_folder", type=str, help="folder name to save the results")
    # For Image Diffusion
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--num_inference_steps", type=int, default=30, help="number of inference steps")
    parser.add_argument("--num_images_per_prompt", type=int, default=4, help="number of images per prompt")
    parser.add_argument("--controlnet_conditioning_scale", type=float, default=0.5,
                        help="controlnet conditioning scale")
    parser.add_argument("--guidance_scale", type=float, default=7.0, help="guidance scale")
    parser.add_argument("--clip_skip", type=int, default=2, help="clip skip")
    parser.add_argument("--strength", type=float, default=1.0, help="strength")
    parser.add_argument("--blur_radius", type=int, default=7, help="blur radius")
    parser.add_argument("--diffusion_model_id", type=str, default="models/aamXLAnimeMix_v10.safetensors",
                        help="diffusion model id")
    parser.add_argument("--controlnet_id", type=str, default="xinsir/controlnet-tile-sdxl-1.0", help="controlnet id")
    # For SAM
    parser.add_argument("--sam_checkpoint", type=str, default="./models/sam_vit_h_4b8939.pth",
                        help="Path to the SAM checkpoint")
    parser.add_argument("--model_type", type=str, default="vit_h", help="Model type for SAM")
    parser.add_argument("--thresh_iou", type=float, default=0.4, help="IoU threshold for selecting masks")
    parser.add_argument("--remove_outer_mask", type=bool, default=True, help="Whether to remove the outer mask")
    parser.add_argument("--thresh_visible_area_ratio", type=float, default=0.2,
                        help="Threshold of exposed area ratio for selecting masks")
    parser.add_argument("--thresh_area_ub_ratio", type=float, default=0.5,
                        help="Upper bound of area for selecting masks, e.g., 0.1 for 10% of the second largest mask in target")
    parser.add_argument("--thresh_area_lb", type=float, default=30,
                        help="Lower bound of area for selecting masks, absolute value")
    parser.add_argument("--prompt", type=str, default="")

    args = parser.parse_args()

    # segment init image
    args.sam_config_coarse = {
        "points_per_side": 32,
        "pred_iou_thresh": 0.86,
        "points_per_batch": 200,
        "min_mask_region_area": 100,
        "crop_n_layers": 1,
        "crop_n_points_downscale_factor": 1,
    }
    # segment target image
    args.sam_config_fine = {
        "points_per_side": 64,
        "pred_iou_thresh": 0.87,
        "points_per_batch": 200,
        "min_mask_region_area": 30,
    }

    if args.prompt is None or args.prompt == "":
        args.prompt = get_prompt(args.target)
    args.root_folder = f"{args.output_path}/{args.output_folder}"
    args.output_folder = f"{args.root_folder}/stage_2"
    args.sam_folder = f"{args.output_folder}/sam"
    os.makedirs(args.output_folder, exist_ok=True)
    os.makedirs(args.sam_folder, exist_ok=True)

    return args


if __name__ == "__main__":
    cfg = parse_arguments()
    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"======== Stage 2: Detail Enhancement for {cfg.target} ========")

    print(f"***** SVG Processing *****")
    svg_template_path = f"{cfg.root_folder}/{cfg.target}_template.svg"
    svg_cleaned_path = f"{cfg.root_folder}/{cfg.target}_clean.svg"
    assert os.path.exists(svg_template_path), f"Input SVG file {svg_template_path} does not exist"
    svg_clean = clean_data(cfg, svg_template_path, svg_cleaned_path)
    print("Cleaned SVG file saved to", svg_cleaned_path)

    print(f"***** Image Diffusion *****")
    target_image_path = f"{cfg.root_folder}/{cfg.target}_target.png"
    if not os.path.exists(target_image_path):
        assert os.path.exists(svg_cleaned_path), f"Cleaned SVG file {svg_cleaned_path} does not exist"
        detail_enhancement(cfg, svg_cleaned_path, target_image_path, index=1)
    else:
        print(f"Target image {target_image_path} already exists")

    print(f"***** SAM *****")
    final_svg_path = f"{cfg.root_folder}/{cfg.target}_with_new_path.svg"
    if not os.path.exists(final_svg_path):
        mask_generator_coarse = initialize_sam(cfg.sam_checkpoint, cfg.sam_config_coarse, cfg.model_type, device)
        mask_generator_fine = initialize_sam(cfg.sam_checkpoint, cfg.sam_config_fine, cfg.model_type, device)
        sam_add_paths(cfg, svg_clean, target_image_path, final_svg_path, mask_generator_coarse, mask_generator_fine)
        print(f"The best SVG is: {final_svg_path}")
        print("Done!")
    else:
        print(f"Final SVG file {final_svg_path} already exists")
        print(f"The best SVG is: {final_svg_path}")
        print("Done!")
