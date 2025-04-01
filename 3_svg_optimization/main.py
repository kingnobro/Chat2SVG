import os
import sys
import argparse
import time

import numpy as np
import yaml
import torch
from transformers import set_seed
from PIL import Image

from deepsvg.model.config import _DefaultConfig
from deepsvg.model.model_pts_vae import SVGTransformer

sys.path.append("..")
from svglib.svg import SVG
from svglib.geom import Bbox
from losses import laplacian_smoothing_loss, kl_divergence, svg_emd_loss, IoULoss, curvature_loss
from painter import Painter, PainterOptimizer, PointPainter, PointPainterOptimizer
from optim_utils.util_svg import (
    get_cubic_segments_from_points,
    sample_bezier,
    make_clockwise,
    sample_points_by_length_distribution,
)
from optim_utils.util import setup_logger, init_diffvg


def latent_inversion(config, painter: Painter, model):
    logger.info("----- latent inversion -----")
    start_time = time.time()
    device = painter.device

    samples_per_cubic = config.samples_per_cubic
    target_data = []
    for cubic_points in painter.cur_cubic_curves:
        sampled_points = sample_bezier(cubic_points, samples_per_cubic).to(device)
        clockwise_points = make_clockwise(sampled_points)
        pred_samples_count = config.max_total_len // 4 * samples_per_cubic
        target_subset, matching_indices = sample_points_by_length_distribution(
            p_target=clockwise_points, n=pred_samples_count, device=device
        )
        target_data.append({
            'sampled_points': clockwise_points,
            'target_subset': target_subset,
            'matching_indices': matching_indices
        })

    painter_optimizer = PainterOptimizer(
        config=config,
        lr_config=config.lr_config_latent,
        renderer=painter,
        num_training_steps=config.epoch_latent_inversion,
        num_warmup_steps=config.num_warmup_steps
    )
    painter_optimizer.init_optimizers()

    smoothness_weight = config.smoothness_weight_latent
    kl_weight = config.kl_weight_latent

    for epoch in range(config.epoch_latent_inversion):
        latent_batch = torch.stack(painter.get_latent_parameters()).to(device).squeeze(1)
        generated_data = model(args_enc=None, args_dec=None, z=latent_batch.unsqueeze(1).unsqueeze(2))
        reconstructed_points = generated_data["args_logits"].squeeze(1)  # [num_path, 4 * num_bezier_curve, 2]

        if kl_weight > 0:
            kl_loss = kl_divergence(latent_batch) * kl_weight
        else:
            kl_loss = 0.0

        total_emd_loss = 0.0
        smoothness_loss = 0.0

        for idx in range(painter.num_paths):
            transformed_points = painter.apply_affine_transform(reconstructed_points, idx)

            if smoothness_weight > 0:
                path_smoothness_loss = laplacian_smoothing_loss(transformed_points)
                smoothness_loss += path_smoothness_loss / transformed_points.size(0)

            cubic_points = get_cubic_segments_from_points(transformed_points)
            sampled_pred_points = sample_bezier(cubic_points.view(-1, 4, 2), samples_per_cubic)

            emd_loss = svg_emd_loss(
                p_pred=sampled_pred_points,
                p_target=target_data[idx]['sampled_points'],
                p_target_sub=target_data[idx]['target_subset'],
                matching=target_data[idx]['matching_indices'],
            )
            total_emd_loss += emd_loss

        total_emd_loss /= painter.num_paths
        smoothness_loss = smoothness_loss * smoothness_weight / painter.num_paths

        total_loss = total_emd_loss + smoothness_loss + kl_loss

        if epoch == 0 or (epoch + 1) % config.log_every == 0:
            logger.info(f"epoch: {epoch + 1}, total_loss: {total_loss.item():.4f}")

        painter_optimizer.zero_grad_()
        total_loss.backward()
        painter_optimizer.step_()
        painter_optimizer.update_lr()

    optimized_shapes, optimized_groups = painter.convert_points_to_shapes(reconstructed_points)
    painter.save_svg(f"{config.svg_dir}/{config.target}_inverted.svg", optimized_shapes, optimized_groups)

    end_time = time.time()
    logger.info(f"Latent inversion completed in {end_time - start_time:.2f} seconds")


def latent_optimization(config, painter: Painter, model):
    logger.info("----- image optimization -----")
    start_time = time.time()
    device = painter.device

    painter_optimizer = PainterOptimizer(
        config=config,
        lr_config=config.lr_config_img,
        renderer=painter,
        num_training_steps=config.epoch_img_optim * 2,  # extend cos schedule
        num_warmup_steps=config.num_warmup_steps
    )
    painter_optimizer.init_optimizers()

    if config.enable_path_iou_loss:
        iou_loss = IoULoss()
        svg = SVG.load_svg(config.color_align_svg_path)
        path_img_masks = []
        for i, svg_path_group in enumerate(svg.svg_path_groups):
            path_svg = SVG([svg_path_group], svg.viewbox)
            path_svg_path = os.path.join(config.path_mask_dir, f"path_{i}.png")
            path_svg.save_png(path_svg_path)
            path_img = painter.target_file_preprocess(path_svg_path, config.output_size)
            path_img_grey = torch.mean(path_img, dim=1, keepdim=True)
            path_img_mask = (path_img_grey < 1.0).float()
            path_img_masks.append(path_img_mask.squeeze(0))
        path_img_masks_init = torch.stack(path_img_masks)

    optimized_shapes = None
    optimized_shape_groups = None

    smoothness_weight = config.smoothness_weight_img
    kl_weight = config.kl_weight_img
    mse_loss_weight = config.mse_loss_weight_img
    path_iou_loss_weight = config.path_iou_loss_weight_img

    for epoch in range(config.epoch_img_optim):
        latent_batch = torch.stack(painter.get_latent_parameters()).to(device).squeeze(1)
        generated_data = model(args_enc=None, args_dec=None, z=latent_batch.unsqueeze(1).unsqueeze(2))
        reconstructed_points = generated_data["args_logits"].squeeze(1)

        paths = []
        smoothness_loss = 0.0

        for idx in range(painter.num_paths):
            transformed_points = painter.apply_affine_transform(reconstructed_points, idx)

            # if smoothness_weight > 0:
            #     path_smoothness_loss = laplacian_smoothing_loss(transformed_points)
            #     smoothness_loss += path_smoothness_loss / transformed_points.size(0)

            paths.append(painter.convert_points_to_path(transformed_points))

        kl_loss = kl_divergence(latent_batch) * kl_weight if kl_weight > 0 else 0.0
        loss_curvature = curvature_loss(paths) * config.curvature_loss_weight_img
        smoothness_loss = smoothness_loss * smoothness_weight / painter.num_paths

        image_tensor, image = painter.render_image(paths)

        if config.enable_path_iou_loss:
            path_imgs = painter.get_path_images(step=epoch).to(device)
            path_imgs_grey = torch.mean(path_imgs, dim=1, keepdim=True)
            path_imgs_mask = 2 * torch.sigmoid(10 * (1 - path_imgs_grey)) - 1
            path_iou_loss = iou_loss(path_imgs_mask, path_img_masks_init) * path_iou_loss_weight
        else:
            path_iou_loss = 0.0

        mse_loss = torch.nn.functional.mse_loss(image_tensor, painter.target_img_tensor) * mse_loss_weight
        total_loss = mse_loss + smoothness_loss + kl_loss + path_iou_loss + loss_curvature

        if epoch == 0 or (epoch + 1) % config.log_every == 0:
            logger.info(f"epoch: {epoch + 1}, total_loss: {total_loss.item():.4f}")
            optimized_shapes, optimized_shape_groups = painter.update_color_and_stroke_width(
                paths=paths,
                fill_colors=painter.get_color_parameters(),
                stroke_widths=painter.get_stroke_width_parameters(),
                stroke_colors=painter.get_stroke_color_parameters()
            )
            painter.save_svg(f"{config.svg_dir}/{config.target}_optim_{epoch + 1}.svg", optimized_shapes,
                             optimized_shape_groups)
            # save image to PIL
            img = Image.fromarray((image.detach().cpu().numpy() * 255).astype(np.uint8))
            img.save(f"{config.png_dir}/image_{epoch + 1}.png")

        painter_optimizer.zero_grad_()
        total_loss.backward()
        painter_optimizer.step_()
        painter_optimizer.update_lr()

    end_time = time.time()
    logger.info(f"Latent optimization completed in {end_time - start_time:.2f} seconds")
    return optimized_shapes, optimized_shape_groups


def prepare_svg(config):
    target = config.target
    logger.info(f"[Target] {os.path.basename(target)}")

    # Load and normalize SVG to 256x256
    input_svg_path = f"{config.svg_folder}/{target}_with_new_path.svg"
    svg = SVG.load_svg(input_svg_path)
    svg.normalize(Bbox(config.output_size))
    normalized_svg_path = f"{config.svg_dir}/{target}_normalized_256.svg"
    svg.save_svg(normalized_svg_path, coordinate_precision=3)

    return normalized_svg_path


def optimization(config, model, model_config):
    canvas_size = config.output_size
    target_image_path = config.target_image_path
    normalized_svg_path = prepare_svg(config)

    # NOTE: You can use the following code that aligns the color of each shape with the region in the target image to ease the optimization
    # target_image = Image.open(target_image_path).convert('RGB').resize((canvas_size, canvas_size))
    # svg_color_align = align_path_color_to_image(svg_init, target_image, logger)
    # color_align_svg_path = f"{config.svg_dir}/{config.target}_color_aligned.svg"
    # config.color_align_svg_path = color_align_svg_path
    # svg_color_align.drop_z()
    # svg_color_align.save_svg(color_align_svg_path, coordinate_precision=3)

    painter = Painter(
        config,
        target_image_path,
        normalized_svg_path,  # OR: color_align_svg_path
        canvas_size=canvas_size,
        device=device,
    )
    painter.init_shapes()
    painter.init_parameters(model, model_config, config)

    latent_inversion(config=config, painter=painter, model=model)
    shapes, shape_groups = latent_optimization(config, painter, model)

    # Save optimized SVG
    painter.save_svg(config.svg_latent_optim_path, shapes, shape_groups)
    logger.info(f"Optimized SVG saved to: {config.svg_latent_optim_path}")


def point_optimization(config):
    logger.info("----- point optimization -----")
    start_time = time.time()

    canvas_size = config.output_size
    target_image_path = config.target_image_path
    svg_input_path = config.svg_latent_optim_path
    svg_output_path = config.svg_point_optim_path

    # normalize 256 to 512
    canvas_size = config.output_size = 512
    svg_input_path_normalized = f"{config.svg_point_dir}/{config.target}_normalized_512.svg"
    svg_256 = SVG.load_svg(svg_input_path)
    svg_256.normalize(Bbox(canvas_size))
    # svg_256.filter_consecutives().filter_empty()  # may cause error; `filter_consecutives` will make start and end point not equal
    for i, path_group in enumerate(svg_256.svg_path_groups):
        area = path_group.to_shapely().area
        if area > 5000:
            svg_256.svg_path_groups[i] = path_group.split(3)
        elif area > 200:
            svg_256.svg_path_groups[i] = path_group.split(2)
    svg_256.save_svg(svg_input_path_normalized, coordinate_precision=6)

    painter = PointPainter(
        config,
        target_image_path,
        svg_input_path_normalized,
        canvas_size=canvas_size,
        device=device,
    )
    painter.init_shapes()
    painter.init_parameters()

    painter_optimizer = PointPainterOptimizer(
        config=config,
        lr_config=config.lr_config_point,
        renderer=painter,
        num_training_steps=config.epoch_point_optim * 2,  # extend cos schedule
        num_warmup_steps=config.num_warmup_steps
    )
    painter_optimizer.init_optimizers()

    for epoch in range(config.epoch_point_optim):
        # render current image
        paths = painter.get_paths()
        image_tensor, image = painter.render_image(paths)
        curvature_loss_weight_start = config.curvature_loss_weight_start_point
        curvature_loss_weight_end = config.curvature_loss_weight_end_point

        mse_loss = torch.nn.functional.mse_loss(image_tensor, painter.target_img_tensor)
        loss_curvature = curvature_loss(paths)
        curvature_loss_weight = curvature_loss_weight_start - (
                    curvature_loss_weight_start - curvature_loss_weight_end) * epoch / config.epoch_point_optim
        total_loss = config.mse_loss_weight_point * mse_loss + curvature_loss_weight * loss_curvature

        if epoch == 0 or (epoch + 1) % config.log_every == 0:
            logger.info(f"epoch: {epoch + 1}, total_loss: {total_loss.item():.6f}")

            # Save intermediate results
            optimized_shapes, optimized_shape_groups = painter.update_color_and_stroke_width(
                paths=paths,
                fill_colors=painter.get_color_parameters(),
                stroke_widths=painter.get_stroke_width_parameters(),
                stroke_colors=painter.get_stroke_color_parameters()
            )
            painter.save_svg(f"{config.svg_point_dir}/{config.target}_point_optim_{epoch + 1}.svg", optimized_shapes,
                             optimized_shape_groups)

            # Save rendered image
            img = Image.fromarray((image.detach().cpu().numpy() * 255).astype(np.uint8))
            img.save(f"{config.png_point_dir}/image_point_optim_{epoch + 1}.png")

        painter_optimizer.zero_grad_()
        total_loss.backward()
        painter_optimizer.step_()
        painter_optimizer.update_lr()

    # Save final optimized SVG
    final_shapes, final_shape_groups = painter.update_color_and_stroke_width(
        paths=paths,
        fill_colors=painter.get_color_parameters(),
        stroke_widths=painter.get_stroke_width_parameters(),
        stroke_colors=painter.get_stroke_color_parameters()
    )
    painter.save_svg(svg_output_path, final_shapes, final_shape_groups)
    logger.info(f"Point-optimized SVG saved to: {svg_output_path}")

    # Save final rendered image
    _, final_image = painter.render_image(paths)
    final_img = Image.fromarray((final_image.detach().cpu().numpy() * 255).astype(np.uint8))
    final_img.save(f"{config.png_point_dir}/image_point_optim_final.png")

    end_time = time.time()
    logger.info(f"Point optimization completed in {end_time - start_time:.2f} seconds")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, required=True, help="The specific target to be optimized")
    parser.add_argument("--svg_folder", type=str, required=True, help="Folder name of input images and svg")
    parser.add_argument("--system_path", type=str, default=".")
    parser.add_argument("--output_size", type=int, default=224, help="Output image size")

    # optimization parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--optim_color", action="store_true", default=True,
                        help="Whether to optimize color during image optimization")
    parser.add_argument("--optim_opacity", action="store_true", default=False,
                        help="Whether to optimize opacity during image optimization")
    parser.add_argument("--optim_stroke_width", action="store_true", default=True,
                        help="Whether to optimize stroke width during image optimization")
    parser.add_argument("--optim_stroke_color", action="store_true", default=True,
                        help="Whether to optimize stroke color during image optimization")

    parser.add_argument("--initial_stroke_width", type=float, default=0.8,
                        help="Initial stroke width")
    parser.add_argument("--epoch_latent_inversion", type=int, default=500,
                        help="Number of epochs for latent inversion")
    parser.add_argument("--epoch_img_optim", type=int, default=500,
                        help="Number of epochs for image optimization")
    parser.add_argument("--epoch_point_optim", type=int, default=500,
                        help="Number of epochs for point optimization")
    parser.add_argument("--num_warmup_steps", type=int, default=100,
                        help="Number of epochs for warmup")
    parser.add_argument("--log_every", type=int, default=100,
                        help="Number of epochs to log the loss")

    # latent inversion
    parser.add_argument("--smoothness_weight_latent", type=float, default=0.0,
                        help="Weight for smoothness loss in latent inversion")
    parser.add_argument("--kl_weight_latent", type=float, default=0.1,
                        help="Weight for KL divergence loss in latent inversion")

    # latent optimization
    parser.add_argument("--kl_weight_img", type=float, default=0.2,
                        help="Weight for KL divergence loss in latent optimization")
    parser.add_argument("--smoothness_weight_img", type=float, default=2.0,
                        help="Weight for smoothness loss in image optimization")
    parser.add_argument("--mse_loss_weight_img", type=float, default=2000.0,
                        help="Weight for MSE loss in image optimization")
    parser.add_argument("--enable_path_iou_loss", action="store_true", default=False,
                        help="Whether to use path iou loss in image optimization")
    parser.add_argument("--path_iou_loss_weight_img", type=float, default=1.0,
                        help="Weight for path iou loss in image optimization")
    parser.add_argument("--curvature_loss_weight_img", type=float, default=0.01,
                        help="Weight for curvature loss in image optimization")

    # point optimization
    parser.add_argument("--mse_loss_weight_point", type=float, default=2000.0,
                        help="Weight for mse loss in point optimization")
    parser.add_argument("--smoothness_loss_weight_point", type=float, default=0.00,
                        help="Weight for smoothness loss in point optimization")
    parser.add_argument("--c1_loss_weight_point", type=float, default=0.0,
                        help="Weight for c1 continuity loss in point optimization")
    parser.add_argument("--curvature_loss_weight_start_point", type=float, default=2.0,
                        help="Weight for curvature loss in point optimization")
    parser.add_argument("--curvature_loss_weight_end_point", type=float, default=0.1,
                        help="Weight for curvature loss in point optimization")
    parser.add_argument("--samples_per_cubic", type=int, default=64,
                        help="Number of samples per cubic in image optimization")

    # vae config
    parser.add_argument("--vae_optim_config", type=str, default="./configs/vae_config_cmd_10.yaml",
                        help="Path to the VAE optimization config file")
    parser.add_argument("--vae_pretrained_path", type=str, default="./vae_model/cmd_10.pth",
                        help="Path to the VAE pretrained model")

    args = parser.parse_args()

    with open(args.vae_optim_config, 'r') as f:
        vae_optim_config = yaml.safe_load(f)

    # load max_total_len
    setattr(args, "max_total_len", vae_optim_config["max_total_len"])

    args.target_image_path = f"{args.svg_folder}/{args.target}_target.png"
    args.svg_latent_optim_path = f"{args.svg_folder}/{args.target}_optim_latent.svg"
    args.svg_point_optim_path = f"{args.svg_folder}/{args.target}_optim_point.svg"

    # Set optimization parameters
    lr_seq_fac = 10.0
    lr_color = 0.2
    lr_stroke_width = 0.1
    lr_stroke_color = 0.1
    lr_point = 0.5

    # Sequence optimization learning rates
    lr_seq_latent = 0.1
    lr_seq_translation = 0.5 * lr_seq_fac
    lr_seq_rotation = 0.006 * lr_seq_fac
    lr_seq_scale = 0.05 * lr_seq_fac

    # Image optimization learning rates
    lr_img_latent = 0.03
    lr_img_translation = 0.1
    lr_img_rotation = 0.003
    lr_img_scale = 0.003

    args.lr_config_latent = {
        "latent": lr_seq_latent,
        "color": 0,
        "translation": lr_seq_translation,
        "rotation": lr_seq_rotation,
        "scale": lr_seq_scale,
        "stroke_width": 0,
        "stroke_color": 0,
    }

    args.lr_config_img = {
        "latent": lr_img_latent,
        "color": lr_color,
        "translation": lr_img_translation,
        "rotation": lr_img_rotation,
        "scale": lr_img_scale,
        "stroke_width": lr_stroke_width,
        "stroke_color": lr_stroke_color,
    }

    args.lr_config_point = {
        "points": lr_point,
        "stroke_width": lr_stroke_width,
        "stroke_color": lr_stroke_color,
        "color": lr_color,
    }

    # Set up output directories
    args.output_folder = f"{args.svg_folder}/stage_3"
    args.svg_dir = f"{args.output_folder}/latent_svg"
    args.png_dir = f"{args.output_folder}/latent_png"
    args.path_mask_dir = f"{args.output_folder}/path_mask"
    args.svg_point_dir = f"{args.output_folder}/point_svg"
    args.png_point_dir = f"{args.output_folder}/point_png"

    for dir_path in [args.output_folder, args.svg_dir, args.png_dir, args.path_mask_dir, args.svg_point_dir,
                     args.png_point_dir]:
        os.makedirs(dir_path, exist_ok=True)

    # Save config
    with open(f'{args.output_folder}/config.yaml', 'w') as f:
        yaml.dump(vars(args), f)

    return args


def load_model_config(config):
    model_config = _DefaultConfig()
    with open(config.vae_optim_config, 'r') as f:
        config_data = yaml.safe_load(f)
    for key, value in config_data.items():
        setattr(model_config, key, value)
    model_config.img_latent_dim = int(model_config.d_img_model / 64.0)
    model_config.vq_edim = int(model_config.dim_z / model_config.vq_comb_num)
    return model_config


if __name__ == "__main__":
    config = parse_arguments()
    logger = setup_logger(config.output_folder)
    set_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    init_diffvg(device)

    # load VAE
    model_config = load_model_config(config)
    model = SVGTransformer(model_config).to(device)
    state = torch.load(config.vae_pretrained_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()

    start_time = time.time()
    if not os.path.exists(config.svg_latent_optim_path):
        optimization(config, model, model_config)  # latent optimization
    else:
        logger.info(f"Optimized SVG already exists: {config.svg_latent_optim_path}")

    if not os.path.exists(config.svg_point_optim_path):
        point_optimization(config)
        print(f"The best SVG is: {config.svg_point_optim_path}")
        print("Done!")
    else:
        logger.info(f"Point optimized SVG already exists: {config.svg_point_optim_path}")
        print(f"The best SVG is: {config.svg_point_optim_path}")
        print("Done!")

    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"Total optimization time: {elapsed_time:.2f} seconds")
