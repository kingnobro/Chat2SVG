import pathlib
import xml.etree.ElementTree as etree

import torch
import torch.nn as nn
from PIL import Image
import pydiffvg
from torchvision import transforms
from transformers import get_cosine_schedule_with_warmup

from deepsvg.my_svg_dataset_pts import Normalize
from deepsvg.test_utils import recon_to_affine_pts
from optim_utils.util import prettify
from optim_utils.util_svg import get_cubic_segments_from_points


def initialize_latent_vectors(system_config, model_config, model, painter):
    """
    Initialize latent vectors for each path using a pre-trained VAE model.
    """
    svg_path = system_config.system_path + "/vae_dataset/circle.svg"
    png_path = svg_path.replace(".svg", ".png")

    # Load and process SVG
    _, _, shapes, _ = pydiffvg.svg_to_scene(svg_path)
    points = painter.normalize(shapes[0].points)
    assert len(
        points) == model_config.max_pts_len_thresh, f"Points length {len(points)} != {model_config.max_pts_len_thresh}"

    cubic_points = get_cubic_segments_from_points(points)
    cubic_points = cubic_points.view(1, -1, 2).unsqueeze(1).to(painter.device)

    # Load and process PNG
    path_img = painter.load_image_tensor(png_path, size=(64, 64))
    path_img = path_img.mean(dim=1, keepdim=True)  # 1x3x64x64 to 1x1x64x64

    # Generate latent vector using the VAE model
    model(args_enc=cubic_points, args_dec=cubic_points, ref_img=path_img)
    z = model.latent_z.detach().cpu().squeeze()

    # Create a list of cloned latent vectors for each path
    return [z.clone().detach().to(painter.device).requires_grad_(True) for _ in range(painter.num_paths)]


class Painter(nn.Module):
    def __init__(
            self,
            config,
            image_path: str,
            svg_path: str,
            canvas_size: int = 224,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        super(Painter, self).__init__()
        self.config = config
        self.device = device
        self.image_path = image_path
        self.svg_path = svg_path
        self.size = (canvas_size, canvas_size)
        self.canvas_width, self.canvas_height = self.size
        self.normalize = Normalize(self.canvas_width, self.canvas_height)

        # Initialize shape-related attributes
        self.shapes = []
        self.shape_groups = []
        self.cur_shapes, self.cur_shape_groups = [], []
        self.color_vars = []
        self.stroke_width_vars = []
        self.stroke_color_vars = []
        self.num_paths = 0

        # Load target image and set background color
        self.target_img_tensor = self.load_image_tensor(image_path, size=self.size)
        self.para_bg = torch.tensor([1., 1., 1.], requires_grad=False, device=self.device)

    def init_shapes(self):
        """Initialize shapes from SVG file."""
        assert self.svg_path is not None and pathlib.Path(
            self.svg_path).exists(), f"SVG file not found: {self.svg_path}"
        # print(f"-> init svg from `{self.svg_path}` ...")

        canvas_width, canvas_height, self.shapes, self.shape_groups = self.load_svg(self.svg_path)
        assert canvas_width == self.canvas_width and canvas_height == self.canvas_height, f"SVG size mismatch: {canvas_width}x{canvas_height} != {self.canvas_width}x{self.canvas_height}"

        self.num_paths = len(self.shapes)
        self.cur_shapes = self.shapes
        self.cur_shape_groups = self.shape_groups
        self.cur_cubic_curves = [self.get_cubic_curves(shape) for shape in self.cur_shapes]

    def get_image(self, step: int = 0):
        img = self.render_warp(step)
        img = img[:, :, 3:4] * img[:, :, :3] + self.para_bg * (1 - img[:, :, 3:4])
        img = img.unsqueeze(0)  # convert img from HWC to NCHW
        img = img.permute(0, 3, 1, 2).to(self.device)  # NHWC -> NCHW
        return img

    def get_path_images(self, step: int = 0):
        imgs = self.render_individual_path_warp(step)  # NHWC
        imgs = imgs[:, :, :, 3:4] * imgs[:, :, :, :3] + self.para_bg * (1 - imgs[:, :, :, 3:4])
        imgs = imgs.permute(0, 3, 1, 2).to(self.device)  # NHWC -> NCHW
        return imgs

    def render_warp(self, seed=0):
        scene_args = pydiffvg.RenderFunction.serialize_scene(
            self.canvas_width, self.canvas_height, self.shapes, self.shape_groups
        )
        _render = pydiffvg.RenderFunction.apply
        img = _render(self.canvas_width,  # width
                      self.canvas_height,  # height
                      2,  # num_samples_x
                      2,  # num_samples_y
                      seed,  # seed
                      None,
                      *scene_args)
        return img

    def render_individual_path_warp(self, seed=0):
        imgs = []
        for idx in range(len(self.shapes)):
            shape = self.shapes[idx]
            shape_group = self.shape_groups[idx]
            old_shape_ids = shape_group.shape_ids
            shape_group.shape_ids = torch.LongTensor([0]).to(self.device)

            scene_args = pydiffvg.RenderFunction.serialize_scene(
                self.canvas_width, self.canvas_height, [shape], [shape_group]
            )
            _render = pydiffvg.RenderFunction.apply
            img = _render(self.canvas_width,  # width
                          self.canvas_height,  # height
                          2,  # num_samples_x
                          2,  # num_samples_y
                          seed,  # seed
                          None,
                          *scene_args)
            imgs.append(img)

            shape_group.shape_ids = old_shape_ids

        return torch.stack(imgs)

    def init_parameters(self, model, model_config, system_config):
        self.set_stroke_parameters()
        self.set_color_parameters()
        self.set_affine_parameters()
        self.set_latent_parameters(model, model_config, system_config)

    def store_optimized_parameters(self):
        """Store the current optimized parameters."""
        optimized_params = {
            'latent_vectors': [latent.clone().detach() for latent in self.latent_vectors],
            'color_vars': [color.clone().detach() for color in self.color_vars],
            'stroke_width_vars': [width.clone().detach() for width in self.stroke_width_vars],
            'stroke_color_vars': [color.clone().detach() for color in self.stroke_color_vars],
            'translation_x': [trans.clone().detach() for trans in self.translation_x],
            'translation_y': [trans.clone().detach() for trans in self.translation_y],
            'rotation_angles': [rot.clone().detach() for rot in self.rotation_angles],
            'scale_factors': [scale.clone().detach() for scale in self.scale_factors],
        }

        # for stroke_width_vars, if width < 0.1, set it to 0
        for i, width in enumerate(optimized_params['stroke_width_vars']):
            if width < 0.1:
                optimized_params['stroke_width_vars'][i] = torch.tensor(0.0, device=self.device)

        return optimized_params

    def load_optimized_parameters(self, optimized_params):
        """Load the stored optimized parameters for existing paths."""
        if optimized_params is None:
            return

        assert len(optimized_params[
                       'latent_vectors']) <= self.num_paths, f"Number of paths mismatch: {len(optimized_params['latent_vectors'])} <= {self.num_paths}"

        num_existing_paths = len(optimized_params['latent_vectors'])
        for i in range(num_existing_paths):
            self.latent_vectors[i] = optimized_params['latent_vectors'][i].clone().detach().requires_grad_(True)
            self.color_vars[i] = optimized_params['color_vars'][i].clone().detach().requires_grad_(
                self.config.optim_color)
            self.stroke_width_vars[i] = optimized_params['stroke_width_vars'][i].clone().detach().requires_grad_(
                self.config.optim_stroke_width)
            self.stroke_color_vars[i] = optimized_params['stroke_color_vars'][i].clone().detach().requires_grad_(
                self.config.optim_stroke_color)
            self.translation_x[i] = optimized_params['translation_x'][i].clone().detach().requires_grad_(True)
            self.translation_y[i] = optimized_params['translation_y'][i].clone().detach().requires_grad_(True)
            self.rotation_angles[i] = optimized_params['rotation_angles'][i].clone().detach().requires_grad_(True)
            self.scale_factors[i] = optimized_params['scale_factors'][i].clone().detach().requires_grad_(True)

    def set_stroke_parameters(self):
        # Re-initialize stroke parameters
        self.stroke_width_vars = []
        self.stroke_color_vars = []

        for group in self.cur_shape_groups:
            group.stroke_color = torch.tensor(
                [0.0, 0.0, 0.0, 1.0],
                requires_grad=self.config.optim_stroke_color and self.config.optim_stroke_width,
                device=self.device
            )
            self.stroke_color_vars.append(group.stroke_color)

        # by default, diffvg set stroke width to 0.5; we set it to `initial_stroke_width`
        for shape in self.cur_shapes:
            shape.stroke_width = torch.tensor(
                self.config.initial_stroke_width,
                requires_grad=self.config.optim_stroke_width,
                device=self.device
            )
            self.stroke_width_vars.append(shape.stroke_width)

    def set_color_parameters(self):
        self.color_vars = []
        for group in self.cur_shape_groups:
            group.fill_color.requires_grad = self.config.optim_color
            self.color_vars.append(group.fill_color)

    def set_latent_parameters(self, model, model_config, system_config):
        self.latent_vectors = initialize_latent_vectors(system_config, model_config, model, self)

    def set_affine_parameters(self):
        self.rotation_angles = [torch.tensor(0.0, device=self.device, requires_grad=True) for _ in
                                range(self.num_paths)]
        self.scale_factors = [torch.tensor(0.2, device=self.device, requires_grad=True) for _ in range(self.num_paths)]
        self.translation_x = [torch.tensor(0.0, device=self.device, requires_grad=True) for _ in range(self.num_paths)]
        self.translation_y = [torch.tensor(0.0, device=self.device, requires_grad=True) for _ in range(self.num_paths)]

    def get_color_parameters(self):
        return self.color_vars

    def get_stroke_width_parameters(self):
        return self.stroke_width_vars

    def get_stroke_color_parameters(self):
        return self.stroke_color_vars

    def get_bg_parameters(self):
        return self.para_bg

    def get_latent_parameters(self):
        return self.latent_vectors

    def get_rotation_parameters(self):
        return self.rotation_angles

    def get_scale_parameters(self):
        return self.scale_factors

    def get_translationX_parameters(self):
        return self.translation_x

    def get_translationY_parameters(self):
        return self.translation_y

    def save_svg(self, filepath, shapes, shape_groups, use_gamma=False):
        def format_float(value):
            return f"{value:.8f}"

        root = etree.Element('svg')
        root.set('version', '1.1')
        root.set('xmlns', 'http://www.w3.org/2000/svg')
        root.set('width', str(self.canvas_width))
        root.set('height', str(self.canvas_height))
        root.set('viewBox', f"0 0 {self.canvas_width} {self.canvas_height}")
        defs = etree.SubElement(root, 'defs')
        g = etree.SubElement(root, 'g')
        if use_gamma:
            f = etree.SubElement(defs, 'filter')
            f.set('id', 'gamma')
            f.set('x', '0')
            f.set('y', '0')
            f.set('width', '100%')
            f.set('height', '100%')
            gamma = etree.SubElement(f, 'feComponentTransfer')
            gamma.set('color-interpolation-filters', 'sRGB')
            feFuncR = etree.SubElement(gamma, 'feFuncR')
            feFuncR.set('type', 'gamma')
            feFuncR.set('amplitude', str(1))
            feFuncR.set('exponent', str(1 / 2.2))
            feFuncG = etree.SubElement(gamma, 'feFuncG')
            feFuncG.set('type', 'gamma')
            feFuncG.set('amplitude', str(1))
            feFuncG.set('exponent', str(1 / 2.2))
            feFuncB = etree.SubElement(gamma, 'feFuncB')
            feFuncB.set('type', 'gamma')
            feFuncB.set('amplitude', str(1))
            feFuncB.set('exponent', str(1 / 2.2))
            feFuncA = etree.SubElement(gamma, 'feFuncA')
            feFuncA.set('type', 'gamma')
            feFuncA.set('amplitude', str(1))
            feFuncA.set('exponent', str(1 / 2.2))
            g.set('style', 'filter:url(#gamma)')

        # Store color
        for i, shape_group in enumerate(shape_groups):
            def add_color(shape_color, name):
                if isinstance(shape_color, pydiffvg.LinearGradient):
                    lg = shape_color
                    color = etree.SubElement(defs, 'linearGradient')
                    color.set('id', name)
                    color.set('x1', str(lg.begin[0].item()))
                    color.set('y1', str(lg.begin[1].item()))
                    color.set('x2', str(lg.end[0].item()))
                    color.set('y2', str(lg.end[1].item()))
                    offsets = lg.offsets.data.cpu().numpy()
                    stop_colors = lg.stop_colors.data.cpu().numpy()
                    for j in range(offsets.shape[0]):
                        stop = etree.SubElement(color, 'stop')
                        stop.set('offset', str(offsets[j]))
                        c = lg.stop_colors[j, :]
                        stop.set('stop-color', 'rgb({}, {}, {})'.format( \
                            int(255 * c[0]), int(255 * c[1]), int(255 * c[2])))
                        stop.set('stop-opacity', '{}'.format(c[3]))

            if shape_group.fill_color is not None:
                add_color(shape_group.fill_color, 'shape_{}_fill'.format(i))
            if shape_group.stroke_color is not None:
                add_color(shape_group.stroke_color, 'shape_{}_stroke'.format(i))

        for i, shape_group in enumerate(shape_groups):
            shape = shapes[shape_group.shape_ids[0]]
            if isinstance(shape, pydiffvg.Circle):
                shape_node = etree.SubElement(g, 'circle')
                shape_node.set('r', format_float(shape.radius.item()))
                shape_node.set('cx', format_float(shape.center[0].item()))
                shape_node.set('cy', format_float(shape.center[1].item()))
            elif isinstance(shape, pydiffvg.Polygon):
                shape_node = etree.SubElement(g, 'polygon')
                points = shape.points.data.cpu().numpy()
                path_str = ' '.join(
                    f"{format_float(points[j, 0])} {format_float(points[j, 1])}" for j in range(shape.points.shape[0]))
                shape_node.set('points', path_str)
            elif isinstance(shape, pydiffvg.Path):
                shape_node = etree.SubElement(g, 'path')
                num_segments = shape.num_control_points.shape[0]
                num_control_points = shape.num_control_points.data.cpu().numpy()
                points = shape.points.data.cpu().numpy()
                num_points = shape.points.shape[0]
                path_str = f"M {format_float(points[0, 0])} {format_float(points[0, 1])}"
                point_id = 1
                for j in range(0, num_segments):
                    if num_control_points[j] == 0:
                        p = point_id % num_points
                        path_str += f" L {format_float(points[p, 0])} {format_float(points[p, 1])}"
                        point_id += 1
                    elif num_control_points[j] == 1:
                        p1 = (point_id + 1) % num_points
                        path_str += f" Q {format_float(points[point_id, 0])} {format_float(points[point_id, 1])} {format_float(points[p1, 0])} {format_float(points[p1, 1])}"
                        point_id += 2
                    elif num_control_points[j] == 2:
                        p2 = (point_id + 2) % num_points
                        path_str += f" C {format_float(points[point_id, 0])} {format_float(points[point_id, 1])} {format_float(points[point_id + 1, 0])} {format_float(points[point_id + 1, 1])} {format_float(points[p2, 0])} {format_float(points[p2, 1])}"
                        point_id += 3
                shape_node.set('d', path_str)
            elif isinstance(shape, pydiffvg.Rect):
                shape_node = etree.SubElement(g, 'rect')
                shape_node.set('x', format_float(shape.p_min[0].item()))
                shape_node.set('y', format_float(shape.p_min[1].item()))
                shape_node.set('width', format_float(shape.p_max[0].item() - shape.p_min[0].item()))
                shape_node.set('height', format_float(shape.p_max[1].item() - shape.p_min[1].item()))
            elif isinstance(shape, pydiffvg.Ellipse):
                shape_node = etree.SubElement(g, 'ellipse')
                shape_node.set('cx', format_float(shape.center[0].item()))
                shape_node.set('cy', format_float(shape.center[1].item()))
                shape_node.set('rx', format_float(shape.radius[0].item()))
                shape_node.set('ry', format_float(shape.radius[1].item()))
            else:
                assert (False)

            # ignore thin strokes
            stroke_width = shape.stroke_width.data.cpu().item()
            if stroke_width <= 0.1:
                stroke_width = 0
            shape_node.set('stroke-width', format_float(stroke_width))
            if shape_group.fill_color is not None:
                if isinstance(shape_group.fill_color, pydiffvg.LinearGradient):
                    shape_node.set('fill', 'url(#shape_{}_fill)'.format(i))
                else:
                    c = shape_group.fill_color.data.cpu().numpy()
                    shape_node.set('fill', 'rgb({}, {}, {})'.format( \
                        int(255 * c[0]), int(255 * c[1]), int(255 * c[2])))
                    shape_node.set('opacity', str(c[3]))
            else:
                shape_node.set('fill', 'none')
            if shape_group.stroke_color is not None:
                if isinstance(shape_group.stroke_color, pydiffvg.LinearGradient):
                    shape_node.set('stroke', 'url(#shape_{}_stroke)'.format(i))
                else:
                    c = shape_group.stroke_color.data.cpu().numpy()
                    shape_node.set('stroke', 'rgb({}, {}, {})'.format( \
                        int(255 * c[0]), int(255 * c[1]), int(255 * c[2])))
                    shape_node.set('stroke-opacity', str(c[3]))
                shape_node.set('stroke-linecap', 'round')
                shape_node.set('stroke-linejoin', 'round')

        with open(filepath, "w") as f:
            f.write(prettify(root))

    def load_svg(self, svg_path):
        canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_to_scene(svg_path)
        return canvas_width, canvas_height, shapes, shape_groups

    def load_image_tensor(self, image_path, size):
        target = Image.open(image_path)

        if target.mode == "RGBA":
            # Composite the image onto a white background
            background = Image.new("RGBA", target.size, "WHITE")
            background.paste(target, (0, 0), target)
            target = background.convert("RGB")
        else:
            target = target.convert("RGB")

        # Resize the image if necessary
        if target.size != size:
            target = target.resize(size, Image.Resampling.BICUBIC)

        # Convert the image to a tensor
        transform = transforms.ToTensor()
        image_tensor = transform(target).unsqueeze(0).to(self.device)

        return image_tensor

    def get_cubic_curves(self, path: pydiffvg.Path) -> torch.Tensor:
        """
        Extracts and organizes cubic Bezier curves from a given path.
        """
        points = path.points
        num_control_points = path.num_control_points.cpu().numpy()
        num_points = points.shape[0]
        assert num_points % 3 == 0, "Number of points must be divisible by 3"

        # Calculate starting indices for each cubic curve
        indices = torch.arange(0, len(num_control_points) * 3, step=3, device=points.device)

        # Gather the four points for each cubic curve
        point1 = points[indices]
        point2 = points[indices + 1]
        point3 = points[indices + 2]
        point4 = points[(indices + 3) % num_points]

        # Stack the points to form cubic curves
        cubics = torch.stack([point1, point2, point3, point4], dim=1)

        return cubics

    def convert_points_to_path(self, points):
        num_segments = points.shape[0] // 3
        num_control_points = torch.full((num_segments,), 2, dtype=torch.long)

        # Create a path object
        path = pydiffvg.Path(
            num_control_points=num_control_points,
            points=points,
            is_closed=True
        )
        return path

    def apply_affine_transform(self, points, idx):
        transformed_points, _ = recon_to_affine_pts(
            recon_data_output=points[idx],
            theta=self.rotation_angles[idx],
            tx=self.translation_x[idx],
            ty=self.translation_y[idx],
            s=self.scale_factors[idx],
            s_norm=self.normalize,
            h=self.config.output_size,
            w=self.config.output_size,
            use_affine_norm=False
        )
        return transformed_points

    def update_color_and_stroke_width(self, paths, fill_colors, stroke_widths=None, stroke_colors=None):
        shapes = []
        shape_groups = []

        for i, path in enumerate(paths):
            if stroke_widths:
                path.stroke_width = stroke_widths[i]
            shapes.append(path)

            shape_group = pydiffvg.ShapeGroup(
                shape_ids=torch.LongTensor([i]),
                fill_color=fill_colors[i],
                stroke_color=stroke_colors[i] if stroke_colors else None,
                use_even_odd_rule=False
            )
            shape_groups.append(shape_group)

        return shapes, shape_groups

    def convert_points_to_shapes(self, points):
        paths = []
        for idx in range(self.num_paths):
            transformed_points = self.apply_affine_transform(points, idx)
            paths.append(self.convert_points_to_path(transformed_points))

        optimized_shapes, optimized_groups = self.update_color_and_stroke_width(paths=paths,
                                                                                fill_colors=self.color_vars)
        return optimized_shapes, optimized_groups

    def render_image(self, paths):
        canvas_width, canvas_height = self.config.output_size, self.config.output_size
        render_function = pydiffvg.RenderFunction.apply

        shapes, shape_groups = self.update_color_and_stroke_width(
            paths,
            fill_colors=self.color_vars,
            stroke_widths=self.stroke_width_vars,
            stroke_colors=self.stroke_color_vars
        )

        scene_args = pydiffvg.RenderFunction.serialize_scene(
            canvas_width, canvas_height, shapes, shape_groups
        )
        rendered_image = render_function(
            canvas_width, canvas_height,
            2, 2,  # num_samples_x, num_samples_y
            0,  # seed
            None,  # background
            *scene_args
        )

        # Compose image with background
        alpha_channel = rendered_image[:, :, 3:4]
        foreground = alpha_channel * rendered_image[:, :, :3]
        background = self.para_bg * (1 - alpha_channel)
        image = foreground + background

        # HWC -> NCHW
        image_tensor = image.unsqueeze(0).permute(0, 3, 1, 2)

        return image_tensor, image

    def target_file_preprocess(self, tar_path, output_size):
        process_comp = transforms.Compose([
            transforms.Resize(size=(output_size, output_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: t.unsqueeze(0)),
        ])

        tar_pil = Image.open(tar_path).convert("RGB")  # open file
        target_img = process_comp(tar_pil)  # preprocess
        target_img = target_img.to(self.device)
        return target_img


class PointPainter(Painter):
    def __init__(
            self,
            config,
            image_path: str,
            svg_path: str,
            canvas_size: int = 224,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        super().__init__(config, image_path, svg_path, canvas_size, device)
        self.point_vars = []

    def set_points_parameters(self, id_delta=0):
        self.point_vars = []
        for i, path in enumerate(self.cur_shapes):
            path.id = i + id_delta  # set point id
            path.points.requires_grad = True
            self.point_vars.append(path.points)

    def set_stroke_parameters(self):
        """
        load stroke parameters from svg, instead of using the `initial_stroke_width`
        """
        # Re-initialize stroke parameters
        self.stroke_width_vars = []
        self.stroke_color_vars = []

        for group in self.cur_shape_groups:
            group.stroke_color = torch.tensor(
                group.stroke_color,
                requires_grad=self.config.optim_stroke_color and self.config.optim_stroke_width,
                device=self.device
            )
            self.stroke_color_vars.append(group.stroke_color)

        # load stroke width from svg
        for shape in self.cur_shapes:
            shape.stroke_width = torch.tensor(
                shape.stroke_width * 2,  # diffvg.svg_to_scene will divide it by 2
                requires_grad=self.config.optim_stroke_width,
                device=self.device
            )
            self.stroke_width_vars.append(shape.stroke_width)

    def get_point_parameters(self):
        return self.point_vars

    def get_paths(self):
        return self.cur_shapes

    def init_parameters(self):
        self.set_points_parameters()
        self.set_stroke_parameters()
        self.set_color_parameters()


class PainterOptimizer:
    def __init__(self,
                 config,
                 lr_config,
                 renderer: Painter,
                 num_training_steps: int,
                 num_warmup_steps: int):
        self.config = config
        self.renderer = renderer
        self.num_training_steps = num_training_steps
        self.num_warmup_steps = num_warmup_steps
        self.lr_base = lr_config
        self.optimizers = {}
        self.schedulers = {}

    def init_optimizers(self):
        # Parameters
        params_latent = self.renderer.get_latent_parameters()
        params_color = self.renderer.get_color_parameters()
        params_affine = [
            {'params': self.renderer.get_rotation_parameters(), 'lr': self.lr_base['rotation']},
            {'params': self.renderer.get_scale_parameters(), 'lr': self.lr_base['scale']},
            {'params': self.renderer.get_translationX_parameters(), 'lr': self.lr_base['translation']},
            {'params': self.renderer.get_translationY_parameters(), 'lr': self.lr_base['translation']},
        ]
        params_stroke_width = self.renderer.get_stroke_width_parameters()
        params_stroke_color = self.renderer.get_stroke_color_parameters()

        # Optimizers
        optimizer_latent = torch.optim.Adam(params_latent, lr=self.lr_base['latent'], betas=(0.9, 0.9), eps=1e-6)
        optimizer_color = torch.optim.Adam(params_color, lr=self.lr_base['color'], betas=(0.9, 0.9), eps=1e-6)
        optimizer_affine = torch.optim.Adam(params_affine, betas=(0.9, 0.9), eps=1e-6)
        optimizer_stroke_width = torch.optim.Adam(params_stroke_width, lr=self.lr_base['stroke_width'],
                                                  betas=(0.9, 0.9), eps=1e-6)
        optimizer_stroke_color = torch.optim.Adam(params_stroke_color, lr=self.lr_base['stroke_color'],
                                                  betas=(0.9, 0.9), eps=1e-6)

        # Schedulers
        scheduler_latent = get_cosine_schedule_with_warmup(optimizer_latent, num_warmup_steps=self.num_warmup_steps,
                                                           num_training_steps=self.num_training_steps)
        scheduler_color = get_cosine_schedule_with_warmup(optimizer_color, num_warmup_steps=self.num_warmup_steps,
                                                          num_training_steps=self.num_training_steps)
        scheduler_affine = get_cosine_schedule_with_warmup(optimizer_affine, num_warmup_steps=self.num_warmup_steps,
                                                           num_training_steps=self.num_training_steps)
        scheduler_stroke_width = get_cosine_schedule_with_warmup(optimizer_stroke_width,
                                                                 num_warmup_steps=self.num_warmup_steps,
                                                                 num_training_steps=self.num_training_steps)
        scheduler_stroke_color = get_cosine_schedule_with_warmup(optimizer_stroke_color,
                                                                 num_warmup_steps=self.num_warmup_steps,
                                                                 num_training_steps=self.num_training_steps)

        # Store optimizers and schedulers
        self.optimizers = {
            'latent': optimizer_latent,
            'color': optimizer_color,
            'affine': optimizer_affine,
            'stroke_width': optimizer_stroke_width,
            'stroke_color': optimizer_stroke_color,
        }
        self.schedulers = {
            'latent': scheduler_latent,
            'color': scheduler_color,
            'affine': scheduler_affine,
            'stroke_width': scheduler_stroke_width,
            'stroke_color': scheduler_stroke_color,
        }

    def update_lr(self):
        for scheduler in self.schedulers.values():
            scheduler.step()

    def zero_grad_(self):
        for optimizer in self.optimizers.values():
            optimizer.zero_grad()

    def step_(self):
        if not self.config.optim_opacity:
            for color in self.renderer.get_color_parameters():
                if color.grad is not None:
                    color.grad[3] = torch.zeros_like(color.grad[3])

            # for color in self.renderer.get_stroke_color_parameters():
            #     if color.grad is not None:
            #         color.grad[3] = torch.zeros_like(color.grad[3])

        for optimizer in self.optimizers.values():
            optimizer.step()

        if self.config.optim_stroke_width:
            for stroke_width in self.renderer.get_stroke_width_parameters():
                stroke_width.data.clamp_(0.0, 2.0)

        if self.config.optim_color:
            for color in self.renderer.get_color_parameters():
                color.data.clamp_(0.0, 1.0)

        if self.config.optim_stroke_color:
            for color in self.renderer.get_stroke_color_parameters():
                color.data.clamp_(0.0, 1.0)


class PointPainterOptimizer(PainterOptimizer):
    def __init__(self,
                 config,
                 lr_config,
                 renderer: PointPainter,
                 num_training_steps: int,
                 num_warmup_steps: int):
        super().__init__(config, lr_config, renderer, num_training_steps, num_warmup_steps)

    def init_optimizers(self):
        # Parameters
        params_points = self.renderer.get_point_parameters()
        params_color = self.renderer.get_color_parameters()
        params_stroke_width = self.renderer.get_stroke_width_parameters()
        params_stroke_color = self.renderer.get_stroke_color_parameters()

        # Optimizers
        optimizer_points = torch.optim.Adam(params_points, lr=self.lr_base['points'], betas=(0.9, 0.9), eps=1e-6)
        optimizer_color = torch.optim.Adam(params_color, lr=self.lr_base['color'], betas=(0.9, 0.9), eps=1e-6)
        optimizer_stroke_width = torch.optim.Adam(params_stroke_width, lr=self.lr_base['stroke_width'],
                                                  betas=(0.9, 0.9), eps=1e-6)
        optimizer_stroke_color = torch.optim.Adam(params_stroke_color, lr=self.lr_base['stroke_color'],
                                                  betas=(0.9, 0.9), eps=1e-6)

        # Schedulers
        scheduler_points = get_cosine_schedule_with_warmup(optimizer_points, num_warmup_steps=self.num_warmup_steps,
                                                           num_training_steps=self.num_training_steps)
        scheduler_color = get_cosine_schedule_with_warmup(optimizer_color, num_warmup_steps=self.num_warmup_steps,
                                                          num_training_steps=self.num_training_steps)
        scheduler_stroke_width = get_cosine_schedule_with_warmup(optimizer_stroke_width,
                                                                 num_warmup_steps=self.num_warmup_steps,
                                                                 num_training_steps=self.num_training_steps)
        scheduler_stroke_color = get_cosine_schedule_with_warmup(optimizer_stroke_color,
                                                                 num_warmup_steps=self.num_warmup_steps,
                                                                 num_training_steps=self.num_training_steps)

        # Store optimizers and schedulers
        self.optimizers = {
            'points': optimizer_points,
            'color': optimizer_color,
            'stroke_width': optimizer_stroke_width,
            'stroke_color': optimizer_stroke_color,
        }
        self.schedulers = {
            'points': scheduler_points,
            'color': scheduler_color,
            'stroke_width': scheduler_stroke_width,
            'stroke_color': scheduler_stroke_color,
        }
