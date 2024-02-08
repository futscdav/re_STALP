import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
import torchvision.models as models
import torchvision.io
import numpy as np
import rotate

from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch import profiler
from torchvision.transforms import functional as Tfunc
from contextlib import suppress as suppress
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from tqdm import tqdm

from futscml_exports import *
from config import TRANSFER_CONFIG as config

# Don't need to import here but torchvision video would fail without it.
import av


ModuleProducer = Callable[..., torch.nn.Module]


_KEY_DATASET_PAIRS = 'pairs_dataset'
_KEY_DATASET_AUXILIARY = 'aux_dataset'
_KEY_DATASET_STYLEFRAMES = 'style_dataset'
_KEY_DATASET_VALIDATE = 'val_dataset'


class ImageToImageGenerator_JohnsonFutschik(nn.Module):
  """Default neural net model architecture to use for im2im."""

  def __init__(
          self, norm_layer: str = 'batch_norm', use_bias: bool = False,
          resnet_blocks: int = 9, tanh: bool = False,
          filters: Sequence[int] = (64, 128, 128, 128, 128, 64),
          width: float = 1.0,
          input_channels: int = 3, output_channels: int = 3,
          append_blocks: bool = False):
    super().__init__()
    self.norm_layer = {
        'batch_norm': nn.BatchNorm2d, 'instance_norm': nn.InstanceNorm2d,
        None: None}[norm_layer]
    self.use_bias = use_bias
    self.resnet_blocks = resnet_blocks
    self.append_blocks = append_blocks

    # Adjust filters.
    filters = [int(f * width) for f in filters]

    self.conv0 = self.relu_layer(
        in_filters=input_channels, out_filters=filters[0],
        size=7, stride=1, padding=3, bias=self.use_bias,
        norm_layer=self.norm_layer, nonlinearity=nn.LeakyReLU(.2))

    self.conv1 = self.relu_layer(
        in_filters=filters[0],
        out_filters=filters[1],
        size=3, stride=2, padding=1, bias=self.use_bias,
        norm_layer=self.norm_layer, nonlinearity=nn.LeakyReLU(.2))

    self.conv2 = self.relu_layer(
        in_filters=filters[1],
        out_filters=filters[2],
        size=3, stride=2, padding=1, bias=self.use_bias,
        norm_layer=self.norm_layer, nonlinearity=nn.LeakyReLU(.2))

    self.resnets = nn.ModuleList()
    for _ in range(self.resnet_blocks):
      self.resnets.append(
          self.resnet_block(
              in_filters=filters[2],
              out_filters=filters[2],
              size=3, stride=1, padding=1, bias=self.use_bias,
              norm_layer=self.norm_layer, nonlinearity=nn.ReLU()))

    self.upconv2 = self.upconv_layer(
        in_filters=filters[3] + filters[2],
        out_filters=filters[4],
        size=4, stride=2, padding=1, bias=self.use_bias,
        norm_layer=self.norm_layer, nonlinearity=nn.ReLU())

    self.upconv1 = self.upconv_layer(
        in_filters=filters[4] + filters[1],
        out_filters=filters[4],
        size=4, stride=2, padding=1, bias=self.use_bias,
        norm_layer=self.norm_layer, nonlinearity=nn.ReLU())

    self.conv_11 = nn.Sequential(
        nn.Conv2d(filters[0] + filters[4] + input_channels, filters[5],
                  kernel_size=7, stride=1, padding=3, bias=self.use_bias),
        nn.ReLU()
    )

    self.end_blocks = None
    if self.append_blocks:
      self.end_blocks = nn.Sequential(
          nn.Conv2d(filters[5], filters[5], kernel_size=3,
                    bias=self.use_bias, padding=1),
          nn.ReLU(),
          nn.BatchNorm2d(num_features=filters[5]),
          nn.Conv2d(filters[5], filters[5], kernel_size=3,
                    bias=self.use_bias, padding=1),
          nn.ReLU()
      )

    self.conv_12 = nn.Sequential(
        nn.Conv2d(
            filters[5],
            output_channels, kernel_size=1, stride=1, padding=0, bias=True))
    if tanh:
      self.conv_12.add_module('tanh', nn.Tanh())

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    output_0 = self.conv0(x)
    output_1 = self.conv1(output_0)
    output = self.conv2(output_1)
    output_2 = self.conv2(output_1)
    for layer in self.resnets:
      output = layer(output) + output

    output = self.upconv2(torch.cat((output, output_2), dim=1))
    output = self.upconv1(torch.cat((output, output_1), dim=1))
    output = self.conv_11(torch.cat((output, output_0, x), dim=1))
    if self.end_blocks is not None:
      output = self.end_blocks(output)
    output = self.conv_12(output)
    return output

  def relu_layer(
          self, in_filters: int, out_filters: int, size: int, stride: int,
          padding: int, bias: bool, norm_layer: Optional[ModuleProducer],
          nonlinearity: torch.nn.Module):
    out = []
    out.append(nn.Conv2d(in_channels=in_filters, out_channels=out_filters,
               kernel_size=size, stride=stride, padding=padding, bias=bias))
    if norm_layer:
      out.append(norm_layer(num_features=out_filters))
    if nonlinearity:
      out.append(nonlinearity)
    return nn.Sequential(*out)

  def resnet_block(
          self, in_filters: int, out_filters: int, size: int, stride: int,
          padding: int, bias: bool, norm_layer: Optional[ModuleProducer],
          nonlinearity: Optional[torch.nn.Module]):
    out = []
    if nonlinearity:
      out.append(nonlinearity)
    out.append(nn.Conv2d(in_channels=in_filters, out_channels=out_filters,
               kernel_size=size, stride=stride, padding=padding, bias=bias))
    if norm_layer:
      out.append(norm_layer(num_features=out_filters))
    if nonlinearity:
      out.append(nonlinearity)
    out.append(nn.Conv2d(in_channels=in_filters, out_channels=out_filters,
               kernel_size=size, stride=stride, padding=padding, bias=bias))
    return nn.Sequential(*out)

  def upconv_layer(
          self, in_filters: int, out_filters: int, size: int, stride: int,
          padding: int, bias: bool, norm_layer: Optional[ModuleProducer],
          nonlinearity: Optional[torch.nn.Module]):
    out = []
    out.append(SmoothUpsampleLayer(in_filters, out_filters))
    if norm_layer:
      out.append(norm_layer(num_features=out_filters))
    if nonlinearity:
      out.append(nonlinearity)
    return nn.Sequential(*out)


class ImageLoss(nn.Module):
  """Image per-pixel loss between two aligned images."""

  def __init__(self):
    super().__init__()
    # L2 is a bit better with multiple keyframes.
    self.objective = nn.MSELoss()

  def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return self.objective(x, y)


def rand_ind_2d(h: int, w: int, numind: int, device: torch.device,
                unique: bool = False):
  """Generates a field of random 2d indices."""
  if not unique:
    hc = torch.randint(low=0, high=h, size=(numind,), device=device)
    wc = torch.randint(low=0, high=w, size=(numind,), device=device)
  else:
    hc = (torch.randperm(h * w, device=device) // w)[:numind]
    wc = (torch.randperm(h * w, device=device) // h)[:numind]
  return hc, wc


class Vgg19_Extractor(nn.Module):
  """VGG-19 deep neural feature extractor."""

  def __init__(self, capture_layer_indices: List[int]):
    super().__init__()
    self.vgg_layers = models.vgg19(pretrained=True)
    self.vgg_layers = self.vgg_layers.features

    for param in self.parameters():
      param.requires_grad = False
    self.capture_layers = capture_layer_indices

  def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
    feat = []
    if -1 in self.capture_layers:
      feat.append(x)
    i = 0
    for mod in self.vgg_layers:
      x = mod(x)
      i += 1
      if i in self.capture_layers:
        feat.append(x)
    return feat


class InnerProductLoss(nn.Module):
  """Module for calculating inner product / gram matrix loss between two images."""
  # Rip out the caching module for now.

  def __init__(
          self, capture_layer_indices: List[int],
          device: torch.device, num_scales: int = 1,
          build_cache_from: Optional[Dict[str, torch.Tensor]] = None):
    super().__init__()
    self.layers = capture_layer_indices
    self.device = device
    self.vgg = Vgg19_Extractor(capture_layer_indices).to(device)
    self.stored_mean = torch.Tensor(
        [0.485, 0.456, 0.406]).to(device).view(1, -1, 1, 1)
    self.stored_std = torch.Tensor(
        [0.229, 0.224, 0.225]).to(device).view(1, -1, 1, 1)
    self.gmm = GramMatrix()
    # Loss between two product matrices.
    self.dist = nn.MSELoss()
    self.scales = [1.0 / (scale + 1) for scale in range(num_scales)]
    # Per-scale inner product cache.
    cache_type = Optional[Dict[str, Dict[float, List[torch.Tensor]]]]
    if build_cache_from is not None:
      self.cache: cache_type = self._build_cache(build_cache_from)
    else:
      self.cache = None

  def _resample_image(self, image: torch.Tensor, scale: float) -> torch.Tensor:
    result = image if scale == 1. else F.interpolate(image, scale_factor=float(
        scale), mode='bilinear', align_corners=False, recompute_scale_factor=False)
    return result

  def _build_cache(self, images: Dict[str, torch.Tensor]):
    cache = {}
    for key in images:
      if key not in cache:
        cache[key] = {}
      for scale in self.scales:
        image = self._resample_image(images[key], scale)
        gmm = [self.gmm(feat) for feat in self.extractor(image)]
        cache[key][scale] = gmm
    return cache

  def extractor(self, x: torch.Tensor) -> List[torch.Tensor]:
    # Remap x to VGG input space. Expects images in [-1, 1] sRGB range.
    x = (x + 1.) / 2.
    x = x - self.stored_mean
    x = x / self.stored_std
    return self.vgg(x)

  def calculate_at_scale(
          self, y1: torch.Tensor, y2: torch.Tensor, cache_key: Optional[str] = None,
          scale: float = 1.) -> torch.Tensor:
    y1 = self._resample_image(y1, scale)
    feat_y1 = self.extractor(y1)
    gmm_y1 = [self.gmm(feat) for feat in feat_y1]
    # if cache_key is not None:
    #   gmm_y2 = self.cache[cache_key][scale]
    # else:
    y2 = self._resample_image(y2, scale)
    feat_y2 = self.extractor(y2)
    gmm_y2 = [self.gmm(feat) for feat in feat_y2]

    num_levels = len(gmm_y1)
    loss = torch.empty((len(feat_y1),)).to(y1.device)
    loss = []
    for l in range(num_levels):
      gmm_y1_l = gmm_y1[l]
      gmm_y2_l = gmm_y2[l]
      assert gmm_y1_l.shape == gmm_y2_l.shape
      dist = self.dist(gmm_y2_l.detach(), gmm_y1_l)
      loss += [dist]
    loss = torch.stack(loss)
    return torch.sum(loss)

  def forward(self, y1: torch.Tensor, y2: torch.Tensor,
              cache_key: Optional[str] = None) -> torch.Tensor:
    scale_losses = []
    for scale in self.scales:
      loss = self.calculate_at_scale(y1, y2, cache_key, scale)
      scale_losses.append(loss)
    scale_losses = torch.stack(scale_losses)
    return torch.mean(scale_losses)


class InferDataset(Dataset):
  """Dataset which globs images in `dataroot/input/` directory."""

  def __init__(
          self, dataroot: str, xform: ImageTensorConverter, look_for_input: bool = True):
    if look_for_input:
      self.root = os.path.join(dataroot, 'input')
    else:
      self.root = dataroot
    self.frames = images_in_directory(self.root)
    self.tensors = []
    self.xform = xform
    for frame in self.frames:
      x = pil_loader(os.path.join(self.root, frame))
      self.tensors.append(self.xform(x))

  def __len__(self) -> int:
    return len(self.tensors)

  def __getitem__(self, idx) -> torch.Tensor:
    return self.tensors[idx]


def rng(min: float, max: float) -> float:
  """Generates a random number in [min, max) interval."""
  return random.random() * (max - min) + min


class NullAugmentations:
  def __init__(self):
    pass

  def __call__(self, *items):
    return items

def tps_warp(image: np.ndarray, ctrl_pts_y: int, ctrl_pts_x: int,
                    offset_scale: float, align_corners: bool = True, 
                    offsets_y: Optional[np.ndarray] = None, 
                    offsets_x: Optional[np.ndarray] = None) -> np.ndarray:
  assert ctrl_pts_y == ctrl_pts_x, 'Require square pts'
  h, w = image.shape[:2]
  ctrl_pts_loc_y = np.linspace(0, h - 1, ctrl_pts_y)
  ctrl_pts_loc_x = np.linspace(0, w - 1, ctrl_pts_x)
  ctrl_pts_loc_y, ctrl_pts_loc_x = np.meshgrid(ctrl_pts_loc_x, ctrl_pts_loc_y)
  ctrl_pts_loc_y = ctrl_pts_loc_y.reshape(-1)
  ctrl_pts_loc_x = ctrl_pts_loc_x.reshape(-1)

  offset_base_y = offsets_y if offsets_y is not None else np.random.rand(*ctrl_pts_loc_y.shape)
  offset_base_x = offsets_x if offsets_x is not None else np.random.rand(*ctrl_pts_loc_x.shape)
  random_offsets_y = (offset_base_y * 2 * h - h) * offset_scale
  random_offsets_x = (offset_base_x * 2 * w - w) * offset_scale

  warp_pts_loc_y = ctrl_pts_loc_y + random_offsets_y
  warp_pts_loc_x = ctrl_pts_loc_x + random_offsets_x

  if align_corners:
    ctrl_pts_loc_y = np.concatenate(
        [ctrl_pts_loc_y, np.array([0, h - 1, 0, h - 1])])
    warp_pts_loc_y = np.concatenate(
        [warp_pts_loc_y, np.array([0, h - 1, 0, h - 1])])

    ctrl_pts_loc_x = np.concatenate(
        [ctrl_pts_loc_x, np.array([0, 0, w - 1, w - 1])])
    warp_pts_loc_x = np.concatenate(
        [warp_pts_loc_x, np.array([0, 0, w - 1, w - 1])])

  # Build system matrix.
  n = ctrl_pts_loc_y.size
  d = np.zeros([n, n], dtype=np.float32)
  for row in range(n):
    d[row] = (warp_pts_loc_x - warp_pts_loc_x[row]
              ) ** 2 + (warp_pts_loc_y - warp_pts_loc_y[row]) ** 2
  d = np.where(d < 1e-8, 1., d)

  #     [   K  P]
  # S = [P.t() 0]
  tps_system = np.zeros([n + 3, n + 3], dtype=np.float32)
  # K = d^2 * log(d)
  K = d * np.log(d)

  P = np.column_stack(
      [np.ones_like(warp_pts_loc_x),
       warp_pts_loc_x, warp_pts_loc_y])
  tps_system[:n, n:] = P
  tps_system[n:, :n] = P.T
  tps_system[:n, :n] = K

  tps_solution = np.linalg.inv(tps_system)

  weights_y = tps_solution[:n, :n] @ np.expand_dims(ctrl_pts_loc_y, 1)
  weights_x = tps_solution[:n, :n] @ np.expand_dims(ctrl_pts_loc_x, 1)

  affine_y = tps_solution[n:, :n] @ np.expand_dims(ctrl_pts_loc_y, 1)
  affine_x = tps_solution[n:, :n] @ np.expand_dims(ctrl_pts_loc_x, 1)

  canonical_grid_x, canonical_grid_y = np.meshgrid(
      np.linspace(0, w - 1, w), np.linspace(0, h - 1, h))

  dy = warp_pts_loc_y[None, None, :] - canonical_grid_y[:, :, None]
  dx = warp_pts_loc_x[None, None, :] - canonical_grid_x[:, :, None]

  d = dy ** 2 + dx ** 2
  d = np.where(d < 1e-8, 1., d)
  u = d * np.log(d)

  warped_grid_y = affine_y[0] + (affine_y[1]
                                 * canonical_grid_x) + (affine_y[2] * canonical_grid_y)
  warped_grid_y = warped_grid_y + np.sum(
      weights_y[:, 0][None, None, :] * u, axis=-1)

  warped_grid_x = affine_x[0] + (affine_x[1]
                                 * canonical_grid_x) + (affine_x[2] * canonical_grid_y)
  warped_grid_x = warped_grid_x + np.sum(
      weights_x[:, 0][None, None, :] * u, axis=-1)

  warp_grid = np.stack([warped_grid_x, warped_grid_y], axis=-1)
  # Numpy doesn't provide a good out-of-the-box sampler.
  t_image = torch.tensor(
      image, dtype=torch.float32).permute(
      2, 0, 1).unsqueeze(0)
  t_grid = torch.tensor(warp_grid, dtype=torch.float32).unsqueeze(0)
  t_grid[:, :, :, 0] = t_grid[:, :, :, 0] / (h / 2) - 1
  t_grid[:, :, :, 1] = t_grid[:, :, :, 1] / (w / 2) - 1
  t_warped = torch.nn.functional.grid_sample(
      t_image, t_grid, padding_mode='reflection', align_corners=True)
  return t_warped[0].permute(1, 2, 0).numpy().astype(image.dtype)


class ShapeAugmentations:
  """Augmentations which possibly change feature pixel locations."""

  def __init__(
          self, angle_min: float, angle_max: float, hflip_chance: float,
          tps_chance: float, tps_scale: float):
    self.angle_min = angle_min
    self.angle_max = angle_max
    self.hflip_chance = hflip_chance
    self.tps_chance = tps_chance
    self.tps_scale = tps_scale
    self.tps_points = 5

  def __call__(self, *items: Image.Image) -> Image.Image:
    angle = rng(self.angle_min, self.angle_max)
    hflip = rng(0, 1) < self.hflip_chance
    tps = rng(0, 1) < self.tps_chance
    tps_y = np.array([rng(0, 1) for _ in range(self.tps_points ** 2)])
    tps_x = np.array([rng(0, 1) for _ in range(self.tps_points ** 2)])
    tps_scale = rng(0, 1) * self.tps_scale

    def transform(x: Image.Image) -> Image.Image:
      if hflip:
        x = Tfunc.hflip(x)
        
      x = np.array(x)
      x = rotate.rotate_image(x, angle, 'linear')
      if tps:
        x = tps_warp(x, self.tps_points, self.tps_points, tps_scale, offsets_y=tps_y, offsets_x=tps_x)
      x = Image.fromarray(x)
      return x

    augmented_items = []
    for item in items:
      augmented_items.append(transform(item))
    return augmented_items


class ColorAugmentations:
  """Augmentations which change color properties of pixels."""

  def __init__(self,
               adjust_contrast_min: float, adjust_contrast_max: float,
               adjust_hue_min: float, adjust_hue_max: float,
               adjust_saturation_min: float, adjust_saturation_max: float):
    self.adjust_contrast_min = adjust_contrast_min
    self.adjust_contrast_max = adjust_contrast_max
    self.adjust_hue_min = adjust_hue_min
    self.adjust_hue_max = adjust_hue_max
    self.adjust_saturation_min = adjust_saturation_min
    self.adjust_saturation_max = adjust_saturation_max

  def __call__(self, *items: Image.Image) -> Image.Image:
    cnts = rng(self.adjust_contrast_min, self.adjust_contrast_max)
    hue = rng(self.adjust_hue_min, self.adjust_hue_max)
    sat = rng(self.adjust_saturation_min, self.adjust_saturation_max)

    def transform(x: Image.Image) -> Image.Image:
      x = Tfunc.adjust_contrast(x, cnts)
      x = Tfunc.adjust_hue(x, hue)
      x = Tfunc.adjust_saturation(x, sat)
      return x

    augmented_items = []
    for item in items:
      augmented_items.append(transform(item))
    return augmented_items


class TrainingDataset(Dataset):
  """Dataset used for pulling in training samples for STALP.
  
    The expected structure is:
      root/
        input
        output
  """

  def __init__(
          self, dataroot: str, xform: ImageTensorConverter,
          augment_config: Dotdict):
    self.root = dataroot
    self.xform = xform

    keys_before = 'input'
    keys_after = 'output'

    image_fnames = images_in_directory(
        os.path.join(self.root, keys_before))
    keys_in = [os.path.join(keys_before, f) for f in image_fnames]
    keys_out = [os.path.join(keys_after, f) for f in image_fnames]
    self.keypair_files = list(zip(keys_in, keys_out))

    self.pairs = []
    for keyframe in self.keypair_files:
      x, y = pil_loader(os.path.join(self.root, keyframe[0])), pil_loader(
          os.path.join(self.root, keyframe[1]))
      self.pairs.append((x, y))
    if augment_config.disable_all:
      self.shape_augment = NullAugmentations()
      self.color_augment = NullAugmentations()
    else:
      self.shape_augment = ShapeAugmentations(**augment_config.shape_augments)
      self.color_augment = ColorAugmentations(**augment_config.color_augments)

  def __len__(self) -> int:
    return 2 ** 32 # len(self.pairs)

  def __getitem__(self, _idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Choose a random sample from the dataset - different on each call, along
    # with the original keyframe. Shape augmentations are applied jointly to
    # keep alignment of images.
    idx = np.random.randint(0, len(self.pairs))
    # print(f'Using img {idx}')

    shaped0, shaped1 = self.shape_augment(
        self.pairs[idx][0], self.pairs[idx][1])
    colored0, = self.color_augment(shaped0)

    return self.xform(colored0), self.xform(shaped1), self.xform(
        self.pairs[idx][1])


def log_verification_images(log: TensorboardLogger, step: int,
                            model: torch.nn.Module, dataset: Dataset,
                            transform: ImageTensorConverter,
                            additional_image: Optional[torch.Tensor] = None):
  with torch.no_grad():
    model.eval()
    example = enumerate(dataset).__next__()
    pred = model(example[1].to(guess_model_device(model)))
    if additional_image is not None:
      log.log_image('Keyframe', pil_to_np(
          transform(additional_image[0].data.cpu())), step, format='HWC')
    log.log_image('First_Unpaired_Frame', pil_to_np(
        transform(pred[0].data.cpu())), step, format='HWC')
    log.flush()


def log_verification_video(
        log: TensorboardLogger, step: int, label: str, model: torch.nn.Module,
        dataset: Dataset, transform: ImageTensorConverter, device: torch.device,
        shape: Sequence[int],
        max_frames: Optional[int] = None, fps: int = 1):
  """Produces infered video in loggers datadir."""
  with torch.no_grad():
    model.eval()
    if max_frames is None:
      # Limit max_frames to take up X MB at most. Assume FP32.
      mb_per_img = (np.prod(shape) * 4) / (1024 * 1024)
      max_frames = min(len(dataset), max(
          1, config.max_mem_for_log_video // mb_per_img))

    vid_tensor = torch.empty(
        (1, int(max_frames), int(shape[1]), int(shape[2]), int(shape[3])),
        device=device)
    for i, b in enumerate(dataset):
      if i >= max_frames:
        break
      if len(b.shape) == 3:
        b = b.unsqueeze(0)
      b = tensor_resample(b.to(device), [shape[2], shape[3]])
      output_frame = model(b)
      vid_tensor[:, i, :, :, :] = transform.denormalize_tensor(output_frame)
    cpu_video = (vid_tensor[0].cpu().permute(
        (0, 2, 3, 1)) * 255).to(torch.uint8)
    torchvision.io.write_video(
        log.location() + f"/{step}.mp4",
        cpu_video,
        fps=fps)


def log_profiler_trace(prof):
  print(prof.key_averages().table(
        sort_by='self_cuda_time_total', row_limit=-1))


def train_with_similarity(
        model: torch.nn.Module, iters: int, datasets: Dict[str, DataLoader],
        transform: ImageTensorConverter, device: torch.device,
        log: TensorboardLogger):
  # Setup gadgets.
  stopwatch = Stopwatch()

  def tformat(t: float):
    return f'{int(t * 60)}m' if t < 1 else f'{int(t)}h'
  snapshots = [(t * 3600, tformat(t)) for t in config.model_snapshot_hours]

  loss_config = config.loss_args
  if loss_config.image_weight > 0. or loss_config.use_annealed_image_loss:
    image_loss_fn = ImageLoss()
    if loss_config.use_annealed_image_loss:
      image_error_weight_annealing = ValueAnnealing(
          loss_config.annealed_image_loss_start,
          loss_config.annealed_image_loss_end, loss_config.annealed_image_loss_steps)

  def image_loss_weight():
    if loss_config.use_annealed_image_loss:
      return image_error_weight_annealing.next()
    return loss_config.image_weight

  if loss_config.similarity_weight > 0.:
    similarity_loss_fn = InnerProductLoss(loss_config.layers, device)
    if config.jit_model:
      similarity_loss_fn = torch.jit.script(similarity_loss_fn)

  model = model.to(device)
  optimizer = opt.Adam(model.parameters(), lr=config.lr)
  aux_sample = InfiniteDatasetSampler(datasets[_KEY_DATASET_AUXILIARY])
  ebest = float('inf')
  train_sample = InfiniteDatasetSampler(datasets[_KEY_DATASET_PAIRS])

  trange = tqdm(range(iters))
  for train_iter in trange:
    # Reset to train mode & init random with new seed (double check if needed).
    model.train()
    # np.random.seed(train_iter)

    with profiler.profile(
            activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
            profile_memory=True,
            record_shapes=True,
            schedule=profiler.schedule(
                wait=100,
                warmup=5,
                active=10),
            on_trace_ready=log_profiler_trace) as prof:
      for _ in range(1):
        _, (keyframe_x, keyframe_y, pure_y) = next(train_sample)
        if loss_config.similarity_weight > 0.:
          _, frame_x = aux_sample()
          frame_x = frame_x.to(device)
        keyframe_x, keyframe_y, pure_y = keyframe_x.to(device), keyframe_y.to(device), pure_y.to(device)

        optimizer.zero_grad()
        with profiler.record_function("L1_Calc"):
          y = model(keyframe_x)
        total_loss = 0.

        # L1 Loss Calculation.
        current_im_weight = image_loss_weight()
        if current_im_weight > 0.:
          image_loss = current_im_weight * image_loss_fn(y, keyframe_y)
          total_loss += image_loss

        # Similarity Loss Calculation.
        if loss_config.similarity_weight > 0.:
          with profiler.record_function("Y2_Calc"):
            y2 = model(frame_x)
          with profiler.record_function("Sim_Calc"):
            if loss_config.probed_multiframe_loss:
              best_loss = float('inf')
              # Enumerate all style frames.
              for style_frame in datasets[_KEY_DATASET_STYLEFRAMES]:
                style_frame = style_frame.to(device)
                local_sim_loss = similarity_loss_fn(y2, style_frame)
                if local_sim_loss < best_loss:
                  best_loss = local_sim_loss
              similarity_loss = loss_config.similarity_weight * best_loss
            else:
              similarity_loss = loss_config.similarity_weight * \
                  similarity_loss_fn(y2, pure_y)
          total_loss += similarity_loss

        # Track values for logging.
        tracked_scalars = ['image_loss', 'similarity_loss', 'total_loss']
        scalars = {name: value for name, value in locals().items()
                   if name in tracked_scalars}
        log.log_multiple_scalars(scalars, train_iter)

        trange.set_description(
            f'Similarity Error: {float(similarity_loss):.5f} | '+
            f'Image Error: {float(image_loss):.5f} | '+
            f'Error: {float(total_loss):.5f}')
        with profiler.record_function("Backward"):
          total_loss.backward()
        optimizer.step()
      # End of training loop.

      def log_video(dataset: Dataset, label: str):
        log_verification_video(
            log, train_iter, label, model, dataset,
            transform, device, y.shape, fps=config.video_fps)
      def log_checkpoint(label: str):
        log.log_checkpoint({'state_dict': model.state_dict(),
                           'opt_dict': optimizer.state_dict()}, label)

      # Take snapshots.
      for deadline, snap in snapshots:
        if stopwatch.just_passed(deadline):
          log_checkpoint(f'{snap}_snapshot')
          log.log_file(log._best_checkpoint_location(),
                       output_name=f'{snap}_snapshot_best.pth')

      # Check time limit.
      if config.time_limit_minutes is not None and stopwatch.just_passed(
              config.time_limit_minutes * 60):
        log_video(datasets[_KEY_DATASET_AUXILIARY].dataset, 'Auxiliary Frames')
        log_checkpoint('latest')
        print("Maximum time passed, exiting..")
        return

      # Log images and save best checkpoints.
      if train_iter % config.log_image_update_every == 0 and train_iter != 0:
        log_verification_images(log, train_iter, model,
                                datasets[_KEY_DATASET_AUXILIARY], transform, y)

        if total_loss < ebest:
          ebest = total_loss
          log.log_checkpoint_best(
              {'state_dict': model.state_dict(),
               'opt_dict': optimizer.state_dict()})

      # Log videos.
      if train_iter % config.log_video_update_every == 0 and train_iter != 0:
        if datasets[_KEY_DATASET_VALIDATE] is not None:
          log_video(datasets[_KEY_DATASET_VALIDATE].dataset, 'Validation Frames')
        log_video(datasets[_KEY_DATASET_AUXILIARY].dataset, 'Auxiliary Frames')
        log_checkpoint('latest')
        log.flush()
  # Save model just before exiting when finishing all iters.
  log_checkpoint('latest')


def check_dataset_consistency(dataset_train: TrainingDataset,
                              dataset_infer: InferDataset):
  size = None
  for pair in dataset_train.pairs:
    x, y = pair
    if size is None:
      size = x.size
    assert x.size == size, "One of the input images has different size"
    assert y.size == size, "One of the output images has different size"
  for im in dataset_infer.tensors:
    assert im.size == size, "One of the video frames has different size"


def make_datasets(transform: ImageTensorConverter) -> Dict[str, Dataset]:
  data_root = config.data_root
  data_root_pairs = os.path.join(data_root, 'pairs')
  data_root_train = os.path.join(data_root, 'train')

  # Additional validation data if valid exists.
  data_root_valid = None
  if os.path.exists(os.path.join(data_root, 'valid')):
    data_root_valid = os.path.join(data_root, 'valid')

  # Probe images to check sizes.
  data_aux_probe = InferDataset(data_root_train, lambda x: x)
  data_train_probe = TrainingDataset(
      data_root_pairs, lambda x: x, config.augment_config)
  check_dataset_consistency(data_train_probe, data_aux_probe)
  del data_aux_probe
  del data_train_probe

  data_aux = InferDataset(data_root_train, transform)
  data_style = InferDataset(os.path.join(data_root_pairs, 'output'),
                    transform, look_for_input=False)
  data_train = TrainingDataset(
      data_root_pairs, transform, config.augment_config)
  data_validate = InferDataset(
      data_root_valid, transform) if data_root_valid is not None else None

  return {
    _KEY_DATASET_PAIRS: data_train,
    _KEY_DATASET_AUXILIARY: data_aux,
    _KEY_DATASET_VALIDATE: data_validate,
    _KEY_DATASET_STYLEFRAMES: data_style,
  }


def make_loaders(datasets: Dict[str, Dataset]) -> Dict[str, DataLoader]:
  def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

  batch_size = config.batch_size

  def device_collate(x): 
    default_collated = torch.utils.data.dataloader.default_collate(x)
    if isinstance(default_collated, (list, tuple)):
      return [t.to(device) for t in default_collated]
    return default_collated.to(device)
  
  loaders = {}
  for key in datasets:
    loaders[key] = DataLoader(
      datasets[key], num_workers=2, worker_init_fn=worker_init_fn,
      batch_size=batch_size) if datasets[key] is not None else None
  return loaders


if __name__ == "__main__":

  torch.manual_seed(0)
  torch.backends.cudnn.benchmark = True
  # torch.backends.cudnn.deterministic = True
  np.random.seed(0)

  device = config.device
  transform = ImageTensorConverter(
      mean=[0.5, 0.5, 0.5],
      std=[0.5, 0.5, 0.5],
      resize=f'flex;8;max;{config.resize}'
      if config.resize is not None else f'flex;8', drop_alpha=True)
  model = ImageToImageGenerator_JohnsonFutschik(**config.model_config)
  if config.jit_model:
    model = torch.jit.script(model)

  datasets = make_datasets(transform)
  dataloaders = make_loaders(datasets)

  log = TensorboardLogger(config.logdir, suffix=config.log_suffix,
                          checkpoint_fmt='checkpoint_%s.pth')
  train_with_similarity(
      model, config.iter_limit, dataloaders, transform, device, log)
