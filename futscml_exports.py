import os
import shutil
import torch
import PIL.Image
import numpy as np
import torch.utils.tensorboard
import torch.nn.functional

from typing import Any, List, Sequence
from torchvision import transforms
from datetime import datetime
from time import perf_counter, time


class Dotdict(dict):
  """Lua-like dict."""    
  def __getattr__(self, name: str) -> Any:
    try:
      return self[name]
    except KeyError:
      raise AttributeError(name)

  def __setattr__(self, name: str, value: Any) -> None:
    self[name] = value

  def __delattr__(self, name: str) -> None:
    del self[name]

  def has_set(self, attr: str) -> bool:
    return attr in self and self[attr] is not None


def pil_to_np(pil: PIL.Image.Image) -> np.ndarray:
  return np.array(pil)


def np_to_pil(npy: np.ndarray):
  return PIL.Image.fromarray(npy.astype(np.uint8))


def guess_model_device(model: torch.nn.Module) -> torch.device:
  return next(model.parameters()).device


def is_image(fname: str) -> bool:
  fname = fname.lower()
  exts = ('jpg', 'png', 'bmp', 'jpeg', 'tiff')
  ok = fname.endswith(exts)
  return ok


def images_in_directory(dir: str) -> List[str]:
  ls = os.listdir(dir)
  return sorted(list(filter(is_image, ls)))


def pil_loader(path: str) -> PIL.Image.Image:
  with open(path, 'rb') as f:
    img = PIL.Image.open(f)
    return img.convert('RGB')


def tensor_resample(tensor: torch.Tensor, dst_size, mode: str = 'bilinear'):
  return torch.nn.functional.interpolate(
      tensor, dst_size, mode=mode, align_corners=None
      if mode == 'nearest' else False)


class Stopwatch:
  def __init__(self, resolution: str = 'low', start_at_creation: bool = True):
    self.time = time if resolution == 'low' else perf_counter
    self.started = 0.
    self.running = False
    self.last_request = 0.
    if start_at_creation:
      self.start()

  def start(self):
    if self.running:
      return
    self.running = True
    self.started = self.time()

  def elapsed(self) -> float:
    if not self.running:
      return 0.
    return self.time() - self.started

  def reset(self):
    self.running = False
    self.started = 0.

  def just_passed(self, target_time: float) -> bool:
    elapsed = self.elapsed()
    if elapsed >= target_time and self.last_request < target_time:
      self.last_request = target_time
      return True
    return False


class ValueAnnealing:
  def __init__(self, initial_value: float, final_value: float, over_steps: int):
    self.initial_value = initial_value
    self.step = 0
    self.max_step = over_steps
    self.final_value = final_value
    self.piece = (self.initial_value - self.final_value) / self.max_step

  def next(self) -> float:
    value = self.initial_value - (self.piece * self.step)
    if self.step < self.max_step:
      self.step += 1
    return value


class InfiniteDatasetSampler:
  def __init__(self, dataloader):
    self.dataloader = dataloader
    self.enumerator = None
    self.generator = self._generator()

  def reset_enumerator(self):
    self.enumerator = enumerate(self.dataloader)

  def _generator(self):
    if self.enumerator is None:
      self.reset_enumerator()
    while True:
      try:
        yield next(self.enumerator)
      except StopIteration:
        self.reset_enumerator()
        yield next(self.enumerator)

  def __next__(self):
    return next(self.generator)

  def __call__(self):
    return next(self)


class SmoothUpsampleLayer(torch.nn.Module):
  def __init__(
          self, in_filters: int, out_filters: int, scale_factor: int = 2,
          scaling_mode: str = 'nearest', bias: bool = False):
    super().__init__()
    self.upsample = torch.nn.Upsample(
        scale_factor=scale_factor, mode=scaling_mode)
    self.conv = torch.nn.Conv2d(
        in_channels=in_filters, out_channels=out_filters, bias=bias, padding=1,
        kernel_size=3)

  def forward(self, x):
    return self.conv(self.upsample(x))


class GramMatrix(torch.nn.Module):
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    n, c, h, w = x.shape
    M = x.view(n, c, h*w)
    G = torch.bmm(M, M.transpose(1, 2))
    G.div_(h*w*c)
    return G


class TensorboardLogger:
  def __init__(self, output_path, suffix=None,
               checkpoint_fmt='checkpoint_%03d.pth'):
    self.output_path = output_path
    self.suffix = '' if suffix is None else ('_' + suffix)

    self.created_at = datetime.now()
    self.dir_created = False
    self.summary_writer = None
    self.checkpoint_fmt = checkpoint_fmt

  def __del__(self):
    self.flush()

  def location(self):
    return os.path.join(self.output_path, self.created_at.strftime(
        '%Y_%m_%d-%H_%M_%S') + self.suffix)

  def _init_summary(self):
    self._init_dir()
    if self.summary_writer is not None:
      return
    print(f'Writing logs to {self.location()}.')
    self.summary_writer = torch.utils.tensorboard.SummaryWriter(
        self.location())

  def _init_dir(self):
    if self.dir_created:
      return
    if not os.path.exists(self.location()):
      os.makedirs(self.location())
    self.dir_created = True

  def log_scalar(self, tag, value, step):
    self._init_summary()
    self.summary_writer.add_scalar(
        tag=tag, scalar_value=value, global_step=step)

  def log_multiple_scalars(self, scalar_dict, step):
    self._init_summary()
    for k, v in scalar_dict.items():
      if v is None:
        continue
      self.log_scalar(k, v, step)

  def log_scalars_single_plot(self, tag, subtag_value_dict, step):
    self._init_summary()
    self.summary_writer.add_scalars(
        main_tag=tag, tag_scalar_dict=subtag_value_dict, global_step=step)

  def log_histogram(self, tag, values, step, bins='auto'):
    self._init_summary()
    self.summary_writer.add_histogram(
        tag=tag, values=values, global_step=step, bins=bins)

  def log_image(self, tag, image, step, format='CHW'):
    self._init_summary()
    self.summary_writer.add_image(
        tag=tag, img_tensor=image, global_step=step, dataformats=format)

  def log_multiple_images(self, tag, images, step, format='NCHW'):
    self._init_summary()
    self.summary_writer.add_images(
        tag=tag, img_tensor=images, global_step=step, dataformats=format)

  def log_figure(self, tag, figure, step):
    self._init_summary()
    self.summary_writer.add_figure(
        tag=tag, figure=figure, global_step=step, close=True)

  def log_text(self, tag, text, step):
    self._init_summary()
    self.summary_writer.add_text(tag=tag, text_string=text, global_step=step)

  # Expects uint8 or [0, 1].
  def log_video(self, tag, video, step, fps=4):
    self._init_summary()
    self.summary_writer.add_video(
        tag=tag, vid_tensor=video, global_step=step, fps=fps)

  def log_audio(self, tag, audio, step, samplerate=44100):
    self._init_summary()
    self.summary_writer.add_audio(
        tag=tag, snd_tensor=audio, global_step=step, sample_rate=samplerate)

  def log_checkpoint(self, state, epoch_tag):
    self._init_dir()
    where = os.path.join(self.location(), self.checkpoint_fmt % epoch_tag)
    torch.save(state, where)

  def _best_checkpoint_location(self):
    where = os.path.join(self.location(), "checkpoint_best.pth")
    return where

  def log_checkpoint_best(self, state):
    self._init_dir()
    where = self._best_checkpoint_location()
    torch.save(state, where)

  def log_mkdir(self, dirname):
    self._init_dir()
    os.makedirs(os.path.join(self.location(), dirname))

  def log_file(self, path: str, output_name=None, output_path=None):
    self._init_dir()
    where = os.path.join(
        self.location(),
        '' if output_path is None else output_path, os.path.basename(path)
        if output_name is None else output_name)
    shutil.copy2(path, where)

  def flush(self):
    if self.summary_writer:
      self.summary_writer.flush()


def dict_safe_get(dict, attr):
  return dict[attr] if (attr in dict and dict[attr] is not None) else None


class ResizeArgs:
  def __init__(self, **kwargs):
    self.align_to = int(
        kwargs['align_to']) if dict_safe_get(
        kwargs, 'align_to') else None
    self.max_long_edge = int(
        kwargs['max_long_edge']) if dict_safe_get(
        kwargs, 'max_long_edge') else None
    self.max_short_edge = int(
        kwargs['max_short_edge']) if dict_safe_get(
        kwargs, 'max_short_edge') else None

  @staticmethod
  def parse_from_string(config, sep=';'):
    # flex;8;max;512 = Keeps aspect ratio give or take and assures that each side is divisible by 8 and has max len of 512
    align_to = None
    max_long_edge = None
    max_short_edge = None
    args = config.split(sep)
    for i in range(len(args) // 2):
      if args[2*i] == 'flex':
        align_to = args[2*i+1]
      if args[2*i] == 'max':
        max_long_edge = args[2*i+1]
      if args[2*i] == 'max_short':
        max_short_edge = args[2*i+1]
    return ResizeArgs(
        align_to=align_to, max_long_edge=max_long_edge,
        max_short_edge=max_short_edge)


class FlexResize():
  def __init__(self, args):
    self.args = args
    self.align_to = args.align_to
    self.max_long = args.max_long_edge
    self.max_short = args.max_short_edge

  def keep_ar_sizes(self, x, max_long, max_short):
    new_h, new_w = x.height, x.width
    # max long takes precedence
    short_w = x.width < x.height
    if max_long:
      ar_resized_long = (max_long / new_h) if short_w else (max_long / new_w)
      new_h, new_w = round(
          new_h * ar_resized_long), round(new_w * ar_resized_long)
    if max_short:  # and min(new_h, new_w) > max_short: # Does not upsample..?
      ar_resized_short = (
          max_short / new_w) if short_w else (max_short / new_h)
      new_h, new_w = round(
          new_h * ar_resized_short), round(new_w * ar_resized_short)
    return new_h, new_w

  def __call__(self, x):
    h, w = self.keep_ar_sizes(x, self.max_long, self.max_short)
    h = h - (h % self.align_to)
    w = w - (w % self.align_to)
    return transforms.functional.resize(x, [h, w])


def apply_mask_to_np_image(
        im: np.ndarray, mask: np.ndarray, mask_range=None, invert: bool = False) -> np.ndarray:
  imf = im.astype(np.float)
  maskf = mask.astype(np.float)
  if mask_range is None:
    maskf += mask.min()
    maskf /= mask.max()
  else:
    maskf += mask_range[0]
    maskf /= mask_range[0] + mask_range[1]
  if invert:
    maskf = 1. - maskf
  # If mask is 1 channel but im is not.
  if mask.ndim == im.ndim - 1 or mask.shape[-1] == 1:
    maskf = np.stack([maskf] * im.shape[-1], axis=2)
  masked = imf * maskf
  return masked.astype(im.dtype)


class ImageTensorConverter:
  """Converts between PIL.Image and torch.Tensor.
  
    mean = Mean value per channel (In RBG even if is_rgb is true)
    std  = Std value per channel (In RBG even if is_rgb is true)
    resize = Same as torchvision.transforms.Resize OR string in the form of 'flex;<alignment>;max;<max_long_edge>' OR ResizeArgs instance
    is_bgr = swap channels when going from image to tensor
    mul_by = denormalize by this value after being cast to [0,1]
    unsqueeze = unsqueeze the resulting tensor in 0th dim and squeeze the tensor when going to PIL
    device = destination device for the tensor
    clamp_to_pil = Tuple-like [min, max] to which the result will be clamped before casting to PIL
    drop_alpha = convert the PIL to 'RGB' before passing it to ToTensor
    force_rgb = convert PIL to 'RGB' regardless.
  """

  def __init__(
          self, mean: Sequence[float] = [0.5, 0.5, 0.5],
          std: Sequence[float] = [0.5, 0.5, 0.5],
          resize=None, is_bgr=False, mul_by=None, unsqueeze=False, device=None,
          clamp_to_pil=None, drop_alpha=False, force_rgb=True):

    self.mean = mean
    self.std = std
    self.resize = resize
    self.to_tensor_transform = []
    self.inverse_transform = []

    # To Tensor Transform

    if force_rgb:
      self.to_tensor_transform.append(
          transforms.Lambda(lambda x: x.convert('RGB')))
    # Remove A channel & convert to RGB
    if drop_alpha:
      def mask_alpha(pil):
        if pil.mode == 'RGBA':
          as_np = np.array(pil)
          return np_to_pil(
              apply_mask_to_np_image(
                  as_np[..., 0: 3],
                  as_np[..., 3:]))
        else:
          return pil.convert('RGB')
      self.to_tensor_transform.append(transforms.Lambda(mask_alpha))
    # Resize
    if resize is not None:
      if isinstance(resize, ResizeArgs):
        self.to_tensor_transform.append(FlexResize(resize))
      elif isinstance(resize, str):
        if resize.startswith('flex'):
          self.to_tensor_transform.append(
              FlexResize(ResizeArgs.parse_from_string(resize)))
        else:
          print("Resize arguments unknown.")
      elif isinstance(resize, tuple):
        # Height, Width
        # If either argument is None, scale the non-None side to the specified size and keep AR for the other one
        if resize[0] is None:
          new_w = resize[1]
          self.to_tensor_transform.append(
              transforms.Lambda(
                  lambda img: img.resize(
                      (new_w, int((new_w / img.width) * img.height)))))
        elif resize[1] is None:
          new_h = resize[0]
          self.to_tensor_transform.append(
              transforms.Lambda(
                  lambda img: img.resize(
                      (int((new_h / img.height) * img.width),
                       new_h))))
        else:
          self.to_tensor_transform.append(transforms.Resize(resize))
      else:
        # Defer to transforms.Resize
        self.to_tensor_transform.append(transforms.Resize(resize))
    # ToTensor
    self.to_tensor_transform.append(transforms.ToTensor())
    # Denormalize if needed
    if mul_by:
      self.to_tensor_transform.append(
          transforms.Lambda(lambda x: x.mul_(mul_by)))
    # Normalization
    self.to_tensor_transform.append(transforms.Normalize(self.mean, self.std))
    # Channel swap Stuff
    if is_bgr:
      self.to_tensor_transform.append(transforms.Lambda(
          lambda x: x[torch.LongTensor([2, 1, 0])]))
    if unsqueeze:
      self.to_tensor_transform.append(
          transforms.Lambda(lambda x: x.unsqueeze(0)))
    if device:
      self.to_tensor_transform.append(
          transforms.Lambda(lambda x: x.to(device)))

    # To PIL transform

    if unsqueeze:
      self.inverse_transform.append(transforms.Lambda(lambda x: x.squeeze()))
    if is_bgr:
      self.inverse_transform.append(transforms.Lambda(
          lambda x: x[torch.LongTensor([2, 1, 0])]))

    self.inverse_transform.append(transforms.Lambda(
        lambda x: x * torch.tensor(self.std).view(-1, 1, 1).to(x.device)))
    self.inverse_transform.append(transforms.Lambda(
        lambda x: x + torch.tensor(self.mean).view(-1, 1, 1).to(x.device)))
    if mul_by:
      self.inverse_transform.append(
          transforms.Lambda(lambda x: x.div_(mul_by)))
    if clamp_to_pil is not None:
      self.inverse_transform.append(
          transforms.Lambda(
              lambda x: x.clamp(
                  min=clamp_to_pil[0],
                  max=clamp_to_pil[1])))
    # Cast it to cpu anyway just before PIL
    self.inverse_transform.append(transforms.Lambda(lambda x: x.cpu()))
    self.inverse_transform.append(transforms.ToPILImage())

    self.to_tensor_transform = transforms.Compose(self.to_tensor_transform)
    self.to_pil_transform = transforms.Compose(self.inverse_transform)

  def __call__(self, input):
    if type(input).__name__ == 'Tensor':
      return self.get_pil(input)
    else:
      return self.get_tensor(input)

  def get_tensor(self, input):
    return self.to_tensor_transform(input)

  def get_pil(self, input):
    # Fix some latent bugs with a clone.
    return self.to_pil_transform(input.clone())

  def denormalize_tensor(self, x):
    x = x * torch.tensor(self.std).view(-1, 1, 1).to(x.device)
    x = x + torch.tensor(self.mean).view(-1, 1, 1).to(x.device)
    return x
