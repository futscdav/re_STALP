import numpy as np
from build_rotate import checked_build

checked_build()
import _cc_rotate


def rotate_image(image: np.ndarray, degrees: float, interp='linear') -> np.ndarray:
  """Rotates `image` by `degrees`."""
  cc_interp_method = {'nearest': 0, 'linear': 1}[interp]

  clip_dim = False
  if image.ndim == 2:
    image = image[:, :, None]
    clip_dim = True
  rotated = _cc_rotate.rotate_image(image, degrees, cc_interp_method)
  if clip_dim:
    rotated = rotated[:, :]
  return rotated
  