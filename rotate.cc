#include <cstdint>
#include <cmath>
#include <exception>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

enum class Interp {
  NEAREST,
  BILINEAR
};

py::array_t<uint8_t> rotate(
  py::array_t<uint8_t, py::array::c_style | py::array::forcecast> image, 
  double angle_degrees, 
  int interpolation_mode) {

  // Asserts 3 dims.
  auto input_accessor = image.unchecked<3>();
  
  const py::ssize_t h = input_accessor.shape(0);
  const py::ssize_t w = input_accessor.shape(1);
  const py::ssize_t c = input_accessor.shape(2);

  py::array_t<uint8_t> output({h, w, c});
  auto output_accessor = output.mutable_unchecked<3>();

  if (interpolation_mode != 0 and interpolation_mode != 1) {
    throw std::invalid_argument("interpolation_mode must be 0 (nearest) or 1 (bilinear)");
  }
  const Interp interpolation = static_cast<Interp>(interpolation_mode);
  const double angle = M_PI * angle_degrees / 180;
  
  const double r11 = cos(angle);
  const double r12 = -sin(angle);
  const double r21 = sin(angle);
  const double r22 = r11;
  
  const double m = 1 / ((r11 * r22) - (r12 * r21));
  const double ri11 = r22 * m;
  const double ri12 = - r12 * m;
  const double ri21 = - r21 * m;
  const double ri22 = r11 * m;

  auto index_align = [=](auto& sx, auto& sy) {
    bool change = true;
    while (change) {
      change = false;
      if (sx < 0) { sx = -sx; change = true; }
      if (sy < 0) { sy = -sy; change = true; }
      if (sx >= w) { sx = w - ( sx - w + 1 ); change = true; }
      if (sy >= h) { sy = h - ( sy - h + 1 ); change = true; }
    }
  };
  
  #pragma omp parallel for
  for (py::ssize_t y = 0; y < h; y++) {
    #pragma omp parallel for
    for (py::ssize_t x = 0; x < w; x++) {
      const double rel_x = x - (static_cast<double>(w) / 2);
      const double rel_y = y - (static_cast<double>(h) / 2);
      
      const double sourcex = ri11 * rel_x + ri12 * rel_y;
      const double sourcey = ri21 * rel_x + ri22 * rel_y;
      
      double sx = sourcex + (static_cast<double>(w) / 2);
      double sy = sourcey + (static_cast<double>(h) / 2);
      
      if (interpolation == Interp::NEAREST) {
        index_align(sx, sy);
        for (int k = 0; k < c; k++) {
          output_accessor(y, x, k) = input_accessor(static_cast<int>(sy), 
                                                    static_cast<int>(sx), k);
        }
      } else {
        auto bilinear = [](
          uint8_t bl_val, uint8_t br_val,
          uint8_t tl_val, uint8_t tr_val,
          double relx, double rely
        ) -> uint8_t {
          const double cbl = relx * (1 - rely);
          const double cbr = (1 - relx) * (1 - rely);
          const double ctl = relx * rely;
          const double ctr = (1 - relx) * rely;
          const double value = cbl * bl_val + cbr * br_val + ctl * tl_val + ctr * tr_val;
          return static_cast<uint8_t>(value);
        };

        const double xfloor = floor(sx);
        const double xceil = xfloor + 1;
        const double yfloor = floor(sy);
        const double yceil = yfloor + 1;

        for (int k = 0; k < c; k++) {
          double ax = xfloor;
          double ay = yceil;
          index_align(ax, ay);
          uint8_t tr = input_accessor(static_cast<int>(ay), static_cast<int>(ax), k);
          ax = xceil;
          ay = yceil;
          index_align(ax, ay);
          uint8_t tl = input_accessor(static_cast<int>(ay), static_cast<int>(ax), k);
          ax = xfloor;
          ay = yfloor;
          index_align(ax, ay);
          uint8_t br = input_accessor(static_cast<int>(ay), static_cast<int>(ax), k);
          ax = xceil;
          ay = yfloor;
          index_align(ax, ay);
          uint8_t bl = input_accessor(static_cast<int>(ay), static_cast<int>(ax), k);

          uint8_t value = bilinear(bl, br, tl, tr, sx - xfloor, sy - yfloor);
          output_accessor(y, x, k) = value;
        }
      }
    }
  }

  return output;
}

PYBIND11_MODULE(_cc_rotate, m) {
  m.doc() = "Plugin providing image rotation with reflection padding in out-of-bound "
      "areas of the rotated image.";

  m.def("rotate_image", &rotate, "Performs image rotation of [angle] degrees (!) about "
                 "the image center point.");
}
