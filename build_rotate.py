import platform
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension


_PLATFORM_ARGS = []
if platform.system() == 'Windows':
  _PLATFORM_ARGS += ['/permissive-', '/Ox']
else:
  _PLATFORM_ARGS += ['-fopenmp']


_EXT_MODULES = [
  Pybind11Extension(
    '_cc_rotate',
    sources=['rotate.cc'],
    extra_compile_args=_PLATFORM_ARGS,
    cxx_std=14,
  ),
]


def checked_build(force: bool = False):
  """Builds extension only if necessary."""
  def do_build():
    setup(ext_modules=_EXT_MODULES, script_args=['build_ext', '--inplace'])
  if force:
    do_build()
    return

  try:
    import _cc_rotate  # pylint: disable=unused-import
  except ImportError:
    do_build()


if __name__ == '__main__':
  setup(ext_modules=_EXT_MODULES)
