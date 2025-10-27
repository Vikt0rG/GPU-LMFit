"""cpu_lmfit package shim.

This package re-exports symbols from the compiled extension module
`_cpu_lmfit` so users can simply `import cpu_lmfit` and access
`LMFit` and other functions/classes implemented in the C++ binding.
"""

# Import everything from the compiled extension submodule and re-export it, suppressing linting warnings.
try:
	from ._cpu_lmfit import *  # noqa: F401,F403
except Exception:
	raise ImportError("Failed to import the cpu_lmfit extension module. "
                      "Make sure the package is properly installed.")

__all__ = [name for name in globals().keys() if not name.startswith('_')]
