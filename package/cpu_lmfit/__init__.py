"""cpu_lmfit package shim.

This package re-exports symbols from the compiled extension module
`_cpu_lmfit` so users can simply `import cpu_lmfit` and access
`LMFit` and other functions/classes implemented in the C++ binding.
"""

# Import everything from the compiled extension submodule and re-export it, suppressing linting warnings.
try:
	from ._cpu_lmfit import *  # noqa: F401,F403
except Exception:
	# If the compiled extension is not present, keep the package importable
	# so we can show a clearer error at runtime when functions are used.
	pass

__all__ = [name for name in globals().keys() if not name.startswith('_')]
