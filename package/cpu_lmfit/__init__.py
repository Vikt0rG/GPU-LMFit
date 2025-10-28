"""cpu_lmfit package shim.

This package re-exports symbols from the compiled extension module
`_cpu_lmfit` so users can simply `import cpu_lmfit` and access
`LMFit` and other functions/classes implemented in the C++ binding.
"""

# Import everything from the compiled extension submodule and re-export it, suppressing linting warnings.
try:
    from ._cpu_lmfit import *  # noqa: F401,F403
except Exception as e:
    import sys, os, traceback
    # Find the compiled extension in the package directory (if present)
    pkg_dir = os.path.dirname(__file__)
    so_files = [f for f in os.listdir(pkg_dir) if f.startswith('_cpu_lmfit') and f.endswith('.so')]
    built_tag = so_files[0] if so_files else '<not found>'
    msg = (
        f"Failed to import the cpu_lmfit extension module (built: {built_tag}).\n"
        f"Running Python: {sys.executable} {sys.version.splitlines()[0]}\n"
        "If you built the extension locally, rebuild/install using the same Python interpreter, e.g.:\n"
        "  python -m pip install .\n"
        "Or install a prebuilt wheel matching your Python minor version.\n"
        "Original error follows:"
    )
    raise ImportError(msg) from e

__all__ = [name for name in globals().keys() if not name.startswith('_')]
