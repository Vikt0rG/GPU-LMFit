from setuptools import setup

setup(
    name='cpu_lmfit',
    version='0.1.0',
    packages=['cpu_lmfit'],
    # When installing from within this directory, treat the current folder
    # as the package source. This makes `python -m pip install .` work from
    # package/cpu_lmfit.
    package_dir={'cpu_lmfit': '.'},
    package_data={
        'cpu_lmfit': ['_cpu_lmfit*.*'],
    },
    include_package_data=True,
    description='Python bindings for CPU-LMFit (prebuilt extension included)',
    long_description='',
    zip_safe=False,
)
