from setuptools import setup, find_packages

setup(
    name='cpu_lmfit',
    version='0.1.0',
    packages=find_packages(),
    include_package_data=True,
    package_data={
        '': ['*.pyd', '*.dll', '*.so'],
    },
    description='Python bindings for CPU-LMFit (prebuilt extension included)',
    long_description='',
    zip_safe=False
)
