from setuptools import setup 

setup(
        name="DDPM-mRNA-augmentation-light",
        py_modules=["diffusion"],
        install_requires=["blobfile>=1.0.5", "torch"],
)
