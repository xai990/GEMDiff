from setuptools import setup 

setup(
        name="seq2seq-diffusion",
        py_modules=["diffusion"],
        install_requires=["blobfile>=1.0.5", "torch"],
)
