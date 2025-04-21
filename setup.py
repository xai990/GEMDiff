from setuptools import setup 

setup(
        name="GEMDiff",
        py_modules=["diffusion"],
        install_requires=["blobfile>=1.0.5", "torch=2.5.0", "einops", "omegaconf", "umap-learn[plot]"],
)
