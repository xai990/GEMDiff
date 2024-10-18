from setuptools import setup 

setup(
        name="GEMDiff",
        py_modules=["diffusion"],
        install_requires=["blobfile>=1.0.5", "torch", "einops", "omegaconf", "umap-learn[plot]"],
)
