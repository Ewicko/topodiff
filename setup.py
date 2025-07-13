from setuptools import setup

setup(
    name="topodiff",
    py_modules=["topodiff"],
    install_requires=["blobfile>=1.0.5", "torch", "tqdm", "matplotlib", "scikit-learn", "solidspy", "opencv-python"],
)
