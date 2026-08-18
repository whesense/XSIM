from setuptools import setup, find_packages

setup(
    name="xsim",
    version="0.1.0",
    description="Dynamic driving-scene reconstruction with Gaussian splatting",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
    python_requires=">=3.11"
)
