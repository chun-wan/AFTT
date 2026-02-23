from setuptools import setup, find_packages

setup(
    name="aftt",
    version="0.1.0",
    description="ASM Fine-Tuning Tool - Static analysis and optimization advisor for AMD GPU kernels",
    packages=find_packages(),
    py_modules=["cli"],
    install_requires=[
        "click>=8.1.0",
        "rich>=13.0.0",
        "pyyaml>=6.0",
        "sqlalchemy>=2.0",
        "jinja2>=3.1.0",
        "tabulate>=0.9.0",
    ],
    entry_points={
        "console_scripts": [
            "aftt=cli:main",
        ],
    },
    python_requires=">=3.10",
)
