#nsml: nvcr.io/nvidia/tensorflow:20.03-tf2-py3
from setuptools import setup

setup(
    name='airush-2021-2-2',
    version='1.0.0',
    install_requires=[
        'pandas',
        'pyarrow',
        'tensorflow==2.2.0',
        'numpy',
        'seaborn',
        'matplotlib',
        'imbalanced-learn'
    ],
    python_requires='>=3.6',
)
