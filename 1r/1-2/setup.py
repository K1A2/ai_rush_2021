#nsml: nvcr.io/nvidia/tensorflow:20.01-tf2-py3
from distutils.core import setup

setup(
    name='airush-2021-pubtrans',
    version='1.0',
    install_requires=[
        'pandas',
        'pyarrow',
        'tensorflow >= 2.2.0',
        'numpy',
        'scipy'
    ],
    python_requires='>=3.6',
)