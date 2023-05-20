## setup file to make this as a package
from setuptools import setup, find_packages


## This is dependent on detectron2, fiftyone, and pytorch
## detectron2 is not available on pip, so we need to install it from source

setup(
    name='DWODLib',
    version='0.1.0',    
    description='Object detection on Dhiwise dataset',
    author='Dhaivat Bhatt',
    author_email='dhaivat1994@gmail.com',
    packages=['DWODLib'],
    python_requires='>=3.6',)
