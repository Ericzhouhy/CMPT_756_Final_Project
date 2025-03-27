from setuptools import setup, find_packages

setup(
    name='cifar100_trainer',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'torch==1.13.1',
        'torchvision==0.14.1',
        'google-cloud-storage'
    ]
)