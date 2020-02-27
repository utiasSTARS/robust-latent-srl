from setuptools import setup, find_packages

setup(name='robust-latent-srl',
      packages=[package for package in find_packages()
                if package in ('args',
                               'data_collection',
                               'experiments',
                               'learning_utils'
                               'real_sense_server',
                               'rl',
                               'senseact.senseact',
                               'srl')],
      install_requires=[],
      version='0.0.1',
)
