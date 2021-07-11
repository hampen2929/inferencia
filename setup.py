import codecs
import os
from setuptools import setup, find_packages

PACKAGE = 'inferencia'
README = 'README.md'
# REQUIREMENTS = './docker/gpu/requirements.txt'

VERSION = '0.0.1'

DESCRIPTION = 'The implementation of onnx model for inference.'


def read(fname):
    # file must be read as utf-8 in py3 to avoid to be bytes
    return codecs.open(os.path.join(os.path.dirname(__file__), fname),
                       encoding='utf-8').read()


setup(name=PACKAGE,
      version=VERSION,
      # long_description=read(README),
      #   install_requires=list(read(REQUIREMENTS).splitlines()),
      url='https://github.com/hampen2929/inferencia',
      author='hampen2929',
      author_email='yuya.mochimaru.ym@gmail.com',
      packages=find_packages(),
      license='Apache'
      )
