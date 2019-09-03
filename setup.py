from setuptools import setup, find_packages

setup(name='mllib',
      version='0.0.0',
      url='https://github.com/ryu577/mllib',
      license='MIT',
      author='Rohit Pandey',
      author_email='rohitpandey576@gmail.com',
      description='Wrappers and toy examples for all kinds of ML algorithms.',
      packages=find_packages(exclude=['tests']),
      long_description=open('README.md').read(),
      zip_safe=False)

