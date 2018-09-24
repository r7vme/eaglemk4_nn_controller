from setuptools import setup

setup(name='eaglemk4nn',
      version='0.1',
      description='Neural networks based controller for Eagle MK4 robot',
      url='http://github.com/r7vme/eaglemk4_nn_controller',
      author='Roma Sokolkov',
      author_email='rsokolkov@gmail.com',
      license='MIT',
      packages=['eaglemk4nn'],
      zip_safe=False,
      install_requires=[
          'numpy',
          'tensorflow==1.10.1'
          ])
