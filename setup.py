from setuptools import setup
from codecs import open
from os import path

dir_path = path.abspath(path.dirname(__file__))

with open(path.join(dir_path, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='gcndesign',
    packages=['gcndesign'],
    license='GPLv3',

    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    install_requires=['numpy', 'pandas', 'torch', 'tqdm'],

    author='Shintaro Minami',
    author_email='shintaro.minami@gmail.com',

    url='https://github.com/ShintaroMinami/GCNdesign',

    description='Neural network model for predicting amino-acid sequence from a protein backbone structure',
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords='gcndesign',

    scripts=[
        'scripts/gcndesign_training.py',
        'scripts/gcndesign_predict.py',
        'scripts/gcndesign_test.py',
        'scripts/gcndesign_pdb2csv.py',
    ],

    classifiers=[
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3.8',
    ],
)