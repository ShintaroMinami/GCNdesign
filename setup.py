from setuptools import setup
from codecs import open
from os import path
from setuptools_scm import get_version

dir_path = path.abspath(path.dirname(__file__))

with open(path.join(dir_path, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='gcndesign',
    packages=['gcndesign'],
    license='GPLv3',
    url='https://github.com/ShintaroMinami/GCNdesign',
    description='Neural network model for predicting amino-acid sequence from a protein backbone structure',
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords=['gcndesign', 'autodesign'],

    author='Shintaro Minami',
    author_email='shintaro.minami@gmail.com',

    use_scm_version={'local_scheme': 'no-local-version'},

    setup_requires=['setuptools_scm'],
    install_requires=['numpy', 'pandas', 'torch', 'tqdm', 'dataclasses'],
 
    include_package_data=True,
    scripts=[
        'scripts/gcndesign_autodesign',
        'scripts/gcndesign_predict',
        'scripts/gcndesign_resfile',
        'scripts/gcndesign_training',
        'scripts/gcndesign_test',
        'scripts/gcndesign_pdb2csv'
    ],

    classifiers=[
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3.8',
    ],
)