from setuptools import setup, find_packages

with open('requirements.txt') as f:
    install_requires = f.read().splitlines()

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='algo-collusion-mm',
    version='0.1.0',
    author='Matteo Onger',
    author_email='matteo.onger@studenti.unimi.it',
    description='Algorithmic collusion in market making',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/MatteoOnger/algo-collusion-mm',
    packages=find_packages(),
    install_requires=install_requires,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    license='MIT', 
    python_requires='>=3.12',
)
