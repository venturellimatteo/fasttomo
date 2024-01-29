from setuptools import setup, find_packages

setup(
    name='fasttomo',
    version='0.1',
    author='Matteo Venturelli',
    author_email='matteo.venturelli2000@gmail.com',
    description='Master\'s thesis project consisting in the development of a pipeline to segment and render tomography data of lithium-ion batteries during abuse testing.',
    url='https://github.com/venturellimatteo/fasttomo',
    packages=find_packages(),
    install_requires=[
        'matplotlib==3.7.3',
        'napari==0.4.18',
        'numpy==1.24.4',
        'numpy-stl==3.1.1',
        'opencv-python==4.8.1.78',
        'pandas==2.0.3',
        'scikit-image==0.21.0',
        'scikit-learn==1.3.1',
        'seaborn==0.13.0',
        'tqdm==4.66.1'
    ],
    python_requires='>=3.8.17',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)

