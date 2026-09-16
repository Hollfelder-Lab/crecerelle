from setuptools import setup, find_packages

setup(
    name='crecerelle',
    version='0.1.0',
    author='Friedrich-Maximilian Weberling',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    package_data={'crecerelle.default_figures': ['*.png'],},
    include_package_data=True,
    license='MIT',
    classifiers=[
        'Programming Language :: Python :: 3.12',
        'License :: OSI Approved :: MIT License',
    ],
    install_requires=[
        'torch>=1.10.0',
        'anndata',
        'scanpy',
        'numpyro',
        # Versions used in the working Google Colab installation.
        'pandas==2.2.3',
        'scvi-tools==1.4.3',
        'scverse-misc[settings]==0.1.5',
        'pydantic-settings',
        'python-dotenv',
        'numpy',
        'matplotlib',
        'seaborn',
        'scikit-learn',
        'umap-learn',
        'leidenalg',
        'igraph',
        'scib',
        'upsetplot',
        'gprofiler-official',
        'matplotlib-venn',
    ],
    # Required by scvi-tools 1.4.3 and scverse-misc 0.1.5.
    python_requires='>=3.12',
    description='Crecerelle: deep generative cell embeddings embeddings from single-cell gene expression alternative splicing-induced transcript usage data',
)
