import setuptools
import neural_pipeline


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="cv_utils",
    version=neural_pipeline.__version__,
    author="Anton Fedotov",
    author_email="anton.fedotov.af@gmail.com",
    description="Utils for computer vision tasks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/toodef/cv_utils",
    packages=setuptools.find_packages(exclude=['tests']),
    install_requires=['numpy', 'torch>=0.4.1', 'opencv-python'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
)
