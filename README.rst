This is SUPER in development and SUBJECT TO CHANGE
--------------------------------------------------

Yes, there is no documentation at the moment, its not really intended as useable to the outside world atm, I personally need a convenient pip install.

=========
Konductor
=========

|version| |python| |license| |ci| |coverage| |codestyle|

.. |version| image:: https://img.shields.io/pypi/v/konductor
    :target: https://pypi.org/project/konductor/
    :alt: PyPI - Package Version
.. |python| image:: https://img.shields.io/pypi/pyversions/konductor
    :target: https://pypi.org/project/konductor/
    :alt: PyPI - Python Version
.. |license| image:: https://img.shields.io/pypi/l/konductor
    :target: https://github.com/konductor/konductor/blob/main/LICENSE
    :alt: PyPI - License
.. |ci| image:: https://img.shields.io/circleci/build/github/konductor/konductor/main
    :target: https://app.circleci.com/pipelines/github/konductor/konductor
    :alt: CircleCI
.. |coverage| image:: https://img.shields.io/codecov/c/gh/konductor/konductor
    :target: https://app.codecov.io/gh/konductor/konductor
    :alt: Codecov
.. |codestyle| image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black

Model training framework bundled with performance evaluation and comparison tools for rapid prototyping. Go deep into setting up automatic configuration for ablations or quickly throw in a dataloader, model and loss function and run distributed training in seconds.

The aims of this project are
 - Empower you to only need to touch a few parts of the framework to get some wacky non-standard training loop done.
 - Easily roll in existing code bases into this framework so you can git clone yolo-69, register it to konductor, throw in your data and start training instead of whatever random framework of their codebase.
