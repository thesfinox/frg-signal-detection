Abstract
========

.. image:: _static/abstract.png
   :width: 100%
   :align: center
   :alt: graphical abstract

Signal detection is one of the main challenges of data science.
According to the nature of the data, the presence of noise may corrupt measurements and hinder the discovery of significant patterns.
A wide range of techniques aiming at extracting the relevant degrees of freedom from data has been thus developed over the years.
However, signal detection in almost continuous spectra, when the signal-to-noise ratio is small, remains a known difficult issue.
This paper develops over recent advancements proposing to tackle this issue by analysing the properties of the underlying effective field theory arising as a kind of maximal entropy distribution in the vicinity of universal random matrix distributions.
Nearly continuous spectra provide an intrinsic and non-conventional scaling law for field and couplings, the scaling dimensions depending on the energy scale.
The related coarse-graining over small eigenvalues of the empirical spectrum defines a specific renormalization group, whose characteristics change when the collective behaviour of “informational” modes become significant, that is, stronger than the intrinsic fluctuations of noise.
This paper pursues three different goals.
First, we propose to quantify the real effects of fluctuations relative to what can be called “signal”, while improving the robustness of the results obtained in our previous work.
Second, we show that quantitative changes in the presence of a signal result in a counterintuitive modification of the distribution of eigenvectors.
Finally, we propose a method for estimating the number of noise components and define a limit of detection in a general nearly continuous spectrum using the renormalization group.
The main statements of this paper are essentially numeric, and their reproducibility can be checked using the associated code.

Installation
============

You can easily replicate the working environment under ``python">=3.12"`` (e.g. ``conda create -n frg python"<3.13"``) using:

.. code:: python

   pip install -f requirements.txt

Usage
=====

The package relies on the definition of configuration files based on the `yacs <https://github.com/rbgirshick/yacs>`_ system, following this template:

.. code:: yaml

   DATA:
      OUTPUT_DIR: results
   DIST:
      NUM_SAMPLES: 1000
      RATIO: 0.5
      SEED: 42
      SIGMA: 1.0
   PLOTS:
      OUTPUT_DIR: plots
   POT:
      UV_SCALE: 1.0e-05
      KAPPA_INIT: 1.0e-09
      U2_INIT: 1.0e-05
      U4_INIT: 1.0e-05
      U6_INIT: 0.0
      UV_SCALE: 0.7
   SIG:
      INPUT: "/path/to/image-or-covariance-matrix"
      SNR: 0.0


Allowed entries are:

- ``DATA.OUTPUT_DIR``: directory where the results will be stored,
- ``DIST.NUM_SAMPLES``: size of the data sample to use,
- ``DIST.RATIO``: ratio between the number of variables (degrees of freedom, or columns of the data matrix) and the sample size (rows of the data matrix),
- ``DIST.SEED``: random seed to use,
- ``DIST.SIGMA``: standard deviation of the distribution,
- ``PLOTS.OUTPUT_DIR``: directory where the plots will be stored,
- ``POT.UV_SCALE``: high energy scale at which to start the computations,
- ``POT.KAPPA_INIT``: initial value for the location of the zero of the potential,
- ``POT.U2_INIT``: initial value for the mass (quadratic) coupling,
- ``POT.U4_INIT``: initial value for the quartic coupling,
- ``POT.U6_INIT``: initial value for the sextic coupling,
- ``POT.UV_SCALE``: UV high energy scale,
- ``SIG.INPUT``: path to the input signal or covariance matrix,
- ``SIG.SNR``: signal-to-noise ratio (the signal will be scaled by this factor).

Generation of Multiple Configuration Files
------------------------------------------

Starting from a base configuration file, multiple derived configurations can be automatically generated using the ``generate_config.py`` script:

.. code:: bash

   ./scripts/generate_config.py \
      --config /path/to/base_config.yaml \
      --params /path/to/parameters.json \
      --n_samples <number_of_files_to_generate> \
      --output_dir /path/to/output_directory \
      --seed <random_seed>

New points are generated using random sampling of the parameter space, using a *Latin Hypercube Sampling* (LHS) algorithm.

The JSON file containing the parameters to sample must be formatted using the configuration keys as keys (case-insensitive) of the dictionary.
Values can then be input as lists containing the minimum value and maximum value.
For instance:

.. code:: json

   {
      "pot": {
         "u2_init": [-1e-05, 1e-05],
         "u4_init": [-1e-05, 1e-05],
         "u6_init": [-1e-05, 1e-05]
      }
   }

will act on the parameters ``POT.U2_INIT``, ``POT.U4_INIT`` and ``POT.U6_INIT`` in the configuration files.

.. note::

   You can use the option ``--plots`` to visualise the sampled points in the parameter space.

Computation of the Canonical Dimensions
---------------------------------------

The file ``canonical_dimensions.py`` can be used to compute the canonical dimensions of the distribution of singular values:

.. code:: bash

   ./scripts/canonical_dimensions.py \
      --config /path/to/config.yaml

.. note::

   The ``--analytic`` argument can be used to run an analytic simulation instead of a numerical one.

Computation of the Functional Renormalization Group Equations
-------------------------------------------------------------

The file ``frg_equations.py`` can be used to compute the functional renormalization group equations:

.. code:: bash

   ./scripts/frg_equations.py \
      --config /path/to/config.yaml

.. note::

   The ``--analytic`` argument can be used to run an analytic simulation instead of a numerical one.

Computation of the Functional Renormalization Group Equations in Local Potential Approximation
-------------------------------------------------------------------------------------------------------

The file ``frg_equations_lpa.py`` can be used to compute the functional renormalization group equations in the Local Potential Approximation (LPA) with an expansion around a non trivial vacuum:


.. code:: bash

   ./scripts/frg_equations_lpa.py \
      --config /path/to/config.yaml

.. note::

   The ``--analytic`` argument can be used to run an analytic simulation instead of a numerical one.

Analysis of the Eigenvector Components
--------------------------------------

The script ``evc_distribution.py`` computes the distribution of the eigenvectors of the correlations:

.. code:: bash

   ./scripts/evc_distribution.py \
      --config /path/to/config.yaml
