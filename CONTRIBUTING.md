# Contributing to HyperPCA

> **Everyone** is welcome and encouraged to contribute to **HyperPCA**!

We are happy to receive pull requests and issues on the **GitLab** repository.
Make sure to respect the work of others and the phylosophy of this piece of software:

1. keep it **simple and clear**,
2. keep it **consistent** (i.e. use the same style),
3. keep it **compatible** (i.e. be `scikit-learn`-compatible),
4. use a **pre-commit** pipeline to check and format your code.

> 💡 PRE-COMMIT
>
> The **pre-commit** pipeline can be activated by installing the `pre-commit` package:
>
> ```bash
> pip install pre-commit
> ```
>
> You will then be able to run `pre-commit install` to locally install the pipeline to check and format your code.
>
> You can optionally run `pre-commit run --all-files` to apply the pipeline to all files in the local copy of the repository.

On top of these simple rules, two more fundamental guidelines apply:

1. **document** your code (comments, _docstrings_, Sphinx documentation, README, etc.),
2. write and use **tests** to avoid bugs.

> 🛑 **WARNING**
>
> Failure to comply with the first set of rules will end in an infinite **waste of time** due to discussions between developers (and **no progress**).

> 🛑 **WARNING**
>
> Failure to comply with the second set of rules will **immediately** result in the **rejection** of the pull request.

## Guidelines

In order to contribute, you can proceed as follows:

1. fork the project in your own space,
2. code your own modules or modify the existing,
3. **document** your code (see below for a template _docstring_ for your functions),
4. implement **unitary tests** for your code (take a look at the `tests` folder),
5. submit a pull request.

> 💡 DOCUMENTATION
>
> Please be mindful that code without documentation and without tests is not usable!
> Please, provide as much documentation and testing suites as possible.
> Ideally, tests and documentation should reflect the same high quality standards of your code!

> 💡 TYPING
>
> Use **typing** (type hints) in your Python code!
> This helps users to track what kind of data types to provide as input and to expect as output.

> 🛑 TESTS
>
> Tests are implemented using the `pytest` suite.
> You can run the tests by running:
>
> ```bash
> pytest
> ```
>
> in the root directory of the project.

The preferred format for _docstrings_ is the _NumPy_ format:

```python
def foo(x: float, y: float) -> float:
    """
    This function is a placeholder for any function that will be implemented.

    Parameters
    ----------
    x : float
        A first float number.
    y : float
        A second float number.

    Returns
    -------
    float
        The sum of x and y.

    Raises
    ------
    ValueError
        If x or y are not float numbers.
    """
    if (not isinstance(x, float)) or (not isinstance(y, float)):
      raise ValueError('This is a generic error that you might want to raise')
    return x + y

class BarClass:
    """Example class."""

    def __init__(self, x: float):
      """
      Parameters
      ----------
      x : float
          A float number.

      Raises
      ------
      ValueError
          If x is not a float number.
      """
      if not isinstance(x, float):
        raise ValueError('This is a generic error that you might want to raise')
      self.x = x

    def bar(self, y: float) -> float:
        """
        Function explanation.

        Parameters
        ----------
        y : float
            A float number.

        Returns
        -------
        float
            The sum of x and y.
        """
        return self.x + y
```
