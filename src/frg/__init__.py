"""``frg`` is a Python package to deal with the functional renormalisation group for signal detection. It leverages the action of the RG on different sources of signal to compute a limit of detection and assess the presence of a signal.

The work is based on `theoretical advancements <https://arxiv.org/abs/2201.04250>`__ and previous `numerical works <https://arxiv.org/abs/2310.07499>`__ and `improvements <https://arxiv.org/abs/2507.01064>`__.
"""

from email.utils import getaddresses
from importlib.metadata import PackageNotFoundError, metadata

from frg.distributions.distributions import (
    EmpiricalDistribution,
    MarchenkoPastur,
)
from frg.utils.analysis import (
    add_values,
    canonical_dimensions_argsort,
    canonical_dimensions_files,
    canonical_dimensions_files_poiss,
    canonical_dimensions_ratio_files,
    compute_roi,
    direct_relative_adherence,
    extract_interp_values,
    interp_canonical_dimensions,
    plot_canonical_dimensions,
    plot_canonical_dimensions_eigenvalues,
    plot_canonical_dimensions_scan,
    plot_canonical_dimensions_scan2d,
    plot_distribution,
    plot_eigenvalues,
    plot_localization,
    plot_localization_scan,
    plot_potential,
    plot_ratio_scan,
    plot_symmetry_size,
    plot_symmetry_surface,
    plot_trajectories,
)
from frg.utils.utils import get_cfg_defaults, get_logger, load_data


def _parse_authors_and_emails(author_email_field: str) -> tuple[str, str]:
    """Parse a metadata Author-email field into author and email lists.

    Returns:
        A pair of comma-separated strings: authors, emails.
    """
    parsed = getaddresses([author_email_field])
    if not parsed:
        return "", ""

    authors = ", ".join(name for name, _ in parsed if name)
    emails = ", ".join(email for _, email in parsed if email)

    return authors, emails


try:
    _meta = metadata("frg-signal-detection")

    # Parse version
    __version__: str = _meta["Version"]

    # Parse author(s) and email(s)
    __author__: str
    __email__: str
    __author__, __email__ = _parse_authors_and_emails(
        _meta.get("Author-email", "")
    )
    __author__ = __author__ or _meta.get("Author", "")

    # Parse URL
    __url__: str = _meta.get(
        "Project-URL", "https://github.com/thesfinox/frg-signal-detection"
    )
except PackageNotFoundError:
    __version__ = "0.0.0-unknown"
    __author__ = "Riccardo Finotello, Parham Radpay"
    __email__ = "riccardo.finotello@cea.fr, parhamradpay@gmail.com"
    __url__ = "https://github.com/thesfinox/frg-signal-detection"

__license__ = "MIT License"

# Package imports
__all__ = [
    "EmpiricalDistribution",
    "MarchenkoPastur",
    "__author__",
    "__email__",
    "__license__",
    "__url__",
    "__version__",
    "add_values",
    "canonical_dimensions_argsort",
    "canonical_dimensions_files",
    "canonical_dimensions_files_poiss",
    "canonical_dimensions_ratio_files",
    "compute_roi",
    "direct_relative_adherence",
    "extract_interp_values",
    "get_cfg_defaults",
    "get_logger",
    "interp_canonical_dimensions",
    "load_data",
    "plot_canonical_dimensions",
    "plot_canonical_dimensions_eigenvalues",
    "plot_canonical_dimensions_scan",
    "plot_canonical_dimensions_scan2d",
    "plot_distribution",
    "plot_eigenvalues",
    "plot_localization",
    "plot_localization_scan",
    "plot_potential",
    "plot_ratio_scan",
    "plot_symmetry_size",
    "plot_symmetry_surface",
    "plot_trajectories",
]
