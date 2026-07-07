"""
CondDiffNonlinearTheory — core library for validating random-feature
conditional denoiser theory.
"""

from .hermite import (
    hermite_poly,
    hermite_poly_all,
    hermite_coeffs_mc,
    hermite_coeffs_batch,
    hermite_series_covariance,
    smoothed_activation_mc,
)
from .denoiser import (
    RandomFeatureMap,
    fit_optimal_denoiser,
    empirical_loss,
    empirical_covariances,
    theoretical_loss_from_cov,
    theoretical_Sigma_phi,
    theoretical_Cov_x0_phi,
)
from .gaussian import (
    JointGaussian,
    feature_gaussian_params,
    gaussian_theoretical_loss,
)
from .metrics import (
    mi_integrand,
    mi_sigma_sweep,
    explained_variance,
    conditioning_gain_r2,
    summarize_results,
)
from .dnn_estimator import (
    extract_features,
    build_conditional_features,
    mmse_from_features,
    wiener_filter_loss,
    wiener_filter_cond_loss,
)

__all__ = [
    # hermite
    "hermite_poly", "hermite_poly_all",
    "hermite_coeffs_mc", "hermite_coeffs_batch",
    "hermite_series_covariance", "smoothed_activation_mc",
    # denoiser
    "RandomFeatureMap",
    "fit_optimal_denoiser", "empirical_loss",
    "empirical_covariances", "theoretical_loss_from_cov",
    "theoretical_Sigma_phi", "theoretical_Cov_x0_phi",
    # gaussian
    "JointGaussian", "feature_gaussian_params", "gaussian_theoretical_loss",
    # metrics
    "mi_integrand", "mi_sigma_sweep",
    "explained_variance", "conditioning_gain_r2", "summarize_results",
]
