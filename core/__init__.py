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
from .conv_rf_mmse import (
    make_random_conv_filters,
    conv_rf_features,
    multiscale_conv_rf_features,
    dense_wiener_precompute,
    dense_wiener_predict,
    dense_wiener_loss_from_precomp,
    accumulate_conv_rf_stats,
    accumulate_dense_wiener_residual_stats,
    accumulate_multiscale_dense_wiener_residual_stats,
    accumulate_dense_wiener_residual_patch_stats,
    mmse_from_conv_stats,
    mmse_from_patch_stats,
    conv_linear_mmse_fft,
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
    # conv RF
    "make_random_conv_filters", "conv_rf_features", "multiscale_conv_rf_features",
    "dense_wiener_precompute", "dense_wiener_predict",
    "dense_wiener_loss_from_precomp",
    "accumulate_conv_rf_stats", "mmse_from_conv_stats",
    "accumulate_dense_wiener_residual_stats",
    "accumulate_multiscale_dense_wiener_residual_stats",
    "accumulate_dense_wiener_residual_patch_stats",
    "mmse_from_patch_stats",
    "conv_linear_mmse_fft",
]
