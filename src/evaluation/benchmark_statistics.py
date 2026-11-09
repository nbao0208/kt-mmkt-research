"""
Statistical toolkit for the unified final test benchmark of KT / MMKT models.

Provides:
1. `compute_binary_metrics`: AUC, ACC, RMSE, LogLoss (BCE), Brier score, Expected Calibration Error.
2. `reliability_curve`: binned calibration curve for reliability diagrams.
3. `student_macro_auc`: per-student AUC averaged over students having both response classes.
4. `fast_delong`: O(n log n) DeLong covariance estimator (Sun & Xu, 2014) for correlated ROC-AUCs
   of several models evaluated on the same interactions, plus pairwise z-tests.
5. `holm_bonferroni`: family-wise error-rate correction for multiple pairwise comparisons.
6. `StudentClusterBootstrap`: student-level (cluster) bootstrap of AUC / ACC and pairwise differences.
   Interactions of the same student are correlated, so resampling students (not interactions)
   produces honest confidence intervals. Weighted AUC is computed in O(n) per replicate thanks to
   a one-off pre-sorting of every prediction vector.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.stats as stats
from sklearn.metrics import roc_auc_score


_EPS = 1e-7


def expected_calibration_error(
    preds: np.ndarray,
    targets: np.ndarray,
    num_bins: int = 15,
) -> float:
    """
    Expected Calibration Error with equal-width probability bins.

    ECE = sum_b (|B_b| / N) * |mean(y in B_b) - mean(p in B_b)|

    Args:
        preds: Predicted probabilities, shape [N].
        targets: Binary labels, shape [N].
        num_bins: Number of equal-width bins on [0, 1].

    Returns:
        Scalar ECE value.
    """
    preds = np.asarray(preds, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if len(preds) == 0:
        return 0.0
    bin_ids = np.clip((preds * num_bins).astype(np.int64), 0, num_bins - 1)
    counts = np.bincount(bin_ids, minlength=num_bins).astype(np.float64)
    sum_p = np.bincount(bin_ids, weights=preds, minlength=num_bins)
    sum_y = np.bincount(bin_ids, weights=targets, minlength=num_bins)
    nonzero = counts > 0
    gap = np.abs(sum_y[nonzero] / counts[nonzero] - sum_p[nonzero] / counts[nonzero])
    return float(np.sum(counts[nonzero] / len(preds) * gap))


def reliability_curve(
    preds: np.ndarray,
    targets: np.ndarray,
    num_bins: int = 15,
) -> Dict[str, List[float]]:
    """
    Compute a binned reliability curve (mean predicted probability vs. empirical accuracy per bin).

    Returns:
        Dict with 'bin_mean_pred', 'bin_empirical_rate', 'bin_count' lists (empty bins dropped).
    """
    preds = np.asarray(preds, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    bin_ids = np.clip((preds * num_bins).astype(np.int64), 0, num_bins - 1)
    counts = np.bincount(bin_ids, minlength=num_bins).astype(np.float64)
    sum_p = np.bincount(bin_ids, weights=preds, minlength=num_bins)
    sum_y = np.bincount(bin_ids, weights=targets, minlength=num_bins)
    nonzero = counts > 0
    return {
        "bin_mean_pred": (sum_p[nonzero] / counts[nonzero]).tolist(),
        "bin_empirical_rate": (sum_y[nonzero] / counts[nonzero]).tolist(),
        "bin_count": counts[nonzero].astype(int).tolist(),
    }


def compute_binary_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
    num_ece_bins: int = 15,
) -> Dict[str, float]:
    """
    Compute the complete set of pointwise binary-prediction metrics used in the benchmark.

    Metrics:
        auc       : ROC-AUC (threshold-free ranking quality; primary pyKT metric).
        acc       : Accuracy at threshold 0.5 (secondary pyKT metric).
        rmse      : Root mean squared error between probability and label.
        log_loss  : Binary cross-entropy (negative log-likelihood per interaction).
        brier     : Brier score (= MSE).
        ece       : Expected calibration error (15 equal-width bins).
        pos_rate  : Empirical correct-response rate in the subset.
        mean_pred : Average predicted probability in the subset.

    Args:
        preds: Predicted probabilities, shape [N].
        targets: Binary labels, shape [N].
        num_ece_bins: Number of bins for ECE.

    Returns:
        Dictionary of metrics plus 'num_samples'.
    """
    preds = np.asarray(preds, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    n = int(len(targets))
    if n == 0:
        return {
            "auc": float("nan"), "acc": float("nan"), "rmse": float("nan"), "log_loss": float("nan"),
            "brier": float("nan"), "ece": float("nan"), "pos_rate": float("nan"),
            "mean_pred": float("nan"), "num_samples": 0,
        }
    if np.unique(targets).size < 2:
        auc = float("nan")  # AUC is undefined when only one class is present
    else:
        auc = float(roc_auc_score(targets, preds))
    clipped = np.clip(preds, _EPS, 1.0 - _EPS)
    sq_err = (preds - targets) ** 2
    return {
        "auc": auc,
        "acc": float(np.mean((preds >= 0.5).astype(np.int64) == targets)),
        "rmse": float(np.sqrt(np.mean(sq_err))),
        "log_loss": float(-np.mean(targets * np.log(clipped) + (1 - targets) * np.log(1.0 - clipped))),
        "brier": float(np.mean(sq_err)),
        "ece": expected_calibration_error(preds, targets, num_bins=num_ece_bins),
        "pos_rate": float(np.mean(targets)),
        "mean_pred": float(np.mean(preds)),
        "num_samples": n,
    }


def student_macro_auc(
    preds: np.ndarray,
    targets: np.ndarray,
    student_index: np.ndarray,
) -> Dict[str, float]:
    """
    Per-student ROC-AUC averaged across students (macro AUC).

    Only students whose evaluated interactions contain both correct and incorrect responses
    contribute (AUC is undefined otherwise). Complements the global micro AUC, which is dominated
    by students with very long trajectories.

    Args:
        preds: Predicted probabilities [N].
        targets: Binary labels [N].
        student_index: Dense student index in [0, S) for every interaction [N].

    Returns:
        Dict with 'macro_auc', 'std_auc', 'num_students_used'.
    """
    order = np.argsort(student_index, kind="stable")
    s_sorted = student_index[order]
    boundaries = np.flatnonzero(np.diff(s_sorted)) + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [len(s_sorted)]])
    p_sorted = preds[order]
    y_sorted = targets[order]
    aucs = []
    for a, b in zip(starts, ends):
        y = y_sorted[a:b]
        pos = int(y.sum())
        if pos == 0 or pos == len(y):
            continue
        p = p_sorted[a:b]
        # Mann-Whitney formulation with midranks handles ties exactly
        ranks = stats.rankdata(p)
        auc = (ranks[y == 1].sum() - pos * (pos + 1) / 2.0) / (pos * (len(y) - pos))
        aucs.append(auc)
    aucs = np.asarray(aucs, dtype=np.float64)
    return {
        "macro_auc": float(aucs.mean()) if len(aucs) else float("nan"),
        "std_auc": float(aucs.std()) if len(aucs) else float("nan"),
        "num_students_used": int(len(aucs)),
    }


def fast_delong(
    preds_matrix: np.ndarray,
    targets: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fast DeLong estimator of AUCs and their covariance matrix (Sun & Xu, IEEE SPL 2014).

    Args:
        preds_matrix: Array [K, N] of K models' predictions on the same N interactions.
        targets: Binary labels [N].

    Returns:
        Tuple (aucs [K], covariance [K, K]).
    """
    preds_matrix = np.atleast_2d(np.asarray(preds_matrix, dtype=np.float64))
    targets = np.asarray(targets, dtype=np.int64)
    pos_idx = np.flatnonzero(targets == 1)
    neg_idx = np.flatnonzero(targets == 0)
    m, n = len(pos_idx), len(neg_idx)
    k = preds_matrix.shape[0]

    v01 = np.empty((k, m), dtype=np.float64)
    v10 = np.empty((k, n), dtype=np.float64)
    aucs = np.empty(k, dtype=np.float64)
    for r in range(k):
        x = preds_matrix[r, pos_idx]
        y = preds_matrix[r, neg_idx]
        tx = stats.rankdata(x)
        ty = stats.rankdata(y)
        tz = stats.rankdata(np.concatenate([x, y]))
        aucs[r] = (tz[:m].sum() / m - (m + 1.0) / 2.0) / n
        v01[r] = (tz[:m] - tx) / n          # structural components of positives
        v10[r] = 1.0 - (tz[m:] - ty) / m    # structural components of negatives
    sx = np.atleast_2d(np.cov(v01))
    sy = np.atleast_2d(np.cov(v10))
    cov = sx / m + sy / n
    return aucs, cov


def delong_pairwise_tests(
    preds_by_model: Dict[str, np.ndarray],
    targets: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """
    Two-sided DeLong z-tests for every ordered pair (A, B) of models evaluated on the same data.

    Args:
        preds_by_model: Mapping model_name -> predictions [N].
        targets: Binary labels [N].

    Returns:
        Mapping "A__vs__B" -> {'auc_a', 'auc_b', 'delta', 'se', 'z', 'p_value', 'ci95_low', 'ci95_high'}.
    """
    names = list(preds_by_model.keys())
    mat = np.stack([preds_by_model[nm] for nm in names], axis=0)
    aucs, cov = fast_delong(mat, targets)
    results: Dict[str, Dict[str, float]] = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            var = cov[i, i] + cov[j, j] - 2.0 * cov[i, j]
            se = float(np.sqrt(max(var, 1e-300)))
            delta = float(aucs[i] - aucs[j])
            z = delta / se
            p = float(2.0 * stats.norm.sf(abs(z)))
            results[f"{names[i]}__vs__{names[j]}"] = {
                "model_a": names[i],
                "model_b": names[j],
                "auc_a": float(aucs[i]),
                "auc_b": float(aucs[j]),
                "delta": delta,
                "se": se,
                "z": float(z),
                "p_value": p,
                "ci95_low": delta - 1.959964 * se,
                "ci95_high": delta + 1.959964 * se,
            }
    return results


def holm_bonferroni(p_values: Dict[str, float]) -> Dict[str, float]:
    """
    Holm-Bonferroni step-down adjusted p-values (controls the family-wise error rate).

    Args:
        p_values: Mapping test_name -> raw p-value.

    Returns:
        Mapping test_name -> adjusted p-value (monotone, capped at 1).
    """
    names = sorted(p_values, key=lambda k: p_values[k])
    m = len(names)
    adjusted: Dict[str, float] = {}
    running_max = 0.0
    for rank, name in enumerate(names):
        adj = min(1.0, (m - rank) * p_values[name])
        running_max = max(running_max, adj)
        adjusted[name] = running_max
    return adjusted


def paired_fold_tests(
    fold_values_a: Sequence[float],
    fold_values_b: Sequence[float],
) -> Dict[str, float]:
    """
    Paired tests across matched cross-validation folds (fold f of A paired with fold f of B).

    Note: with only 5 folds the exact two-sided Wilcoxon signed-rank test has a minimum
    attainable p-value of 0.0625, so it can never reach p < 0.05; the paired t-test is reported
    alongside it.

    Returns:
        Dict with mean/std of differences, t statistic & p, Wilcoxon statistic & p, wins count.
    """
    a = np.asarray(fold_values_a, dtype=np.float64)
    b = np.asarray(fold_values_b, dtype=np.float64)
    diff = a - b
    t_res = stats.ttest_rel(a, b)
    try:
        w_res = stats.wilcoxon(a, b)
        w_stat, w_p = float(w_res.statistic), float(w_res.pvalue)
    except ValueError:
        w_stat, w_p = float("nan"), float("nan")
    return {
        "mean_diff": float(diff.mean()),
        "std_diff": float(diff.std(ddof=1)) if len(diff) > 1 else 0.0,
        "t_statistic": float(t_res.statistic),
        "p_value_ttest": float(t_res.pvalue),
        "wilcoxon_stat": w_stat,
        "p_value_wilcoxon": w_p,
        "num_folds_a_better": int(np.sum(diff > 0)),
        "num_folds": int(len(diff)),
    }


class _PresortedWeightedAUC:
    """
    Pre-sorted representation of one prediction vector enabling O(N) weighted-AUC evaluation.

    For interaction weights w (bootstrap multiplicities), the weighted Mann-Whitney AUC is:
        AUC_w = sum_g P_g * (N_<g + 0.5 * N_g) / (sum P * sum N)
    where g indexes groups of tied scores in ascending order, P_g / N_g are the total weights of
    positive / negative interactions in group g and N_<g is the cumulative negative weight of all
    strictly lower-scored groups.
    """

    def __init__(self, preds: np.ndarray, targets: np.ndarray):
        self.order = np.argsort(preds, kind="stable")
        sorted_preds = preds[self.order]
        change = np.empty(len(sorted_preds), dtype=bool)
        change[0] = True
        change[1:] = sorted_preds[1:] != sorted_preds[:-1]
        self.group_ids = np.cumsum(change) - 1
        self.num_groups = int(self.group_ids[-1]) + 1
        self.sorted_targets = targets[self.order].astype(np.float64)

    def auc(self, weights: np.ndarray) -> float:
        ws = weights[self.order]
        pos_w = ws * self.sorted_targets
        neg_w = ws - pos_w
        p_g = np.bincount(self.group_ids, weights=pos_w, minlength=self.num_groups)
        n_g = np.bincount(self.group_ids, weights=neg_w, minlength=self.num_groups)
        n_below = np.cumsum(n_g) - n_g
        total_p = p_g.sum()
        total_n = n_g.sum()
        return float(np.sum(p_g * (n_below + 0.5 * n_g)) / (total_p * total_n))


class StudentClusterBootstrap:
    """
    Student-level cluster bootstrap for AUC / ACC of several models and their pairwise differences.

    Each replicate draws S students with replacement (S = number of test students); every
    interaction inherits the multiplicity of its student as a weight. Metrics of all models are
    evaluated on the same replicate, so pairwise differences are properly paired.
    """

    def __init__(
        self,
        preds_by_model: Dict[str, np.ndarray],
        targets: np.ndarray,
        student_index: np.ndarray,
        num_replicates: int = 1000,
        seed: int = 2026,
    ):
        """
        Args:
            preds_by_model: Mapping model_name -> predictions [N] (aligned).
            targets: Binary labels [N].
            student_index: Dense student index [N] in [0, S).
            num_replicates: Number of bootstrap replicates B.
            seed: RNG seed for reproducibility.
        """
        self.names = list(preds_by_model.keys())
        self.preds = {k: np.asarray(v, dtype=np.float64) for k, v in preds_by_model.items()}
        self.targets = np.asarray(targets, dtype=np.int64)
        self.student_index = np.asarray(student_index, dtype=np.int64)
        self.num_students = int(self.student_index.max()) + 1
        self.num_replicates = num_replicates
        self.rng = np.random.default_rng(seed)
        self.sorters = {k: _PresortedWeightedAUC(v, self.targets) for k, v in self.preds.items()}
        self.correct = {k: ((v >= 0.5).astype(np.int64) == self.targets).astype(np.float64) for k, v in self.preds.items()}

    def run(self) -> Dict[str, np.ndarray]:
        """
        Execute the bootstrap.

        Returns:
            Dict model_name -> array [B, 2] of (AUC, ACC) replicates.
        """
        out = {k: np.empty((self.num_replicates, 2), dtype=np.float64) for k in self.names}
        for b in range(self.num_replicates):
            draws = self.rng.integers(0, self.num_students, size=self.num_students)
            multiplicity = np.bincount(draws, minlength=self.num_students).astype(np.float64)
            w = multiplicity[self.student_index]
            w_sum = w.sum()
            for k in self.names:
                out[k][b, 0] = self.sorters[k].auc(w)
                out[k][b, 1] = float(np.dot(w, self.correct[k]) / w_sum)
        return out

    @staticmethod
    def summarize(
        replicates: Dict[str, np.ndarray],
        point_estimates: Dict[str, Dict[str, float]],
        alpha: float = 0.05,
    ) -> Dict[str, Dict]:
        """
        Build percentile confidence intervals for each model and each ordered pair difference.

        Args:
            replicates: Output of `run`.
            point_estimates: Mapping model -> {'auc': .., 'acc': ..} computed on the full test set.
            alpha: Significance level (two-sided).

        Returns:
            {'per_model': {...}, 'pairwise': {...}} with CI bounds and bootstrap p-values.
        """
        lo_q, hi_q = 100 * alpha / 2.0, 100 * (1 - alpha / 2.0)
        per_model = {}
        for k, arr in replicates.items():
            per_model[k] = {
                "auc": point_estimates[k]["auc"],
                "auc_ci_low": float(np.percentile(arr[:, 0], lo_q)),
                "auc_ci_high": float(np.percentile(arr[:, 0], hi_q)),
                "auc_boot_se": float(arr[:, 0].std(ddof=1)),
                "acc": point_estimates[k]["acc"],
                "acc_ci_low": float(np.percentile(arr[:, 1], lo_q)),
                "acc_ci_high": float(np.percentile(arr[:, 1], hi_q)),
                "acc_boot_se": float(arr[:, 1].std(ddof=1)),
            }
        names = list(replicates.keys())
        pairwise = {}
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                d_auc = replicates[a][:, 0] - replicates[b][:, 0]
                d_acc = replicates[a][:, 1] - replicates[b][:, 1]
                num_b = len(d_auc)
                # Two-sided percentile bootstrap p-value for H0: delta = 0
                p_auc = min(1.0, 2.0 * min((np.sum(d_auc <= 0) + 1) / (num_b + 1), (np.sum(d_auc >= 0) + 1) / (num_b + 1)))
                p_acc = min(1.0, 2.0 * min((np.sum(d_acc <= 0) + 1) / (num_b + 1), (np.sum(d_acc >= 0) + 1) / (num_b + 1)))
                pairwise[f"{a}__vs__{b}"] = {
                    "model_a": a,
                    "model_b": b,
                    "delta_auc": point_estimates[a]["auc"] - point_estimates[b]["auc"],
                    "delta_auc_ci_low": float(np.percentile(d_auc, lo_q)),
                    "delta_auc_ci_high": float(np.percentile(d_auc, hi_q)),
                    "p_value_auc": float(p_auc),
                    "delta_acc": point_estimates[a]["acc"] - point_estimates[b]["acc"],
                    "delta_acc_ci_low": float(np.percentile(d_acc, lo_q)),
                    "delta_acc_ci_high": float(np.percentile(d_acc, hi_q)),
                    "p_value_acc": float(p_acc),
                }
        return {"per_model": per_model, "pairwise": pairwise}
