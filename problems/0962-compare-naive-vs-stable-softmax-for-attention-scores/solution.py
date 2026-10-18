import numpy as np

def compare_softmax(scores: list) -> dict:
    """Compare naive and numerically stable softmax."""
    scores = np.asarray(scores, dtype=np.float64)

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        naive_exp = np.exp(scores)
        naive_sum = np.sum(naive_exp)
        naive_scores = naive_exp / naive_sum

    stable_scores = scores - np.max(scores)
    stable_scores = np.exp(stable_scores)
    stable_sum = np.sum(stable_scores)
    stable_scores = stable_scores / stable_sum

    naive_invalid = (
        not np.all(np.isfinite(naive_exp))
        or not np.isfinite(naive_sum)
        or not np.all(np.isfinite(naive_scores))
    )

    if naive_invalid:
        naive_scores = np.full_like(scores, np.nan)
        max_abs_diff = float('nan')
    else:
        max_abs_diff = float(np.max(np.abs(naive_scores - stable_scores)))
        max_abs_diff = np.round(max_abs_diff, 6)

    return {'naive': np.round(naive_scores, 6).tolist(), 'stable': np.round(stable_scores, 6).tolist(), 'max_abs_diff': max_abs_diff}
