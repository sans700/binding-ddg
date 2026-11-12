import numpy as np

def ranks(x):
    x = np.asarray(x)
    order = np.argsort(x, kind='stable')
    result = np.empty(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]: j += 1
        result[order[i:j]] = (i + j - 1) / 2.0
        i = j
    return result

def correlation(a, b):
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12: return None
    return float(np.corrcoef(a, b)[0, 1])

def regression_metrics(y, p):
    y, p = np.asarray(y), np.asarray(p)
    denom = float(np.sum((y-y.mean())**2))
    return dict(n=len(y), rmse=float(np.sqrt(np.mean((y-p)**2))),
                mae=float(np.mean(np.abs(y-p))), pearson=correlation(y, p),
                spearman=correlation(ranks(y), ranks(p)),
                r2=(float(1-np.sum((y-p)**2)/denom) if denom > 1e-12 else None))

def cluster_bootstrap(y, p, groups, seed=704, repeats=1000):
    groups = np.asarray(groups)
    unique = np.unique(groups)
    if len(unique) < 5: return dict(groups=len(unique), note='Too few groups for a useful interval')
    rng = np.random.default_rng(seed)
    indices = {g:np.flatnonzero(groups == g) for g in unique}
    rmses = []
    for _ in range(repeats):
        sampled = rng.choice(unique, len(unique), replace=True)
        ix = np.concatenate([indices[g] for g in sampled])
        rmses.append(float(np.sqrt(np.mean((y[ix]-p[ix])**2))))
    return dict(groups=len(unique), repeats=repeats,
                rmse_95_percent_interval=np.quantile(rmses,[.025,.975]).tolist(),
                note='Grouped resampling interval on this fixed test set, not a universal generalization guarantee')
