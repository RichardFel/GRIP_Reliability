import numpy as np
from neurokit2.complexity import complexity_lempelziv
# import EntropyHub as EH
import pandas as pd
import numpy as np
from SRC.utils import *
from collections import Counter


def characteristics(results, data):
    results['Epochs_of_1minute'] = len(data)
    # Info all data
    # results['average_activity_level'] = np.mean(data)

    # Info about changes
    differences = np.diff(data)
    # results['std_change'] = np.std(differences)
    indx_changes = np.where(differences != 0)[0] + 1
    results['per_change'] = len(indx_changes) / len(data) * 100

    results = transitions(data, results)
    # per epoch (can be longer than ML-output length)
    if indx_changes.size == 0:
        epochs = [data]
    else:
        epochs = []
        epochs.append(data[0:indx_changes[0]])
        for num, indx in enumerate(indx_changes[:-1]):
            epochs.append(data[indx:indx_changes[num+1]])
        epochs.append(data[indx_changes[-1]:])

    # Set epoch per activity
    sedentairy = []
    light = []
    moderate = []
    for epoch in epochs:
        if epoch[0] == 0:
            sedentairy.append(epoch)
        elif epoch[0] == 1:
            light.append(epoch)
        elif epoch[0] == 2:
            moderate.append(epoch)

    # Get results per activity
    unique, counts = np.unique(data, return_counts=True)
    for i in zip(unique, counts):
        if i[0] == 0:
            activity = 'sedentairy'
        elif i[0] == 1:
            activity = 'light'
        elif i[0] == 2:
            activity = 'moderate'
        results[f'{activity}_count'] = i[1]
        results[f'{activity}_perc'] = i[1] / len(data) * 100

        # Percentage epochs
        activity_epochs = eval(activity)
        results[f'epochs_{activity}_perc'] = len(
            activity_epochs) / len(epochs) * 100

        # Median and average epoch length
        bout_lengths = [len(sublist) for sublist in activity_epochs]
        results.update(alfa_sigma_gini(bout_lengths, activity))
        results.update(alfa_sigma_gini_new(bout_lengths, activity))
        results[f'{activity}_standard_deviation_bout'] = np.std(bout_lengths)
        results[f'{activity}_median_length'] = np.median(bout_lengths)
        results[f'epochs_{activity}_average_length'] = np.average(bout_lengths)
        results[f'epochs_{activity}_max_length'] = np.max(bout_lengths)
        results[f'epochs_{activity}_IQR_high'] = np.percentile(bout_lengths, 75) 
        results[f'epochs_{activity}_IQR_low'] = np.percentile(bout_lengths, 25)

    # Complexity features
    # results[f'Sample_entropy_m2_tau1'] = EH.SampEn(data, m=2, tau=1)[0][-1]
    # results[f'Sample_entropy_m3_tau1'] = EH.SampEn(data, m=3, tau=1)[0][-1]
    # results[f'Sample_entropy_m4_tau1'] = EH.SampEn(data, m=4, tau=1)[0][-1]
    # results[f'Sample_entropy_m2_tau2'] = EH.SampEn(data, m=2, tau=2)[0][-1]
    # results[f'Sample_entropy_m3_tau2'] = EH.SampEn(data, m=3, tau=2)[0][-1]
    # results[f'Sample_entropy_m4_tau2'] = EH.SampEn(data, m=4, tau=2)[0][-1]
    # p_vector = probability(np.array(data)+1)
    # results[f'info_entropy'] = entropy(p_vector)
    # Markov entropy (entropy rate of the state sequence)
    results['markov_entropy_order1'], results['markov_entropy_order1_norm'] = markov_entropy(data, order=1)
    results['markov_entropy_order2'], results['markov_entropy_order2_norm'] = markov_entropy(data, order=2)
    per_state = markov_entropy_per_state(data)
    for state, name in enumerate(['sedentairy', 'light', 'moderate']):
        results[f'markov_entropy_from_{name}'] = per_state[state]
    
    # Lempel-Ziv complexity. delay/dimension are ignored by neurokit when permutation=False,
    # so only one binarised (mean-split) value is kept for comparison with earlier runs.
    results['LZC_complexity'], _ = complexity_lempelziv(data, permutation=False)
    # results['LZC'] = lempelziv(data)
    # results['LZC_window_120min'] = lempelziv_windowed(data, window=120)
    return results


# def lempelziv(data, n_states=3):
#     """LZ76 complexity on the raw activity states, normalised as c * log_k(n) / n."""
#     data = np.asarray(data).astype(int)
#     n = len(data)
#     _, info = complexity_lempelziv(data, permutation=False, symbolize=None)
#     return info['Complexity_Kolmogorov'] * np.log(n) / np.log(n_states) / n


# def lempelziv_windowed(data, window=120, n_states=3):
#     """
#     Mean LZC over fixed-length windows, so the value does not depend on the
#     measurement duration. Windows are non-overlapping from the start of the
#     day; any remainder is covered by one extra window aligned to the end.
#     """
#     data = np.asarray(data)
#     n = len(data)
#     if n < window:
#         return np.nan
#     starts = list(range(0, n - window + 1, window))
#     if n % window:
#         starts.append(n - window)
#     return np.mean([lempelziv(data[s:s + window], n_states) for s in starts])

def transitions(data, results):
    # Calculates the time normalised transitions from sedentairy to light etc.
    for i in range(3):
        for j in range(3):
            if i != j:
                results[f'norm_transitions_{i}_{j}'] = 0

    for num, _ in enumerate(data[:-1]):
        if data[num] - data[num-1] != 0:
            results[f'norm_transitions_{int(data[num])}_{int(data[num-1])}'] += 1

    for i in range(3):
        for j in range(3):
            if i != j:
                results[f'norm_transitions_{i}_{j}'] /= (len(data) / 100)
    return results

def alfa_sigma_gini(bout_lengths, activity):
    # Calculates the alfa, sigma and gini index
    outcomes = {}
    if len(Counter(bout_lengths)) <= 1:
        outcomes[f'weight_median_{activity}'] = np.nan
        outcomes[f'alfa_{activity}'] = np.nan
        outcomes[f'sigma_{activity}'] = np.nan
        outcomes[f'gini_{activity}'] = np.nan
    else:
        # Sort data
        arraySort = np.sort(bout_lengths)
        totalTim = np.sum(bout_lengths)

        # Weighted median
        cumsumnewWeighted = np.cumsum(arraySort / totalTim)
        nearest = find_nearest(cumsumnewWeighted, 0.5)
        weightedMedian = arraySort[np.where(
            cumsumnewWeighted == nearest)[0]][0]
        outcomes[f'weight_median_{activity}'] = weightedMedian
        # Alfa and sigma
        # temp = []
        # for idx in range(len(bout_lengths)):
        #     temp.append((np.log(bout_lengths[idx] / 1)))
        # n = len(bout_lengths)
        # temp = sum(temp) ** -1
        # alfa = 1 + len(bout_lengths) * temp
        # sigma = (alfa - 1) / np.sqrt(n)
        # outcomes[f'alfa_{activity}'] = alfa
        # outcomes[f'sigma_{activity}'] = sigma

        # Gini
        arraySortPerc = (arraySort / sum(arraySort)) * 100
        arraySortPerc_shares = [i / 100 for i in arraySortPerc]
        arraySortPerc_quintile_shares = [(arraySortPerc_shares[i] + arraySortPerc_shares[i + 1]) for i in
                                         range(0, len(arraySortPerc_shares) - 1, 2)]
        arraySortPerc_quintile_shares.insert(0, 0)
        shares_cumsum = np.cumsum(
            a=arraySortPerc_quintile_shares, axis=None)
        pe_line = np.linspace(start=0.0, stop=1.0, num=len(shares_cumsum))
        x = np.arange(0, 1, 1 / len(shares_cumsum))
        area_under_lorenz = np.trapz(
            y=shares_cumsum, dx=1 / len(shares_cumsum))
        area_under_pe = np.trapz(y=pe_line, dx=1 / len(shares_cumsum))
        gini = (area_under_pe - area_under_lorenz) / area_under_pe
        outcomes[f'gini_{activity}'] = gini
    return outcomes

def alfa_sigma_gini_new(bout_lengths, activity):
    """Weighted median, power-law alfa/sigma and Gini index of a list of bout lengths."""
    outcomes = {}
    bout_lengths = np.asarray(bout_lengths, dtype=float)

    if len(bout_lengths) == 0 or len(Counter(bout_lengths)) <= 1:
        outcomes[f'weight_median_{activity}'] = np.nan
        # outcomes[f'alfa_{activity}'] = np.nan
        # outcomes[f'sigma_{activity}'] = np.nan
        outcomes[f'gini_{activity}'] = np.nan
        return outcomes

    arraySort = np.sort(bout_lengths)
    n = len(arraySort)
    totalTim = arraySort.sum()

    # Weighted median: the bout length at which half of the total time is reached
    cumsumWeighted = np.cumsum(arraySort / totalTim)
    idx = np.argmin(np.abs(cumsumWeighted - 0.5))
    outcomes[f'weight_median_{activity}'] = arraySort[idx]

    # Alfa and sigma (Clauset et al. MLE, x_min = 1)
    # log_sum = np.sum(np.log(arraySort))          # log(x / 1) == log(x)
    # if log_sum > 0:
    #     alfa = 1 + n / log_sum
    #     sigma = (alfa - 1) / np.sqrt(n)
    # else:
    #     alfa = sigma = np.nan
    # outcomes[f'alfa_{activity}'] = alfa
    # outcomes[f'sigma_{activity}'] = sigma

    # Gini index (closed form on the sorted array; no binning, no integration)
    ranks = np.arange(1, n + 1)
    gini = (2 * np.sum(ranks * arraySort)) / (n * totalTim) - (n + 1) / n
    outcomes[f'gini_V2_{activity}'] = gini

    return outcomes
def characteristics_pain(painscores, results, day, time=None):
    # Calculates the pain score characetristices per period
    painscores['Date'] = pd.to_datetime(
        painscores['Date'], format='mixed', dayfirst=True, errors='coerce')
    painscores['Standardized Date'] = painscores['Date'].dt.strftime(
        '%Y-%m-%d')
    painscore = painscores.loc[painscores['Standardized Date'] == day]
    if painscore.empty:
        painscores['Standardized Date'] = painscores['Date'].dt.strftime(
            '%Y-%d-%m')
        painscore = painscores.loc[painscores['Standardized Date'] == day]

    if time:
        results['pijn_score'] = painscore[time].values[0]
        results['tijd'] = time
    else:
        results['pijn_gem'] = painscore.loc[:,
                                            '6:00':'0:00'].dropna(axis=1).values.mean()
        results['pijn_std'] = painscore.loc[:,
                                            '6:00':'0:00'].dropna(axis=1).values.std()
        results['pijn_max'] = painscore.loc[:,
                                            '6:00':'0:00'].dropna(axis=1).values.max()
    return results


def markov_entropy(data, order=1, n_states=3, base=2):
    """
    Entropy rate of the activity sequence modelled as a Markov chain of the
    given order (plug-in estimator).

        H = - sum_s p(s) * sum_j P(j | s) * log P(j | s)

    s is the preceding state (or tuple of `order` states), p(s) the empirical
    frequency of s as a "from" state, P(j | s) the empirical transition
    probability. Returns (H, H_normalised); the normalised value divides by
    log(n_states): 0 = fully predictable, 1 = transitions uniformly random.
    """
    data = np.asarray(data).ravel()
    if len(data) <= order:
        return np.nan, np.nan

    counts = Counter()
    for i in range(order, len(data)):
        counts[(tuple(data[i - order:i]), data[i])] += 1

    n_total = sum(counts.values())
    from_totals = Counter()
    for (hist, _), c in counts.items():
        from_totals[hist] += c

    H = 0.0
    for (hist, _), c in counts.items():
        p_joint = c / n_total              # p(s, j)
        p_cond = c / from_totals[hist]     # P(j | s)
        H -= p_joint * np.log(p_cond)
    H /= np.log(base)

    H_norm = H / (np.log(n_states) / np.log(base)) if n_states > 1 else np.nan
    return H, H_norm


def markov_entropy_per_state(data, n_states=3, base=2):
    """Conditional entropy of the next state given each current state (first order)."""
    data = np.asarray(data).ravel()
    T = np.zeros((n_states, n_states))
    for a, b in zip(data[:-1], data[1:]):
        T[int(a), int(b)] += 1

    out = {}
    for s in range(n_states):
        row = T[s]
        if row.sum() == 0:
            out[s] = np.nan
            continue
        p = row / row.sum()
        p = p[p > 0]
        out[s] = -np.sum(p * np.log(p)) / np.log(base)
    return out