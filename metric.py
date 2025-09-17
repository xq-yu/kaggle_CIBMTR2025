import numpy as np
from numba import jit
import pandas as pd
from lifelines.utils import concordance_index

@jit(nopython=True)
def init_BTree(values):
    times_to_compare = np.empty_like(values)
    last_full_row = int(np.log2(len(values) + 1) - 1)
    len_ragged_row = len(values) - (2 ** (last_full_row + 1) - 1)
    if len_ragged_row > 0:
#        bottom_row_ix = np.s_[: 2 * len_ragged_row : 2]
        bottom_row_ix = slice(None, 2 * len_ragged_row, 2)
        times_to_compare[-len_ragged_row:] = values[bottom_row_ix]
        values = np.delete(values, bottom_row_ix)
    values_start = 0
    values_space = 2
    values_len = 2 ** last_full_row
    while values_start < len(values):
        times_to_compare[values_len - 1 : 2 * values_len - 1] = values[values_start::values_space]
        values_start += int(values_space / 2)
        values_space *= 2
        values_len = int(values_len / 2)
    return times_to_compare

@jit(nopython=True)
def insert(counts, pred, times_to_compare):
    i = 0
    n = len(times_to_compare)
    while (i < n):
        cur = times_to_compare[i]
        counts[i] += 1
        if pred < cur:
            i = 2 * i + 1
        elif pred > cur:
            i = 2 * i + 2
        else:
            return counts
    #raise ValueError("Value %s not contained in tree." "Also, the counts are now messed up." % times_to_compare)

@jit(nopython=True)
def fn_rank(pred, times_to_compare, counts):
    i = 0
    n = len(times_to_compare)
    rank = 0
    count = 0
    while (i < n):
        cur = times_to_compare[i]
        if pred < cur:
            i = 2 * i + 1
            continue
        elif pred > cur:
            rank += counts[i]
            # subtract off the right tree if exists
            nexti = 2 * i + 2
            if nexti < n:
                rank -= counts[nexti]
                i = nexti
                continue
            else:
                return rank, count
        else:  # value == cur
            count = counts[i]
            lefti = 2 * i + 1
            if lefti < n:
                nleft = counts[lefti]
                count -= nleft
                rank += nleft
                righti = lefti + 1
                if righti < n:
                    count -= counts[righti]
            return rank, count
    return rank, count


@jit(nopython=True)
def handle_pairs(truth, pred, first_ix, times_to_compare, counts):
    next_ix = first_ix
    while next_ix < len(truth) and truth[next_ix] == truth[first_ix]:
        next_ix += 1
    pairs = counts[0] * (next_ix - first_ix)
    correct = np.int64(0)
    tied = np.int64(0)
    for i in range(first_ix, next_ix):
#        rank, count = times_to_compare.rank(censored_pred[i])
        rank, count = fn_rank(pred[i], times_to_compare, counts)
        correct += rank
        tied += count
    return (pairs, correct, tied, next_ix)


@jit(nopython=True)
def fast_concordance_index(event_times, predicted_event_times, event_observed):
    
    died_mask = event_observed==1#.astype(bool)
    # TODO: is event_times already sorted? That would be nice...
    died_truth = event_times[died_mask]
    ix = np.argsort(died_truth)
    died_truth = died_truth[ix]
    died_pred = predicted_event_times[died_mask][ix]

    censored_truth = event_times[~died_mask]
    ix = np.argsort(censored_truth)
    censored_truth = censored_truth[ix]
    censored_pred = predicted_event_times[~died_mask][ix]

    censored_ix = 0
    died_ix = 0
    
    times_to_compare = init_BTree(np.unique(died_pred))
#    counts = np.zeros_like(times_to_compare, dtype=int)
    counts = np.full(len(times_to_compare), 0)
    
    num_pairs = np.int64(0)
    num_correct = np.int64(0)
    num_tied = np.int64(0)

    # we iterate through cases sorted by exit time:
    # - First, all cases that died at time t0. We add these to the sortedlist of died times.
    # - Then, all cases that were censored at time t0. We DON'T add these since they are NOT
    #   comparable to subsequent elements.
    while True:
        has_more_censored = censored_ix < len(censored_truth)
        has_more_died = died_ix < len(died_truth)
        # Should we look at some censored indices next, or died indices?
        if has_more_censored and (not has_more_died or died_truth[died_ix] > censored_truth[censored_ix]):
            pairs, correct, tied, next_ix = handle_pairs(censored_truth, censored_pred, censored_ix, times_to_compare, counts)
            censored_ix = next_ix
            
        elif has_more_died and (not has_more_censored or died_truth[died_ix] <= censored_truth[censored_ix]):
            pairs, correct, tied, next_ix = handle_pairs(died_truth, died_pred, died_ix, times_to_compare, counts)

            for pred in died_pred[died_ix:next_ix]:
                insert(counts, pred, times_to_compare)
                                
            died_ix = next_ix
        else:
            assert not (has_more_died or has_more_censored)
            break

        num_pairs += pairs
        num_correct += correct
        num_tied += tied
        
#    print(num_pairs, num_correct, num_tied)
    return (num_correct + num_tied / 2) / (num_pairs+0.01)

def CIBMTR_score(y, y_hat,efs,race_group):
    merged_df = pd.DataFrame({'y':y,'y_hat':y_hat,'efs':efs,'race_group':race_group})
    merged_df = merged_df.reset_index(drop=True)
    merged_df_race_dict = dict(merged_df.groupby(['race_group']).groups)
    metric_list = []
    race_list = []
    for race in merged_df_race_dict.keys():
        indices = sorted(merged_df_race_dict[race])
        merged_df_race = merged_df.iloc[indices]
        c_index_race = fast_concordance_index(
                        np.array(merged_df_race['y']),
                        np.array(merged_df_race['y_hat']),
                        np.array(merged_df_race['efs']))
        metric_list.append(c_index_race)
        race_list.append(race)
    return float(np.mean(metric_list)-np.sqrt(np.var(metric_list))),np.sqrt(np.var(metric_list)),{race:cindex for race,cindex in zip(race_list,metric_list)}

