# From official organiser of ICFHR 2020 Competition
# https://github.com/anguelos/wi19_evaluate

import numpy as np


def compute_pr_a_k(sorted_retrievals, k):
    pr_a_k = sorted_retrievals[:, :k].sum(axis=1) / np.minimum(sorted_retrievals.sum(axis=1), k)
    return pr_a_k.sum() / len(pr_a_k)


def get_metrics(distance_matrix, labels, remove_self_column=True):
    precision_at, recall_at, sorted_retrievals = get_precision_recall_matrices(
        distance_matrix, labels, remove_self_column)

    non_singleton_idx = sorted_retrievals.sum(axis=1) > 0
    mAP = compute_map(precision_at[non_singleton_idx, :], sorted_retrievals[non_singleton_idx, :])
    top_1 = sorted_retrievals[:, 0].sum() / len(sorted_retrievals)
    pr_a_k10 = compute_pr_a_k(sorted_retrievals, 10)
    pr_a_k100 = compute_pr_a_k(sorted_retrievals, 100)
    # roc = wi19_evaluate.compute_roc(sorted_retrievals)
    return mAP, top_1, pr_a_k10, pr_a_k100


def get_sorted_retrievals(D, classes, remove_self_column=True, correct_retrievals=None):
    if correct_retrievals is None:
        correct_retrievals = classes[None, :] == classes[:, None]
    sorted_indexes = np.argsort(D, axis=1)
    if remove_self_column:
        sorted_indexes = sorted_indexes[:, 1:]  # removing self
    sorted_retrievals = correct_retrievals[np.arange(
        sorted_indexes.shape[0], dtype="int64")[:, None], sorted_indexes]
    return sorted_retrievals


def get_precision_recall_matrices(D, classes, remove_self_column=True, correct_retrievals=None):
    sorted_retrievals = get_sorted_retrievals(
        D, classes, remove_self_column=remove_self_column, correct_retrievals=correct_retrievals)
    relevant_count = sorted_retrievals.sum(axis=1).reshape(-1, 1)
    precision_at = np.cumsum(sorted_retrievals, axis=1).astype(
        "float") / np.cumsum(np.ones_like(sorted_retrievals), axis=1)
    recall_at = np.cumsum(sorted_retrievals, axis=1).astype(
        "float") / np.maximum(relevant_count, 1)
    recall_at[relevant_count.reshape(-1) == 0, :] = 1
    return precision_at, recall_at, sorted_retrievals


def compute_map(precision_at, sorted_retrievals):
    # Removing singleton queries from mAP computation
    valid_entries = sorted_retrievals.sum(axis=1) > 0
    precision_at = precision_at[valid_entries, :]
    sorted_retrievals = sorted_retrievals[valid_entries, :]
    AP = (precision_at * sorted_retrievals).sum(axis=1) / \
        sorted_retrievals.sum(axis=1)
    return AP.mean()


def compute_fscore(sorted_retrievals, relevant_estimate):
    relevant_mask = np.cumsum(np.ones_like(
        sorted_retrievals), axis=1) <= relevant_estimate.reshape(-1, 1)
    tp = float((sorted_retrievals * relevant_mask).sum())
    retrieved = relevant_estimate.sum()
    relevant = sorted_retrievals.sum()
    precision = tp / retrieved
    recall = tp / relevant
    fscore = 2 * precision * recall / (precision + recall)
    return fscore, precision, recall


def compute_roc(sorted_retrievals):
    # https://en.wikipedia.org/wiki/Receiver_operating_characteristic @ 22/3/2019
    true_positives = sorted_retrievals.sum(axis=0).cumsum().astype("float")
    false_positives = (1-sorted_retrievals).sum(axis=0).cumsum().astype("float")
    relevant = np.ones_like(true_positives) * sorted_retrievals.sum()
    recalls = true_positives / relevant
    fallout = false_positives / (1-sorted_retrievals).sum()# FP+TN
    return {"fallout": np.array(fallout), "recall": np.array(recalls)}
