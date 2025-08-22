import math
import heapq
import multiprocessing as mp
import numpy as np


def evaluate_model(model, test_ratings, test_negatives, K: int, num_threads: int = 1):
    """
    Evaluate recommendation performance with leave-one-out.

    Args:
        model: Keras model for prediction.
        test_ratings: List of [user, ground_truth_item].
        test_negatives: List of negative items for each user.
        K: Number of top items for evaluation.
        num_threads: Number of parallel threads.

    Returns:
        hits: List[int], hit ratio per test case.
        ndcgs: List[float], NDCG score per test case.
    """
    if num_threads > 1:
        with mp.Pool(processes=num_threads) as pool:
            res = pool.map(
                lambda idx: _eval_one_rating(model, test_ratings, test_negatives, K, idx),
                range(len(test_ratings))
            )
        hits, ndcgs = zip(*res)
    else:
        hits, ndcgs = [], []
        for idx in range(len(test_ratings)):
            hr, ndcg = _eval_one_rating(model, test_ratings, test_negatives, K, idx)
            hits.append(hr)
            ndcgs.append(ndcg)

    return list(hits), list(ndcgs)


def _eval_one_rating(model, test_ratings, test_negatives, K: int, idx: int):
    """
    Evaluate a single test rating.

    Args:
        model: Keras model.
        test_ratings: Ground-truth pairs.
        test_negatives: Negative items per test case.
        K: Top-K.
        idx: Index of test rating.

    Returns:
        (hit, ndcg)
    """
    user, gt_item = test_ratings[idx]
    items = list(test_negatives[idx])  # copy negatives
    items.append(gt_item)

    # Predict scores
    users = np.full(len(items), user, dtype=np.int32)
    predictions = model.predict([users, np.array(items)], batch_size=100, verbose=0).flatten()

    # Map items to predicted scores
    item_score = {item: predictions[i] for i, item in enumerate(items)}

    # Get top-K items
    ranklist = heapq.nlargest(K, item_score, key=item_score.get)

    hr = get_hit_ratio(ranklist, gt_item)
    ndcg = get_ndcg(ranklist, gt_item)
    return hr, ndcg


def get_hit_ratio(ranklist, gt_item) -> int:
    """Return 1 if ground-truth item is in top-K, else 0."""
    return int(gt_item in ranklist)


def get_ndcg(ranklist, gt_item) -> float:
    """Compute Normalized Discounted Cumulative Gain (NDCG)."""
    if gt_item in ranklist:
        index = ranklist.index(gt_item)
        return math.log(2) / math.log(index + 2)
    return 0.0
