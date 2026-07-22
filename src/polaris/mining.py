from __future__ import annotations

from typing import Dict, FrozenSet, List, Optional


def _popcount(x: int) -> int:
    return x.bit_count()


def find_largest_frequent_itemset(
    transactions: List[FrozenSet[str]],
    minsup: int,
    max_k: int,
    max_search_items: int = 80,
) -> Optional[FrozenSet[str]]:
    if max_k <= 0 or minsup <= 0:
        return None
    n_tx = len(transactions)
    if n_tx == 0:
        return None

    item_bits: Dict[str, int] = {}
    for tid, t in enumerate(transactions):
        bit = 1 << tid
        for it in t:
            item_bits[it] = item_bits.get(it, 0) | bit

    freq = [(it, b) for it, b in item_bits.items() if _popcount(b) >= minsup]
    if not freq:
        return None
    freq.sort(key=lambda kv: (-_popcount(kv[1]), kv[0]))
    if max_search_items and len(freq) > int(max_search_items):
        freq = freq[: int(max_search_items)]
    items = [it for it, _ in freq]
    bits_arr = [b for _, b in freq]
    n = len(items)

    full = (1 << n_tx) - 1
    state = {"best_size": 0, "best_sup": 0, "matches": []}

    def dfs(start: int, prefix: List[str], bits: int, sup: int) -> None:
        size = len(prefix)
        if size > 0:
            bsize, bsup = state["best_size"], state["best_sup"]
            if size > bsize or (size == bsize and sup > bsup):
                state["best_size"], state["best_sup"] = size, sup
                state["matches"] = [tuple(prefix)]
            elif size == bsize and sup == bsup:
                state["matches"].append(tuple(prefix))
        if size == max_k:
            return
        for idx in range(start, n):
            reachable = min(max_k, size + (n - idx))
            if reachable < state["best_size"]:
                break
            if reachable == state["best_size"] and sup < state["best_sup"]:
                continue
            nb = bits & bits_arr[idx]
            ns = _popcount(nb)
            if ns < minsup:
                continue
            prefix.append(items[idx])
            dfs(idx + 1, prefix, nb, ns)
            prefix.pop()

    dfs(0, [], full, n_tx)
    matches = state["matches"]
    if not matches:
        return None
    best = max(matches, key=lambda p: tuple(sorted(p)))
    return frozenset(best)
