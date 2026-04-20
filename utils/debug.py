def get_leaf_tensors(fn, leaves=set()):
    if fn is None:
        return

    if hasattr(fn, "variable"):  # 叶子节点
        leaves.add(fn.variable)

    for next_fn, _ in fn.next_functions:
        get_leaf_tensors(next_fn, leaves)

    return leaves
