def print_dict_pipe(d: dict):
    parts = []
    for k, v in d.items():
        parts.append(f"{k}={v}")
    return " | ".join(parts)