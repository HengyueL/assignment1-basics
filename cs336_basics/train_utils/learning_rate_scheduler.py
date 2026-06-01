import math


def cosine_annealing_scheduler(
    t: int,
    alpha_max: float,
    alpha_min: float,
    T_w: int,
    T_c: int
):
    if t < T_w:
        alpha_t = t / T_w * alpha_max
    elif T_w <= t <= T_c:
        alpha_t = alpha_min + 0.5 * (1 + math.cos((t - T_w) / (T_c - T_w) * math.pi)) * (alpha_max - alpha_min)
    else:
        alpha_t = alpha_min
    return alpha_t