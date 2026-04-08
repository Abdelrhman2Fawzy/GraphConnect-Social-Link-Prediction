
def weight_features(precomputed: dict, src: int, dst: int) -> dict:
    weight_in_map = precomputed["weight_in"]
    weight_out_map = precomputed["weight_out"]

    weight_in = weight_in_map.get(dst, 0.0)
    weight_out = weight_out_map.get(src, 0.0)

    return {
        "weight_in": weight_in,
        "weight_out": weight_out,
        "weight_in_plus_weight_out": weight_in + weight_out,
        "weight_in_mul_weight_out": weight_in * weight_out,
        "weight_in_2x_plus_weight_out": 2.0 * weight_in + weight_out,
        "weight_in_plus_weight_out_2x": weight_in + 2.0 * weight_out,
    }