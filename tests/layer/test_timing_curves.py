import pytest
from pyfame import timing_constant, timing_gaussian, timing_linear, timing_sigmoid

@pytest.mark.parametrize(
    "timing_func",
    [
        timing_linear,
        timing_sigmoid,
        timing_gaussian,
        timing_constant
    ]
)
@pytest.mark.parametrize(
    "param, val",
    [
        ("time_delta", "tst"),
        ("time_start", "tst"),
        ("time_end", "tst"),
        ("positive_slope", "tst")
    ]
)
def test_timing_curves_invalid_param_types(timing_func, param, val):
    kwargs = {
        "time_delta" : 1.85,
        "time_start" : 0.5,
        "time_end" : 2.0,
        "positive_slope" : True
    }
    kwargs[param] = val

    with pytest.raises(ValueError):
        timing_func(**kwargs)