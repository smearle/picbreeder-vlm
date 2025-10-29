import random
from typing import Optional

import numpy


def apply_random_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    try:
        import numpy
    except ImportError:
        return
    numpy.random.seed(seed)
