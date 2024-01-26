from enum import StrEnum, IntEnum

class StochasticModelEnum(StrEnum):
    RANDOM_WALK = "RANDOM_WALK"
    ARITHMETIC_BROWNIAN_MOTION = "arithmetic_brownian_motion"
    GEOMETRIC_BROWNIAN_MOTION = "geometric_brownian_motion"
    HESTON_WITH_JUMPS = "heston_with_jumps"