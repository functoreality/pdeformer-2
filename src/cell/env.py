r"""Some implementational settings."""
# We do not include [VNode] as in original Graphormer. This option currently
# has no effect.
USE_GLOBAL_NODE = False
# Dropout causes training failure on AICC machines. Reasons unknown.
ENABLE_DROPOUT = False
SPACE_DIM = 3
