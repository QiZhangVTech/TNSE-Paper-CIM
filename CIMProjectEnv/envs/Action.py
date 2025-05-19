from enum import IntEnum

class Actions(IntEnum):
    Unf = 0    # Uncertain Neighbor First
    Rnf = 1    # Reading Neighbor First
    Snf = 2    # Sharing Neighbor First
    Cf = 3     # Centrality First

    def __str__(self):
        return self.name