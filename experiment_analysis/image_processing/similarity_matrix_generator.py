import numpy as np 
from sklearn.manifold import MDS
import math
import pandas as pd

class SimilarityMatrixGenerator:
    abs_color_ids = {
        'round_1': {
            'brown': 9,
            'blue': 5,
            'red': 3,
            'green': 2,
            'purple': 6,
            'pink': 7,
            'grey': 8,
            'light-pink': 4,
            'orange': 1,
            'yellow': 10
        },
        'round_2': {
            'brown': 29,
            'blue': 25,
            'red': 23,
            'green': 22,
            'purple': 26,
            'pink': 27,
            'grey': 28,
            'light-pink': 24,
            'orange': 21,
            'yellow': 30
        }
    }

    def __init__(self):
        pass

    def generate_matrix(self, centers, round_id, stacked=True, averaged=False):
        abs_color_ids = self.abs_color_ids[round_id]
        n = len(centers)
        shape = (n, n)
        distances = np.empty(shape)
        for i, (x1, y1) in enumerate(centers.values()):
            for j, (x2, y2) in enumerate(centers.values()):
                d = math.sqrt((x2-x1)**2 + (y2-y1)**2)
                distances[i][j] = d
        indexes = []
        for key in centers.keys():
            wine_id = self.abs_color_ids[round_id][key]
            indexes.append(wine_id)
        df = pd.DataFrame(data=distances, columns=indexes, index=indexes)
        return df