import random
from sklearn.metrics.pairwise import euclidean_distances

class TripletGenerator:
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

    def generate_triplets(self, centers, round_id):
        abs_color_ids = self.abs_color_ids[round_id]
        dot_distances = euclidean_distances(list(centers.values()), list(centers.values())).tolist()
        triplets = []
        color_keys = list(centers.keys())
        
        for i, c1 in enumerate(color_keys):
            for j, c2 in enumerate(color_keys):
                for k, c3 in enumerate(color_keys):
                    if i != j and i != k and j != k:
                        if dot_distances[i][j] > dot_distances[i][k]:
                            triplet = (abs_color_ids[c1], abs_color_ids[c3], abs_color_ids[c2])
                        elif dot_distances[i][j] < dot_distances[i][k]:
                            triplet = (abs_color_ids[c1], abs_color_ids[c2], abs_color_ids[c3])
                        else:
                            a, b = random.sample([c2, c3], 2)
                            triplet = (abs_color_ids[c1], abs_color_ids[a], abs_color_ids[b])

                        if triplet not in triplets:
                            triplets.append(triplet)
        return triplets
