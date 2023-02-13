from sklearn.metrics.pairwise import euclidean_distances
import random

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

class TripletGenerator:
    def __init__(self):
        pass

    def generate_triplets(self, centers, round_id):
        dot_distances = euclidean_distances(list(centers.values()), list(centers.values()))
        triplets = []
        for i, c1 in enumerate(centers.keys()):
            for j, c2 in enumerate(centers.keys()):
                for k, c3 in enumerate(centers.keys()):
                    if not (c1 == c2 or c1 == c3 or c2 == c3):
                        if (dot_distances[i][j] > dot_distances[i][k]):
                            triplet = [
                                abs_color_ids[round_id][c1],
                                abs_color_ids[round_id][c3],
                                abs_color_ids[round_id][c2]
                            ]
                        elif (dot_distances[i][j] < dot_distances[i][k]):
                            triplet = [
                                abs_color_ids[round_id][c1],
                                abs_color_ids[round_id][c2],
                                abs_color_ids[round_id][c3]
                            ]
                        elif (dot_distances[i][j] == dot_distances[i][k]):
                            j = random.choice([c2, c3])
                            if j == c2:
                                k = abs_color_ids[round_id][c3]
                            else:
                                j = abs_color_ids[round_id][c2]
                            triplet = [
                                abs_color_ids[round_id][c1], j, k
                            ]
                        if triplet not in triplets and not (dot_distances[i][j] == dot_distances[i][k]):
                            triplets.append(triplet)
        return triplets

    # def generate_random_triplets(self, centers):
    #     triplets = []
    #     for i, c1 in enumerate(centers.keys()):
    #         for j, c2 in enumerate(centers.keys()):
    #             for k, c3 in enumerate(centers.keys()):
    #                 if not (c1 == c2 or c1 == c3 or c2 == c3):
    #                     j = random.choice([c2, c3])
    #                     if j == c2:
    #                         k = c3
    #                     else:
    #                         j = c2
    #                     triplet = [abs_color_ids[round_id][c1], j, k]
    #                     if triplet not in triplets:
    #                         triplets.append(triplet)
    #     return triplets