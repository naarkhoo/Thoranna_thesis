from sklearn.metrics.pairwise import euclidean_distances
import random

class TripletGenerator:
    def __init__(self):
        pass

    def generate_triplets(self, centers):
        dot_distances = euclidean_distances(list(centers.values()), list(centers.values()))
        triplets = []
        for i, c1 in enumerate(centers.keys()):
            for j, c2 in enumerate(centers.keys()):
                for k, c3 in enumerate(centers.keys()):
                    if not (c1 == c2 or c1 == c3 or c2 == c3):
                        if (dot_distances[i][j] > dot_distances[i][k]):
                            triplet = (c1, c3, c2)
                        elif (dot_distances[i][j] < dot_distances[i][k]):
                            triplet = (c1, c2, c3)
                        elif (dot_distances[i][j] == dot_distances[i][k]):
                            j = random.choice([c2, c3])
                            if j == c2:
                                k = c3
                            else:
                                j = c2
                            triplet = (c1, j, k)
                        if triplet not in triplets and not (dot_distances[i][j] == dot_distances[i][k]):
                            triplets.append(triplet)
        return triplets

    def generate_random_triplets(self, centers):
        triplets = []
        for i, c1 in enumerate(centers.keys()):
            for j, c2 in enumerate(centers.keys()):
                for k, c3 in enumerate(centers.keys()):
                    if not (c1 == c2 or c1 == c3 or c2 == c3):
                        j = random.choice([c2, c3])
                        if j == c2:
                            k = c3
                        else:
                            j = c2
                        triplet = (c1, j, k)
                        if triplet not in triplets:
                            triplets.append(triplet)
        return triplets