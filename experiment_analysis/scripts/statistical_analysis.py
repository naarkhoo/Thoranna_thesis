import json
import numpy as np
from random import shuffle
from scipy.spatial import distance


def load_triplets():
    with open('all_triplets.json', 'r') as _file:
        triplets_arr = json.load(_file)

    triplets_arr = [[int(i), int(j), int(k)] for i, j, k in triplets_arr]
    return triplets_arr

def load_coordinates():
    with open('data.json', 'r') as _file:
        data = json.load(_file)

    coordinates = {}
    for experiment in data:
        for round_key in experiment['generated_data']:
            for round_number, wines in experiment['generated_data'][round_key].items():
                print("this is wines: ", wines)
                for wine_id, coord in wines.items():
                    coordinates[int(wine_id)] = coord
    print("this is coordinates: ", coordinates)
    return coordinates


def compute_metric(triplets, coordinates):
    metric = 0
    for i, j, k in triplets:
        if i in coordinates and j in coordinates and k in coordinates:
            ij_distance = distance.euclidean(coordinates[i], coordinates[j])
            ik_distance = distance.euclidean(coordinates[i], coordinates[k])
            metric += int(ij_distance < ik_distance)
        else:
            print(f"Warning: Wine IDs {i}, {j}, or {k} not found in coordinates. Skipping this triplet.")
    return metric


def permutation_test(triplets, coordinates, n_permutations=1000):
    original_metric = compute_metric(triplets, coordinates)
    permuted_metrics = []

    for _ in range(n_permutations):
        permuted_triplets = [shuffle([i, j, k]) for i, j, k in triplets]
        permuted_metric = compute_metric(permuted_triplets, coordinates)
        permuted_metrics.append(permuted_metric)

    return original_metric, np.array(permuted_metrics)

def main():
    triplets = load_triplets()
    coordinates = load_coordinates()

    original_metric, permuted_metrics = permutation_test(triplets, coordinates)

    print(f"Original metric: {original_metric}")
    print(f"Mean permuted metric: {np.mean(permuted_metrics)}")
    print(f"Standard deviation of permuted metric: {np.std(permuted_metrics)}")

    # Compute p-value
    p_value = np.sum(permuted_metrics >= original_metric) / len(permuted_metrics)
    print(f"P-value: {p_value}")

if __name__ == '__main__':
    main()
