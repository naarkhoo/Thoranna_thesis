import json
import math
from itertools import combinations
from collections import Counter
from scipy.stats import chi2_contingency
import numpy as np
from scipy.stats import binom
import random
import numpy as np
import matplotlib.pyplot as plt

def coin_bias_test(triplet_counter):
    observed_frequencies = []

    for triplet, frequency in triplet_counter.items():
        opposite_triplet = (triplet[0], triplet[2], triplet[1])
        opposite_frequency = triplet_counter[opposite_triplet]
        observed_frequencies.append([frequency, opposite_frequency])

    chi2, p_value, _, _ = chi2_contingency(observed_frequencies, correction=False)

    return p_value

def euclidean_distance(point1, point2):
    print(point1)
    return math.sqrt((int(point1[0]) - int(point2[0])) ** 2 + (int(point1[1]) - int(point2[1])) ** 2)

def process_experiment_round(experiment_round_data):
    
    distances = {
        (id1, id2): euclidean_distance(coords1, coords2)
        for (id1, [coords1, _]), (id2, [coords2, _]) in combinations(experiment_round_data.items(), 2)
    }

    triplets = [
        (id1, id2, id3)
        for (id1, id2), dist_ij in distances.items()
        for (id3, _), dist_ik in distances.items()
        if id1 != id3 and dist_ij < dist_ik and id2 != id3
    ]

    return triplets

def process_data(data):
    all_triplets = []

    for experiment in data:
        generated_data = experiment["generated_data"]

        for experiment_key, experiment_data in generated_data.items():
            for round_key, round_data in experiment_data.items():
                for exp_num, experiment_round_data in round_data.items():
                    print(experiment_round_data)
                    triplets = process_experiment_round(experiment_round_data)
                    all_triplets.extend(triplets)

    return all_triplets

def perform_chi_square_test(triplet_freq, opposite_triplet_freq):
    total_freq = triplet_freq + opposite_triplet_freq
    observed = np.array([triplet_freq, opposite_triplet_freq])
    expected = np.array([total_freq / 2, total_freq / 2])
    chi2, p_value = chi2_contingency([observed, expected], correction=False)[:2]
    return p_value

def cohen_d(triplet_freq, opposite_triplet_freq):
    total_freq = triplet_freq + opposite_triplet_freq
    pooled_std_dev = np.sqrt(((total_freq - 1) * (triplet_freq - opposite_triplet_freq) ** 2) / total_freq)
    if pooled_std_dev == 0:
        return 0
    return (triplet_freq - opposite_triplet_freq) / pooled_std_dev


if __name__ == "__main__":
    all_triplets = []

    with open('data_new1.json', 'r') as json_file:
        data = json.load(json_file)

    generated_data = data["generated_data"]

    for experiment_key, experiment_data in generated_data.items():
        for round_key, round_data in experiment_data.items():
            for exp_num, experiment_round_data in round_data.items():
                triplets = process_experiment_round(experiment_round_data)
                all_triplets.extend(triplets)

    with open('all_triplets4.json', 'w') as json_file:
        json.dump(all_triplets, json_file)

    triplet_counter = Counter(all_triplets)
    sorted_triplets = sorted(triplet_counter.items(), key=lambda x: x[1], reverse=True)
    top_n_triplets = 10
    print(f"Top {top_n_triplets} most common triplets:")
    for i, (triplet, frequency) in enumerate(sorted_triplets[:top_n_triplets], 1):
        opposite_triplet = (triplet[0], triplet[2], triplet[1])
        opposite_frequency = triplet_counter[opposite_triplet]
        p_value = perform_chi_square_test(frequency, opposite_frequency)
        effect_size = cohen_d(frequency, opposite_frequency)
        effect = "Yes" if effect_size >= 0.2 else "No"
        significant = "Yes" if p_value < 0.05 else "No"
        print(f"{i}. Triplet: {triplet}, Frequency: {frequency}, Opposite Triplet: {opposite_triplet}, Opposite Frequency: {opposite_frequency}, p-value: {p_value:.4f}, Statistically Significant: {significant}, Effect: {effect}")

    p_value = coin_bias_test(triplet_counter)
    print(f"The p-value for the coin bias test is: {p_value:.4f}")

    significant_triplets_count = 0
    non_significant_triplets_count = 0
    significant_triplets = []
    p_vals = []

    for i, (triplet, frequency) in enumerate(sorted_triplets, 1):
        opposite_triplet = (triplet[0], triplet[2], triplet[1])
        opposite_frequency = triplet_counter[opposite_triplet]
        p_value = perform_chi_square_test(frequency, opposite_frequency)
        p_vals.append(p_value)
        effect_size = cohen_d(frequency, opposite_frequency)
        effect = "Yes" if effect_size >= 0.2 else "No"
        significant = "Yes" if p_value < 0.25 else "No"

        if significant == "Yes":
            significant_triplets_count += 1
            significant_triplets.append(triplet)
        else:
            non_significant_triplets_count += 1
    
    with open('significant_triplets.json', 'w') as json_file:
        json.dump(significant_triplets, json_file)
    
    # Create a histogram of the p-values
    plt.hist(p_vals, bins=np.linspace(0, 1, 21), alpha=0.75, color='blue', edgecolor='black')

    # Add labels and a title
    plt.xlabel('P-value')
    plt.ylabel('Frequency')
    plt.title('Histogram of P-values')

    # Display the plot
    plt.show()

    print(f"Number of significant triplets: {significant_triplets_count}")
    print(f"Number of non-significant triplets: {non_significant_triplets_count}")
    print("Ratio of significant over not significant: ", significant_triplets_count / non_significant_triplets_count)

    if p_value < 0.05:
        print("The coin is biased (the frequency of triplets and opposite triplets is significantly different).")
    else:
        print("The coin is not biased (the frequency of triplets and opposite triplets is not significantly different).")
