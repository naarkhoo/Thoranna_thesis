import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.offline as pyo
from sklearn.manifold import TSNE
from .visualizations.tste import stochastic_triplet_embedding

# Reading triplets from the text file
with open('triplets.txt', 'r') as f:
    triplets = []
    for line in f:
        triplet = tuple(map(int, line.strip().split()))
        triplets.append(triplet)

triplets_array = np.array(triplets)

# Perform t-SNE to reduce the dimensionality of the data
tsne = TSNE(n_components=3, random_state=42)
embedding_tsne = tsne.fit_transform(triplets_array)

embedding_tste = stochastic_triplet_embedding(triplets, n_dims=3)

# Set up the traces for the 3D scatter plot
trace_tsne = go.Scatter3d(
    x=embedding_tsne[:, 0],
    y=embedding_tsne[:, 1],
    z=embedding_tsne[:, 2],
    mode='markers',
    marker=dict(
        color=np.arange(len(triplets)),
        size=2,
        opacity=0.8,
        colorscale='Viridis'
    ),
    text=[f'Triplet: {t}' for t in triplets],
)

trace_tste = go.Scatter3d(
    x=embedding_tste[:, 0],
    y=embedding_tste[:, 1],
    z=embedding_tste[:, 2],
    mode='markers',
    marker=dict(
        color=np.arange(len(triplets)),
        size=2,
        opacity=0.8,
        colorscale='Viridis'
    ),
    text=[f'Triplet: {t}' for t in triplets],
)

# Plot the t-SNE plot
layout_tsne = go.Layout(title='Triplets t-SNE', scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'))
fig_tsne = go.Figure(data=[trace_tsne], layout=layout_tsne)
pyo.plot(fig_tsne)

# Plot the STE plot
# layout_tste = go.Layout(title='Triplets t-STE', scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'))
# fig_tste = go.Figure(data=[trace_tste], layout=layout_tste)
# pyo.plot(fig_tste)