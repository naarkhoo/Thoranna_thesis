from bokeh.plotting import figure, show
from bokeh.layouts import gridplot
from bokeh.resources import CDN
from bokeh.palettes import Spectral11
from bokeh.models import (
    ColumnDataSource, HoverTool, PanTool, WheelZoomTool,
    Toolbar, ResetTool, DataTable, TableColumn, Div)
from bokeh.embed import file_html
from bokeh.models.widgets import Div

import plotly.express as px
import plotly.graph_objs as go
import plotly.offline as pyo
from plotly.offline import plot as plotly_plot

from matplotlib import cm

from sklearn.manifold import TSNE

from .visualizations.tste import stochastic_triplet_embedding

import pybloqs as p

import numpy as np
import pandas as pd

# Reading triplets from the text file
with open('triplets.txt', 'r') as f:
    triplets = []
    for line in f:
        triplet = tuple(map(int, line.strip().split()))
        triplets.append(triplet)

colors = cm.rainbow(np.linspace(0, 1, len(triplets)))
triplets_array = np.array(triplets)

# Perform t-SNE to reduce the dimensionality of the data
tsne = TSNE(n_components=2, random_state=42)
embedding_tsne = tsne.fit_transform(triplets_array)

embedding_tste = stochastic_triplet_embedding(triplets_array)

# Set up the data sources for the plots
source_tsne = ColumnDataSource(data=dict(
    x=embedding_tsne[:, 0],
    y=embedding_tsne[:, 1],
    color=colors,
    label=[str(t) for t in triplets]
))
source_tste = ColumnDataSource(data=dict(
    x=embedding_tste[:, 0],
    y=embedding_tste[:, 1],
    color=colors,
    label=[str(t) for t in triplets]
))

# Set up the figures
fig_tsne = figure(tools=[PanTool(), WheelZoomTool()], title='t-SNE', width=400, height=400)
fig_tste = figure(tools=[PanTool(), WheelZoomTool()], title='t-STE', width=400, height=400)
fig_df = figure(tools=[PanTool(), WheelZoomTool()], title='Pandas DataFrame', width=400, height=400)

# Add the hover tool to the figures
hover_tsne = HoverTool(
    tooltips=[("Triplet", "@label")]
)
fig_tsne.add_tools(hover_tsne)

hover_tste = HoverTool(
    tooltips=[("Triplet", "@label")]
)
fig_tste.add_tools(hover_tste)

df = pd.read_csv('experiment_analysis/data/wine_flavours.csv')
# Create the data source for the DataFrame plot
source_df = ColumnDataSource(df)
columns = [TableColumn(field=c, title=c) for c in df.columns]
data_table = DataTable(source=source_df, columns=columns, width=800, height=400)

# Add the scatter plots to the figures
fig_tsne.scatter('x', 'y', source=source_tsne, fill_color='color', size=8)
fig_tste.scatter('x', 'y', source=source_tste, fill_color='color', size=8)

# Perform t-SNE to reduce the dimensionality of the data
tsne = TSNE(n_components=3, random_state=42)
embedding_tsne = tsne.fit_transform(triplets_array)
embedding_tste = stochastic_triplet_embedding(triplets, n_dims=3)

# Arrange the plots in a grid and display
grid = gridplot([[fig_tsne, fig_tste, data_table]])
show(grid)
