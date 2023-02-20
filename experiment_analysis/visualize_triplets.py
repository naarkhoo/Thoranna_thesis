#!/usr/bin/env python3
from bokeh.plotting import figure, show
from bokeh.layouts import gridplot
from bokeh.resources import CDN
from bokeh.palettes import Spectral11
from bokeh.models import (
    ColumnDataSource, HoverTool, PanTool, WheelZoomTool,
    Toolbar, ResetTool, DataTable, TableColumn, Div, TabPanel,
    Tabs, CategoricalColorMapper, LinearColorMapper
)
from bokeh.embed import file_html
from bokeh.models.widgets import Div

import plotly.express as px
import plotly.graph_objs as go
import plotly.offline as pyo
from  plotly.offline import plot as plotly_plot

from matplotlib import cm
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE, MDS
from .visualizations.tste_theano.tste import tste as tste
from .snack.datasets.mnist import MNIST2KDataset, TripletMNIST2K

import pybloqs as p

import numpy as np
import pandas as pd
import scipy.io

# Reading triplets from the text file
with open('triplets.txt', 'r') as f:
    triplets = []
    for line in f:
        triplet = tuple(map(int, line.strip().split()))
        triplets.append(triplet)

def make_grid(triplets, split = None, labels = None):
    triplets_array = np.array(triplets)
    # make triplets 0-indexed
    triplets_array -= 1

    if split is not None:
        triplets_array = triplets_array[split[0]:split[1]]

    embedding_tste = tste(triplets=triplets_array, lamb=0, no_dims=2, alpha=1, use_log=False)
    # Like they do in SNaCK
    triplets_array_reversed = triplets_array[:, [0, 2, 1]]
    embedding_tste_reversed = tste(triplets=triplets_array_reversed, lamb=0, no_dims=2, alpha=1, use_log=False)

    if labels is not None:
        source_tste = ColumnDataSource(data=dict(
            x=embedding_tste[:, 0],
            y=embedding_tste[:, 1],
            label=[str(t) for t in embedding_tste]
        ))
        fig_tste = figure(tools=[PanTool(), WheelZoomTool()], title='t-STE', width=600, height=600)
        hover_tste = HoverTool(tooltips=[("Data point: ", "@label")])
        fig_tste.add_tools(hover_tste)
        fig_tste.scatter('x', 'y', source=source_tste, size=10)

        source_tste_rev = ColumnDataSource(data=dict(
            x=embedding_tste_reversed[:, 0],
            y=embedding_tste_reversed[:, 1],
            label=[str(t) for t in embedding_tste_reversed]
        ))
        fig_tste_rev = figure(tools=[PanTool(), WheelZoomTool()], title='t-STE (like they do in SNaCK)', width=600, height=600)
        hover_tste_rev = HoverTool(tooltips=[("Data point: ", "@label")])
        fig_tste_rev.add_tools(hover_tste_rev)
        fig_tste_rev.scatter('x', 'y', source=source_tste_rev, size=10)
    else:
        source_tste = ColumnDataSource(data=dict(
            x=embedding_tste[:, 0],
            y=embedding_tste[:, 1],
            label=[str(t) for t in embedding_tste]
        ))
        fig_tste = figure(tools=[PanTool(), WheelZoomTool()], title='t-STE', width=600, height=600)
        hover_tste = HoverTool(tooltips=[("Data point: ", "@label")])
        fig_tste.add_tools(hover_tste)
        fig_tste.scatter('x', 'y', source=source_tste, size=10)

        source_tste_rev = ColumnDataSource(data=dict(
            x=embedding_tste_reversed[:, 0],
            y=embedding_tste_reversed[:, 1],
            label=[str(t) for t in embedding_tste_reversed]
        ))
        fig_tste_rev = figure(tools=[PanTool(), WheelZoomTool()], title='t-STE (like they do in SNaCK)', width=600, height=600)
        hover_tste_rev = HoverTool(tooltips=[("Data point: ", "@label")])
        fig_tste_rev.add_tools(hover_tste_rev)
        fig_tste_rev.scatter('x', 'y', source=source_tste_rev, size=10)

    # Set up the figures
    # fig_tste = figure(tools=[PanTool(), WheelZoomTool()], title='t-STE', width=600, height=600)
    fig_df = figure(tools=[PanTool(), WheelZoomTool()], title='Pandas DataFrame', width=800, height=600)

    # Arrange the plots in a grid and display
    grid = gridplot([[fig_tste, fig_tste_rev]])
    return grid

grid1 = make_grid(triplets=triplets, split=None, labels=None)
grid2 = make_grid(triplets=triplets, split=[0, 1741], labels=None)
grid3 = make_grid(triplets=triplets, split=[1741, -1], labels=None)

msnitdata = MNIST2KDataset()
labels = list(msnitdata.labels)
triplet_train_dataset = TripletMNIST2K(msnitdata) # Returns triplets of images
triplets = np.array(triplet_train_dataset.test_triplets)
grid4 = make_grid(triplets=triplets, split=None, labels=labels)

df = pd.read_csv('experiment_analysis/data/wine_flavours.csv')
source_df = ColumnDataSource(df)
columns = [TableColumn(field=c, title=c) for c in df.columns]
data_table = DataTable(source=source_df, columns=columns, width=800, height=400)

tab1 = TabPanel(child=grid1, title="Experiments combined")
tab2 = TabPanel(child=grid2, title="Red wine (Round 1 DTU Tasting 25.11.2022)")
tab3 = TabPanel(child=grid3, title="White wine (Round 2 DTU Tasting 25.11.2022)")
tab4 = TabPanel(child=grid4, title="MNIST test")
tab5 = TabPanel(child=data_table, title="Wine data table")

tabs = Tabs(tabs=[tab1, tab2, tab3, tab4, tab5])
show(tabs)
