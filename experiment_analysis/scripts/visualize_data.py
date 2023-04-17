#!/usr/bin/env python3
import json
import numpy as np
import pandas as pd

from bokeh.plotting import figure, show
from bokeh.layouts import gridplot
from bokeh.models import (
    ColumnDataSource, HoverTool, PanTool, WheelZoomTool,
    CategoricalColorMapper, TabPanel, Tabs,
)
from bokeh.palettes import Category20
from experiment_analysis.packages.tste_theano.tste import tste

WINE_TO_LOCATIONS = {-1: 'Unknown', 
                     0: 'France',
                     1: 'Italy',
                     2: 'Australia',
                     3: 'Italy',
                     4: 'Italy',
                     5: 'France',
                     6: 'Italy',
                     7: 'Italy',
                     8: 'Italy',
                     9: 'United States',
                     10: 'Italy',
                     11: 'Italy',
                     12: 'Australia',
                     13: 'Spain',
                     14: 'Italy',
                     15: 'Spain',
                     16: 'Italy',
                     17: 'Italy',
                     18: 'Italy',
                     19: 'Spain',
                     20: 'Spain',
                     21: 'Spain',
                     22: 'Portugal',
                     23: 'Australia',
                     24: 'Spain',
                     25: 'France',
                     26: 'France',
                     27: 'Argentina',
                     28: 'Australia',
                     29: 'Spain',
                     30: 'Argentina',
                     31: 'Italy',
                     32: 'France',
                     33: 'France',
                     34: 'Spain',
                     35: 'Italy',
                     36: 'Italy',
                     37: 'Italy',
                     38: 'Italy',
                     39: 'Italy',
                     40: 'Italy',
                     41: 'United States',
                     42: 'Portugal',
                     43: 'Portugal',
                     44: 'Italy',
                     45: 'Italy',
                     46: 'Italy',
                     47: 'Italy',
                     48: 'Italy',
                     49: 'Spain',
                     50: 'France',
                     51: 'Spain',
                     52: 'Italy',
                     53: 'France',
                     54: 'Italy',
                     55: 'Italy',
                     56: 'Italy',
                     57: 'Italy',
                     58: 'Italy',
                     59: 'Italy',
                     60: 'Spain',
                     61: 'Italy',
                     62: 'France',
                     63: 'Italy',
                     64: 'Argentina',
                     65: 'Australia',
                     66: 'Italy',
                     67: 'Ribena',
                     68: 'Spain',
                     69: 'Spain',
                     70: 'Italy',
                     71: 'Italy',
                     72: 'France',
                     73: 'Italy',
                     74: 'Italy',
                     75: 'Italy',
                     76: 'Spain',
                     77: 'Italy',
                     78: 'Italy',
                     79: 'France',
                     80: 'France',
                     81: 'Italy',
                     82: 'Spain',
                     83: 'Spain',
                     84: 'France',
                     85: 'United States',
                     86: 'Spain',
                     87: 'Portugal',
                     88: 'France',
                     89: 'Spain',
                     90: 'Spain',
                     91: 'Spain',
                     92: 'Italy',
                     93: 'United States',
                     94: 'Italy',
                     95: 'France',
                     96: 'Fance',
                     97: 'Italy',
                     98: 'France',
                     99: 'Spain',
                     100: 'France',
                     101: 'France',
                     102: 'Italy',
                     103: 'Italy',
                     104: 'Italy',
                     105: 'Italy',
                     106: 'Australia',
                     107: 'France',
                     108: 'Spain',
                     109: 'France',
                     110: 'Italy',
                     111: 'Italy',
                     112: 'Italy',
                     113: 'Italy',
                     114: 'Italy',
                     115: 'France',
                     116: 'Spain'}

WINE_STYLE = {0: 'Bordeaux Saint-Émilion',
              1: 'Northern Italy Red',
              2: 'Australian Shiraz',
              3: 'Southern Italy Primitivo',
              4: 'Tuscan Red', 
              5: 'Bordeaux Saint-Émilion',
              6: 'Italian Brunello',
              7: 'Italian Barbera',
              8: 'Southern Italy Red',
              9: 'Californian Pinot Noir',
              10: 'Italian Barbera',
              11: 'Tuscan Red',
              12: 'Australian Shiraz',
              13: 'Spanish Red',
              14: 'Italian Chianti',
              15: 'Spanish Tempranillo',
              16: 'Southern Italy Red',
              17: 'Southern Italy Red',
              18: 'Southern Italy Red',
              19: 'Spanish Syrah',
              20: 'Spanish Ribera Del Duero Red',
              21: 'Spanish Rioja Red',
              22: 'Portuguese Alentejo Red',
              23: 'Australian Shiraz',
              24: 'Spanish Red',
              25: 'Languedoc-Roussillon Red',
              26: 'Bordeaux Médoc',
              27: 'Argentinian Malbec',
              28: 'Australian Shiraz',
              29: 'Spanish Red', 
              30: 'Argentinian Malbec Red Blend',
              31: 'Southern Italy Primitivo',
              32: 'Southern Rhône Red',
              33: 'Bordeaux Saint-Émilion',
              34: 'Spanish Priorat Red',
              35: 'Tuscan Red',
              36: "Italian Montepulciano d'Abruzzo",
              37: 'Tuscan Red',
              38: 'Tuscan Red',
              39: 'Southern Italy Red',
              40: 'Central Italy Red',
              41: 'Californian Pinot Noir',
              42: 'Southern Portugal Red',
              43: 'Portuguese Douro Red',
              44: 'Tuscan Red',
              45: 'Tuscan Red',
              46: 'Central Italy Red',
              47: 'Southern Italy Red',
              48: 'Italian Amarone',
              49: 'Spanish Red',
              50: 'Bordeaux Saint-Émilion',
              51: 'Spanish Ribera Del Duero Red',
              52: 'Northern Italy Red',
              53: 'Burgundy Côte de Beaune Red',
              54: 'Northern Italy Red',
              55: 'Italian Amarone',
              56: 'Southern Italy Red',
              57: 'Central Italy Red',
              58: 'Southern Italy Red',
              59: 'Southern Italy Primitivo',
              60: 'Spanish Rioja Red',
              61: 'Italian Red',
              62: 'Bordeaux Saint-Émilion',
              63: 'Italian Amarone',
              64: 'Argentinian Cabernet Sauvignon - Malbec',
              65: 'Australian Shiraz',
              66: 'Southern Italy Primitivo',
              67: 'Ribena',
              68: 'Spanish Tempranillo',
              69: 'Spanish Ribera Del Duero Red',
              70: 'Southern Italy Primitivo',
              71: 'Central Italy Red',
              72: 'Bordeaux Pauillac',
              73: 'Central Italy Red',
              74: 'Italian Nebbiolo',
              75: 'Tuscan Red',
              76: 'Spanish Toro Red',
              77: 'Italian Chianti',
              78: 'Southern Italy Primitivo',
              79: 'Unknown1',
              80: 'Bordeaux Saint-Émilion',
              81: 'Southern Italy Red',
              82: 'Spanish Grenache',
              83: 'Spanish Rioja Red',
              84: 'Southern Rhône Red',
              85: 'Californian Zinfandel',
              86: 'Spanish Monastrell',
              87: 'Southern Portugal Red',
              88: 'South African Bordeaux Blend',
              89: 'Bordeaux Red',
              90: 'Spanish Ribera Del Duero Red',
              91: 'Spanish Ribera Del Duero Red',
              92: 'Spanish Red',
              93: 'Northern Italy Red',
              94: 'Californian Zinfandel',
              95: 'Southern Italy Primitivo'}

WINE_REGION = {0: 'Saint-Émilion Grand Cru',
               1: 'Italy / Northern Italy / Veneto',
               2: 'Australia / South Australia / Barossa / Barossa Valley',
               3: 'Italy / Southern Italy / Puglia',
               4: 'Italy / Central Italy / Toscana',
               5: 'France / Bordeaux / Libournais / Saint-Émilion / Montagne-Saint-Émilion',
               6: 'Italy / Central Italy / Toscana / Brunello di Montalcino',
               7: 'Italy / Northern Italy / Piemonte',
               8: 'Italy / Southern Italy / Molise / Biferno',
               9: 'United States / California / North Coast / Napa County / Napa Valley',
               10: "Italy / Northern Italy / Piemonte / Barbera d'Asti",
               11: "Italy / Central Italy / Toscana",
               12: 'Australia / South Australia / Barossa',
               13: 'Spain / Vino de España',
               14: 'Italy / Central Italy / Toscana / Chianti',
               15: 'Spain / Castilla y León / Cigales',
               16: 'Italy / Southern Italy / Campania / Irpinia / Irpinia Campi Taurasini',
               17: 'Italy / Southern Italy / Puglia / Salento',
               18: 'Italy / Southern Italy / Campania / Taurasi',
               19: 'Spain / Vino de España',
               20: 'Spain / Castilla y León / Ribera del Duero',
               21: 'Spain / Rioja',
               22: 'Portugal / Alentejano / Alentejo',
               23: 'Australia / South Australia / Barossa / Barossa Valley',
               24: 'Spain / Murcia / Jumilla',
               25: 'France / Languedoc-Roussillon / Languedoc / Terrasses du Larzac',
               26: 'France / Bordeaux / Médoc / Haut-Médoc',
               27: 'Argentina / Mendoza / Uco Valley',
               28: 'Australia / South Australia / Barossa / Barossa Valley',
               29: 'Spain / Vino de España',
               30: 'Argentina / Mendoza / Uco Valley',
               31: 'Italy / Southern Italy / Puglia / Primitivo di Manduria',
               32: 'France / Rhone Valley / Southern Rhône / Cairanne',
               33: 'France / Bordeaux / Libournais / Saint-Émilion / Montagne-Saint-Émilion',
               34: 'Spain / Catalunya / Priorat',
               35: 'Italy / Central Italy / Toscana',
               36: "Italy / Central Italy / Abruzzo / Montepulciano d'Abruzzo",
               37: 'Italy / Central Italy / Toscana',
               38: 'Italy / Central Italy / Toscana',
               39: 'Italy / Southern Italy / Basilicata / Aglianico del Vulture',
               40: 'Italy / Central Italy / Abruzzo',
               41: 'United States / California / Central Coast / Monterey County',
               42: 'Portugal / Península de Setúbal',
               43: 'Portugal / Northern Portugal / Duriense / Douro',
               44: 'Italy / Central Italy / Toscana',
               45: 'Italy / Central Italy / Toscana / Costa Toscana',
               46: 'Italy / Central Italy / Abruzzo',
               47: 'Italy / Southern Italy / Puglia',
               48: 'Italy / Northern Italy / Veneto / Valpolicella / Amarone della Valpolicella / Amarone della Valpolicella Classico',
               49: 'Spain / Islas Baleares / Mallorca',
               50: 'France / Bordeaux / Libournais / Saint-Émilion / Saint-Émilion Grand Cru',
               51: 'Spain / Castilla y León / Ribera del Duero',
               52: 'Italy / Northern Italy / Veneto',
               53: 'France / Bourgogne / Côte de Beaune / Santenay',
               54: 'Italy / Northern Italy / Emilia-Romagna',
               55: 'Italy / Northern Italy / Veneto / Valpolicella / Amarone della Valpolicella',
               56: 'Italy / Southern Italy / Puglia',
               57: 'Italy / Central Italy / Umbria',
               58: 'Italy / Southern Italy / Terre Siciliane',
               59: 'Italy / Southern Italy / Puglia / Salento',
               60: 'Spain / Rioja',
               61: "Italy / Vino d'Italia",
               62: 'France / Bordeaux / Libournais / Saint-Émilion / Saint-Émilion Grand Cru',
               63: 'Italy / Northern Italy / Veneto / Valpolicella / Amarone della Valpolicella / Amarone della Valpolicella Classico',
               64: 'Argentina / Salta',
               65: 'Australia / South Australia',
               66: 'Italy / Southern Italy / Puglia',
               67: 'Ribena',
               68: 'Spain / Castilla y León / Sardón de Duero',
               69: 'Spain / Castilla y León / Ribera del Duero',
               70: 'Italy / Southern Italy / Puglia / Primitivo di Manduria',
               71: 'Italy / Central Italy / Marche',
               72: 'France / Bordeaux / Médoc / Pauillac',
               73: 'Italy / Central Italy / Lazio',
               74: 'Italy / Northern Italy / Piemonte / Langhe',
               75: 'Italy / Central Italy / Toscana',
               76: 'Spain / Castilla y León / Toro',
               77: 'Italy / Central Italy / Toscana / Chianti / Chianti Classico',
               78: 'Italy / Southern Italy / Puglia',
               79: 'France / Vin de France',
               80: 'France / Bordeaux / Libournais / Saint-Émilion / Saint-Émilion Grand Cru',
               81: 'Italy / Southern Italy / Campania / Irpinia / Irpinia Campi Taurasini',
               82: 'Spain / Aragón / Cariñena',
               83: 'Spain / Rioja',
               84: 'France / Rhone Valley / Southern Rhône / Côtes-du-Rhône',
               85: 'United States / California / Central Valley / Lodi',
               86: 'Spain / Vino de España',
               87: 'Portugal / Península de Setúbal',
               88: 'South Africa / Western Cape / Coastal Region / Stellenbosch',
               89: 'France / Bordeaux',
               90: 'Spain / Castilla y León / Ribera del Duero',
               91: 'Spain / Castilla y León / Ribera del Duero',
               92: 'Spain / Valencia',
               93: 'Italy / Northern Italy / Veneto',
               94: 'United States / California',
               95: 'Italy / Southern Italy / Puglia / Salento'}

WINE_YEAR = {0: '2015',
             1: '2015',
             2: '2018',
             3: '2020',
             4: '2017',
             5: '2019',
             6: '2019',
             7: '2019',
             8: '2015',
             9: '2018',
             10: '2018',
             11: '2016',
             12: '2019',
             13: '2019',
             14: '2016',
             15: '2014',
             16: 'Unknown1',
             17: 'Unknown2',
             18: '2015',
             19: '2019',
             20: 'Unknown3',
             21: '2017',
             22: '2020',
             23: '2019',
             24: '2018',
             25: '2018',
             26: '2017',
             27: '2017',
             28: '2016',
             29: 'Unknown4',
             30: '2013',
             31: '2019', 
             32: '2019',
             33: '2015',
             34: '2018',
             35: '2018',
             36: '2018',
             37: '2018',
             38: '2016',
             39: '2020',
             40: '2018',
             41: '2019',
             42 :'2012',
             43: '2019',
             44: '2017',
             45: '2015',
             46: '2021',
             47: '2020',
             48: '2015',
             49: '2018',
             50: '2018',
             51: '2016',
             52: '2018',
             53: '2014',
             54: 'Unknown5',
             55: '2018',
             56: '2020',
             57: '2012',
             58: '2018',
             59: '2020',
             60: '2018',
             61: 'Unknown6',
             62: '2015',
             63: '2015',
             64: '2015',
             65: '2018',
             66: '2017',
             67: 'Ribena',
             68: '2017',
             69: '2018',
             70: '2018',
             71: '2017',
             72: '2016',
             73: '2018',
             74: '2019',
             75: '2016',
             76: '2017',
             77: '2018',
             78: '2019',
             79: '2016',
             80: '2016',
             81: '2017',
             82: '2018',
             83: '2016',
             84: '2019',
             85: '2019',
             86: '2020',
             87: '2020',
             88: '2018',
             89: '2015',
             90: '2020',
             91: '2017',
             92: '2016',
             93: '2019',
             94: '2019',
             95: '2018'}

WINE_ALCOHOL_PERCENTAGE = {0: '14',
                           1: '13',
                           2: '14',
                           3: '14',
                           4: '14',
                           5: '12.5',
                           6: '14',
                           7: '13.5',
                           8: '14',
                           9: '13.5',
                           10: '14.5',
                           11: '13.5',
                           12: '14',
                           13: '14',
                           14: 'Unknown1',
                           15: '14',
                           16: 'Unknown2',
                           17: '13.5',
                           18: '14',
                           19: '13',
                           20: '14',
                           21: '12',
                           22: '14',
                           23: '15',
                           24: '14',
                           25: '13.5',
                           26: '13',
                           27: '14.8',
                           28: '14.5',
                           29: '14',
                           30: '14.5',
                           31: '14',
                           32: '14',
                           33: '14',
                           34: '14.5',
                           35: '14',
                           36: '13.5',
                           37: '13.5',
                           38: '14',
                           39: '13',
                           40: '14.5',
                           41: '13.5',
                           42: '14.5',
                           43: '13.5',
                           44: '13.5',
                           45: '14',
                           46: '14.5',
                           47: '14.5',
                           48: '16.5',
                           49: '15',
                           50: '13',
                           51: '14',
                           52: '13.5',
                           53: '13',
                           54: '18',
                           55: '15',
                           56: '14',
                           57: '13',
                           58: '14.5',
                           59: '14',
                           60: '14',
                           61: '14',
                           62: '14.5',
                           63: '15.5',
                           64: '16',
                           65: '14.5',
                           66: '14',
                           67: 'Ribena',
                           68: '14',
                           69: '14',
                           70: '14.5',
                           71: '14',
                           72: '14',
                           73: '15',
                           74: '14',
                           75: '14.5',
                           76: '15',
                           77: '14',
                           78: '15',
                           79: '14.5',
                           80: '14',
                           81: '15',
                           82: '14.5',
                           83: '13.5',
                           84: '13.5',
                           85: '14.5',
                           86: '15',
                           87: '14.5',
                           88: '13.5',
                           89: '13.5',
                           90: '14',
                           91: '13.5',
                           92: '13.5',
                           93: '14',
                           94: '14.5',
                           95: '14'}

WINE_GRAPE = {0: ['Merlot', 'Cabernet Sauvignon'],
              1: ['Montepulciano'],
              2: ['Shiraz/Syrah'],
              3: ['Primitivo'],
              4: ['Sangiovese'],
              5: ['Cabernet Franc', 'Merlot'],
              6: ['Sangiovese'],
              7: ['Barbera'],
              8: ['Montepulciano', 'Aglianico'],
              9: ['Pinot Noir'],
              10: ['Barbera'],
              11: ['Merlot', 'Cabernet Sauvignon', 'Petit Verdot'],
              12: ['Shiraz/Syrah'],
              13: ['Shiraz/Syrah', 'Tempranillo'],
              14: ['Sangiovese'],
              15: ['Tempranillo'],
              16: ['Aglianico'],
              17: ['Negroamaro'],
              18: ['Aglianico'],
              19: ['Syrah/Shiraz'],
              20: ['Tempranillo'],
              21: ['Tempranillo', 'Mazuelo', 'Garnacha'],
              22: ['Alicante Bouschet'],
              23: ['Shiraz/Syrah'],
              24: ['Shiraz/Syrah', 'Petite Sirah', 'Monastrell'],
              25: ['Grenache'],
              26: ['Cabernet Sauvignon', 'Merlot'],
              27: ['Malbec'],
              28: ['Shiraz/Syrah', 'Tempranillo', 'Mourvedre', 'Grenache'],
              29: ['Shiraz/Syrah', 'Tempranillo'],
              30: ['Cabernet Sauvignon', 'Malbec', 'Shiraz/Syrah'],
              31: ['Primitivo'],
              32: ['Grenache', 'Mourvedre', 'Shiraz/Syrah'],
              33: ['Cabernet Sauvignon', 'Cabernet Franc', 'Merlot'],
              34: ['Garnacha', 'Cabernet Sauvignon', 'Merlot', 'Cariñena'],
              35: ['Merlot', 'Sangiovese'],
              36: ['Montepulciano'],
              37: ['Sangiovese'],
              38: ['Merlot', 'Sangiovese', 'Cabernet Sauvignon'],
              39: ['Aglianico'],
              40: ['Sangiovese'],
              41: ['Pinot Noir'],
              42: ['Shiraz/Syrah'],
              43: ['Tinta Roriz', 'Tinta Barroca', 'Touriga Franca', 'Touriga Nacional'],
              44: ['Sangiovese', 'Cabernet Sauvignon'],
              45: ['Merlot'],
              46: ['Montepulciano'],
              47: ['Negromaro'],
              48: ['Corvina', 'Rondinella', 'Corvinone'],
              49: ['Merlot', 'Cabernet Sauvignon', 'Callet', 'Manto Negro'],
              50: ['Merlot', 'Cabernet Sauvignon'],
              51: ['Tempranillo'],
              52: ['Corvina'],
              53: ['Pinot Noir'],
              54: ['Bonarda'],
              55: ['Corvina', 'Rondinella'],
              56: ['Merlot'],
              57: ['Shiraz/Syrah', 'Merlot', 'Sangiovese'],
              58: ['Cabernet Sauvignon', "Nero d'Avola"],
              59: ['Primitivo'],
              60: ['Tempranillo'],
              61: ['Merlot', "Nero d'Avola", 'MontePulciano', 'Primitivo'],
              62: ['Merlot', 'Cabernet Franc', 'Cabernet Sauvignon'],
              63: ['Corvinone', 'Corvina', 'Rondinella', 'Croatina'],
              64: ['Malbec'],
              65: ['Shiraz/Syrah'],
              66: ['Primitivo'],
              67: ['Ribena (Solbær)'],
              68: ['Tempranillo'],
              69: ['Tempranillo'],
              70: ['Primitivo'],
              71: ['Sangiovese', 'Montepulciano', 'Cabernet Sauvignon'],
              72: ['Cabernet Sauvignon', 'Merlot', 'Cabernet Franc', 'Petit Verdot'],
              73: ['Shiraz/Syrah'],
              74: ['Nebbiolo'],
              75: ['Merlot'],
              76: ['Tinta de toro'],
              77: ['Sangiovese'],
              78: ['Primitivo'],
              79: ['Merlot'],
              80: ['Merlot', 'Cabernet Franc', 'Cabernet Sauvignon'],
              81: ['Aglianico'],
              82: ['Garnacha'],
              83: ['Tempranillo', 'Mazuelo', 'Graciano'],
              84: ['Shiraz/Syrah', 'Grenache'],
              85: ['Zinfandel'],
              86: ['Monastrell'],
              87: ['Touriga Nacional', 'Castelao', 'Alicante Bouschet'],
              88: ['Cabernet Franc', 'Merlot', 'Cabernet Sauvignon'],
              89: ['Cabernet Sauvignon', 'Merlot'],
              90: ['Tempranillo'],
              91: ['Tempranillo'],
              92: ['Tempranillo', 'Monastrell'],
              93: ['Cabernet Sauvignon', 'Merlot'],
              94: ['Zinfandel'],
              95: ['Primitivo']}


def load_triplets():
    '''
    Function loads calculated triplets
    '''
    with open('all_triplets1.json', 'r') as _file:
        triplets_arr = json.load(_file)
    
    # Convert wine IDs to integers
    triplets_arr = [[int(i), int(j), int(k)] for i, j, k in triplets_arr]
    
    return triplets_arr

def make_grid(triplets_lis, wine_to_locations, labels=None):
    '''
    Function makes bokeh visualizations
    '''
    triplets_array = np.array(triplets_lis)

    embedding_tste = tste(
        triplets=triplets_array,
        lamb=0,
        no_dims=2,
        alpha=1,
        use_log=False
    )
    print(embedding_tste)
    print("n points in tste embedding: ", len(embedding_tste))

    # Get unique IDs from the input triplets
    unique_ids = list(wine_to_locations.keys())

    # Create a dictionary to map the unique_ids to the corresponding embedding_tste
    id_to_embedding = {id: embedding_tste[idx] for idx, id in enumerate(unique_ids)}
    print(len(id_to_embedding.keys()))

    # Get the countries for each unique ID
    countries = [wine_to_locations[id] for id in unique_ids]

    source_tste = ColumnDataSource(data=dict(
        x=embedding_tste[:, 0],
        y=embedding_tste[:, 1],
        id=unique_ids,
        label=[str(id) for id in unique_ids],
        country=countries
    ))

    # Set up the color mapper based on countries
    unique_countries = list(set(wine_to_locations.values()))
    color_mapper = CategoricalColorMapper(factors=unique_countries, palette=Category20[len(unique_countries)])

    fig_tste = figure(tools=[PanTool(), WheelZoomTool()], title='t-STE', width=600, height=600)
    hover_tste = HoverTool(tooltips=[("Data point: ", "@label"), ("ID: ", "@id"), ("Country: ", "@country")])
    fig_tste.add_tools(hover_tste)
    fig_tste.scatter('x', 'y', source=source_tste, size=10, color={'field': 'country', 'transform': color_mapper})

    # Arrange the plots in a grid and display
    grid = gridplot([[fig_tste]])
    return grid, id_to_embedding


if __name__ == "__main__":
    # Load data 
    triplets = load_triplets()

    # Load triplet visualizations
    grid1, _ = make_grid(triplets_lis=triplets, wine_to_locations=WINE_TO_LOCATIONS, labels=None)

    # Make tab panels
    tab1 = TabPanel(child=grid1, title="Experiments combined")

    tabs = Tabs(tabs=[tab1])
    show(tabs)
