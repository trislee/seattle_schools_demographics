# -*- coding: utf-8 -*-

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

"""Generates plots of racial demographic data for Seattle
"""

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

import os
from collections import Counter
import pickle
from functools import reduce

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

import pandas as pd
import geopandas as gpd
from census import Census

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

CENSUS_KEY = os.getenv('CENSUS_KEY')

CODE_DICT = {
  'P001001' : 'all',
  'P003002' : 'white',
  'P003005' : 'asian',
  'P003003' : 'black',
  'P004003' : 'hispanic',
  'P003004' : 'native',
  'P003006' : 'pacific',
}

HISPANIC_MIXED_DICT = {
  'P003002' : 'P005011', # white
  'P003003' : 'P005012', # black
  'P003004' : 'P005013', # native
  'P003005' : 'P005014', # asian',
  'P003006' : 'P005015', # pacific'
}

BACKGROUND_COLOR = ( 0.385, ) * 3

CMAP = get_cmap('viridis')

SEATTLE_ASPECT_RATIO = 1.8329517346168405

SEATTLE_PROJ = "+proj=lcc +lat_1=47.5 +lat_2=48.73333333333333 +lat_0=47 \
  +lon_0=-120.8333333333333 +x_0=500000.0000000002 +y_0=0 +datum=NAD83 \
  +units=us-ft +no_defs "

block_shapefile_path = '../shapefiles/Census_Blocks_2010/Census_Blocks_2010.shp'

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

def get_race_percentage(block_response, race_code):

  """Converts the Census API response for one block into a population
  percentage, for a given race code.

  Parameters
  ----------
  block_response : dict
    Result of Census API GET request, containing racial and total population
    data for a single census block
  race_code : string
    Census SF1 variable. {'P003002','P003003','P003004','P003005','P003006'}
    More information at https://api.census.gov/data/2010/dec/sf1/variables.html

  Returns
  -------
  float
    Population percentage for specified race, from specified block

  """

  # if total population is zero, make population percentage zero to avoid zero
  # division
  if float(block_response['P001001']) == 0:
    return 0
  else:
    if race_code in ['P001001', 'P004003']:
      return float(block_response[race_code]) / float(block_response['P001001'])
    else:
      # subtract portion of population that listed hispanic/latino origin
      mixed_hispanic_code = HISPANIC_MIXED_DICT[race_code]
      return float(block_response[race_code] - block_response[mixed_hispanic_code] ) / float(block_response['P001001'])

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

def get_block_number_from_response(block_response):

  """Extracts block number from a block-level Census API GET response
  """

  block_number = block_response['state'] + \
              block_response['county'] + \
              block_response['tract'] + \
              block_response['block']

  return block_number

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

if __name__ == '__main__':

  # read in block shapefiles
  #----------------------------------------------------------------------------#

  blocks_gdf = gpd.read_file(block_shapefile_path)
  blocks_gdf = blocks_gdf[['GEOID10', 'geometry']]

  # extract tract numbers from GEOID10 codes.
  # GEOID10 codes have form <state><county><tract><block>, where <state> is the
  # 2 digit state FIPS code, <county> is the 3 digit county FIPS code, <tract>
  # is the 6 digit track FIPS code, and <block> is the 4 digit block FIPS code.

  # for example, 530330067001001 refers to Washington state (53), King County
  # (033), tract 006700, block 1001.
  #----------------------------------------------------------------------------#

  tracts = set([k[5:11] for k in blocks_gdf['GEOID10']])
  tracts = sorted(list(tracts))

  #----------------------------------------------------------------------------#

  # initializing Census client
  c = Census(CENSUS_KEY, year = 2010)

  tract_responses = dict()

  for i, tract in enumerate(tracts):

    # get census data from te SF1 file, from 2010
    tract_responses[tract] = c.sf1.get(
      tuple(CODE_DICT.keys()) + tuple(HISPANIC_MIXED_DICT.values()),
      {
        'for' : 'block:*',
        'in': f'state:53+county:033+tract:{tract}'
      }
    )

  # Convert Census API responses into dict of dicts for race data, where
  # highest dict level has race as key and block-based race data dict as value,
  # and block-based data dict has block number as key and race population
  # percentage as value, e.g.:
  # {'white' :
  #   { '530330067001001' : 0.65,
  #     '530330069001009' : 0.46,
  #   },
  # 'black' :
  #   { '530330067001001' : 0.18,
  #     '530330069001009' : 0.29,
  #   }
  # }
  #----------------------------------------------------------------------------#

  race_block_data_dict = dict()

  for race_code, race_name in CODE_DICT.items():

    block_data = dict()

    for tract_response in tract_responses.values():

      for block_response in tract_response:

        block_number = get_block_number_from_response(block_response)

        block_data[block_number] = get_race_percentage(
          block_response,
          race_code)

    race_block_data_dict[race_name] = block_data

  # convert race_block_data dict into dataframe
  #----------------------------------------------------------------------------#

  dfl = list()
  for k, v in race_block_data_dict.items():
      df = pd.DataFrame.from_dict(v, orient = 'index', )
      df = df.reset_index(drop = False)
      df.columns = ['GEOID10', k]
      dfl.append(df)
  race_block_data_df = reduce(lambda x, y: pd.merge(x, y, on = 'GEOID10'), dfl)

  race_block_data_df = race_block_data_df[race_block_data_df['all'] > 0]

  blocks_gdf = blocks_gdf.merge(race_block_data_df)
  blocks_gdf = blocks_gdf.to_crs(crs = SEATTLE_PROJ)

  # compute diversity using 'concentration index'/'trace of covariance matrix'
  # method from 'How to measure diversity when you must'
  # http://dx.doi.org/10.1037/a0027129
  #----------------------------------------------------------------------------#

  diversity_column = np.zeros_like(blocks_gdf['all'])
  for race in list(blocks_gdf)[3:]:
      diversity_column += blocks_gdf[race]**2
  diversity_column = 1 - diversity_column
  blocks_gdf['diversity'] = diversity_column

  blocks_gdf.to_pickle(
    path = '../pickle/nonempty_race_block_data_hispanic_correction.gdf')

  # getting bounds of shapefiles for plot
  #----------------------------------------------------------------------------#
  minx, miny, maxx, maxy = blocks_gdf.total_bounds
  ratio = np.abs((maxy - miny) / (maxx - minx))

  # plotting data for all races
  #----------------------------------------------------------------------------#

  for column in list(blocks_gdf)[2:]:

    fig, ax = plt.subplots(1, figsize = (10, 10 * ratio))

    blocks_gdf.plot(column = column, ax = ax, vmin = 0, vmax = 1)
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.axis('off')

    plt.subplots_adjust(0,0,1,1)
    plt.savefig(
      fname = f'../choropleths/{column}.svg',
      facecolor = BACKGROUND_COLOR)
    plt.close()

  #----------------------------------------------------------------------------#

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#