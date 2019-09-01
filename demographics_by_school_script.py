# -*- coding: utf-8 -*-

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

"""Generates plots of racial edmographic composition by school for Seattle
"""

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

import pickle
from functools import reduce

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib import rc
rc('text', usetex = 'true')
from matplotlib import spines

import pandas as pd
import geopandas as gpd

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

SEATTLE_ASPECT_RATIO = 1.8329517346168405

SEATTLE_PROJ = "+proj=lcc +lat_1=47.5 +lat_2=48.73333333333333 +lat_0=47 +lon_0=-120.8333333333333 +x_0=500000.0000000002 +y_0=0 +datum=NAD83 +units=us-ft +no_defs "

BACKGROUND_COLOR = ( 0.385, ) * 3

demog_data_dict_path = '../pickle/nonempty_block_data_by_race_hispanic_correction.dict'

block_shapefiles_path = '../shapefiles/Census_Blocks_2010/Census_Blocks_2010.shp'

school_level_codes = ['HS', 'MS', 'ES']

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

if __name__ == '__main__':

  for school_level_code in school_level_codes:

    demog_data_dict = pickle.load(open(demog_data_dict_path, 'rb'))

    # convert dictionary of demographic data into dataframe
    #--------------------------------------------------------------------------#

    dfl = list()
    for k, v in demog_data_dict.items():
        df = pd.DataFrame.from_dict(v, orient = 'index', )
        df = df.reset_index(drop = False)
        df.columns = ['GEOID10', k]
        dfl.append(df)
    demog_data_df = reduce(lambda x, y: pd.merge(x, y, on = 'GEOID10'), dfl)

    # load block shapefiles into GeoPandas GeoDataFrame, merge demographic data
    # and block shapefile data into a single GeoDataFrame, project into Seattle-
    # specific coordinate reference system
    #--------------------------------------------------------------------------#

    block_gdf = gpd.read_file(block_shapefiles_path)
    block_gdf = block_gdf[['GEOID10', 'geometry']]
    block_gdf = block_gdf.merge(demog_data_df)
    block_gdf = block_gdf.to_crs(crs = SEATTLE_PROJ)

    # load High school GeoZone boundary shapefiles into GeoPandas GeoDataFrame
    # and project into Seattle-specific coordinate reference system
    #--------------------------------------------------------------------------#

    school_shapefiles_path = f'../shapefiles/SPS_AttendanceAreasAndSchools_Shapefiles_2017_2018/sps_attendance_area_{school_level_code}_2017_2018.shp'

    hs_gdf = gpd.read_file(school_shapefiles_path)
    hs_gdf = hs_gdf.to_crs(crs = SEATTLE_PROJ)
    hs_gdf = hs_gdf[[f'{school_level_code}_ZONE', 'geometry']]

    # Determine which school GeoZones contain which blocks
    #--------------------------------------------------------------------------#

    contains_shape = (hs_gdf.shape[0], block_gdf.shape[0])
    contains_arr = np.zeros(contains_shape, dtype = np.bool_)
    for i in range(contains_shape[0]):
      for j in range(contains_shape[1]):
        contains_arr[i, j] = hs_gdf.loc[i]['geometry'].contains(block_gdf.loc[j]['geometry'])

    np.save(f'../pickle/contains_{school_level_code}', contains_arr)

    contains_arr = np.load(f'../pickle/contains_{school_level_code}.npy' )

    # Compute mean radial composition for each race, for each school
    #--------------------------------------------------------------------------#

    school_list = list(hs_gdf[f'{school_level_code}_ZONE'])
    race_list = list(block_gdf)[3:]

    race_by_school = np.zeros((len(school_list), len(race_list)))
    for i, s in enumerate(school_list):
      race_by_school[i] = np.mean(block_gdf.loc[contains_arr[i]])[2:]

    rbs_df = pd.DataFrame(race_by_school)
    rbs_df.columns = race_list
    rbs_df.insert(0, 'school', school_list)

    rbs_sort = race_by_school[np.flipud(np.array(np.argsort(rbs_df['white'])))]

    # Make school names prettier
    #--------------------------------------------------------------------------#

    good_school_names = [i.replace(" Int'l", '') \
      .replace(f' {school_level_code}', '').strip( ) for i in school_list]
    good_school_names = np.array(good_school_names)
    good_school_names = good_school_names[np.flipud(np.array(np.argsort(rbs_df['white'])))]

    print( good_school_names )

    # Generate stacked bar plots
    #--------------------------------------------------------------------------#

    race_list_aug = race_list.copy()
    race_list_aug.append('other')
    race_list_aug = [i[0].upper() + i[1:] for i in race_list_aug]

    rbs_sort_aug = np.hstack([rbs_sort, 1 - np.sum(rbs_sort, axis = 1).reshape(-1,1)])

    ind = np.arange(len(school_list))
    width = 0.5

    fig, ax = plt.subplots(figsize = (6, 8))
    ax.barh(ind, rbs_sort_aug[:, 0], width, label = race_list_aug[0])
    for i in range(1, len(race_list_aug)):
        ax.barh(
            ind,
            rbs_sort_aug[:, i],
            width,
            left = np.sum(rbs_sort_aug[:, :i], axis = 1),
            label = race_list_aug[i])
    ax.legend(loc = 3, fontsize = 12, framealpha = 0)

    ax.set_xlim(0, 1)
    ax.set_yticks(np.arange(len(school_list)))
    ax.set_yticklabels(good_school_names, fontsize = 9.5)
    # ax.tick_params(axis='y', rotation=90)
    ax.set_xticks([])

    for child in ax.get_children():
      if isinstance(child, spines.Spine):
          child.set_color('#ffffff')

    plt.tight_layout()
    plt.savefig(f'../demographics_by_school_plots/demographics_by_school_geozone_{school_level_code}.svg', dpi = 200)

    # Generate map of blocks by GeoZone
    #--------------------------------------------------------------------------#

    minx, miny, maxx, maxy = block_gdf.total_bounds

    colors = get_cmap('tab20')(np.linspace(0, 1, len(school_list)))
    fig, ax = plt.subplots(1, figsize = (10, 10 * SEATTLE_ASPECT_RATIO))

    hs_gdf.plot(ax = ax, facecolor = 'w', edgecolor = 'k')

    for i, c in zip(range(contains_arr.shape[0]), colors):
      block_gdf.loc[contains_arr[i]].plot(ax = ax, color = c)

    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)

    ax.set_xticks([])
    ax.set_yticks([])

    plt.subplots_adjust(0,0,1,1)

    plt.savefig(f'../demographics_by_school_plots/blocks_by_geozone_{school_level_code}.svg')

    #--------------------------------------------------------------------------#

  #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#