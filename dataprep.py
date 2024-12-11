from mypytable import MyPyTable
import pandas as pd
import random
import geopandas as gpd

table1 = MyPyTable().load_from_file('iata-icao.csv')
table1.remove_rows_with_missing_values()

def filter_icao_values(num_airports):
    table2 = MyPyTable()
    table2.column_names = table1.column_names
    for i, val in enumerate(table1.get_column('country_code')):
        if val == 'US':
            table2.data.append(table1.data[i])
    table2.data = random.sample(table2.data, num_airports)      
    table2.save_to_file('hundredairports.csv')

filter_icao_values(5) # Get 100 airports.

ufos = pd.read_excel('UFO_sightings_complete.xlsx').drop(columns=['shape', 'duration (seconds)', 'duration (hours/min)'\
                                                   , 'comments', 'date posted'])
ufos_us = ufos[ufos['country'] == 'us']
ufos_us.drop(columns=['country']) # We don't need this anymore.
dataset = ufos_us.sample(1000) # Get 10,000 random UFO sightings.
dataset.to_excel('ufos_us.xlsx', index=False)

dataset['datetime'] = pd.to_datetime(dataset['datetime']).dt.date
dataset.rename(columns={'datetime':'date'}, inplace=True)

weather = pd.read_csv('hundredairports.csv').drop(columns=['country_code', 'region_name', 'iata']) # Remove superfluous data.
print(weather)
joined = gpd.sjoin_nearest(gpd.GeoDataFrame(dataset.drop(columns=['Unnamed: 11']), geometry=gpd.points_from_xy(dataset['latitude'], dataset['longitude']))\
                                 , gpd.GeoDataFrame(weather, geometry=gpd.points_from_xy(weather['latitude'], weather['longitude']))).drop(columns=['index_right', 'geometry'])

joined.to_excel('ufos_airports.xlsx', index=False)