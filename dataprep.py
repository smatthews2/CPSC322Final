from mypytable import MyPyTable
import pandas as pd
import random

table1 = MyPyTable().load_from_file('iata-icao.csv')
table1.remove_rows_with_missing_values()

def filter_icao_values():
    table2 = MyPyTable()
    table2.column_names = table1.column_names
    for i, val in enumerate(table1.get_column('country_code')):
        if val == 'US':
            table2.data.append(table1.data[i])
    table2.data = random.sample(table2.data, 1)      
    table2.save_to_file('hundredairports.csv')

filter_icao_values()