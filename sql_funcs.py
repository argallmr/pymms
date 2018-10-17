import sqlite3
import pandas as pd
from data_export_sql import data_export as data_download

def checkTableExists(table_name):
    check_table = "SELECT name FROM sqlite_master WHERE type='table' AND name='{}'".format(table_name)
    c.execute(check_table)
    result = c.fetchone()[0]
    if result: return True;
    else: return False;

def getRowVals(rownum, data, spacecraft):
    rowvals = "'" + str(data.loc[[rownum]].values.tolist()[0][0]) + "',"
    rowvals += "'" + spacecraft + "', "
    rowvals += ', '.join([ str(f) for f in data.loc[[rownum]].values.tolist()[0][1:]]);
    return rowvals

def createTable(file_path, table_name):
    data = pd.read_csv(file_path);
    columns = list(data)
    
    # Check if table exists in DB already
    if checkTableExists(table_name):
        drop_table = 'DROP TABLE mms1'
        c.execute(drop_table)
    
    # Create a new table with columns from the dataframe  
    create_table = 'CREATE TABLE ' +  table_name + ' ('
    
    # Create string containing column names and types
    colnames = 'Time TIMESTAMP, Spacecraft STRING , '
    for colname in columns[1:len(columns)-1]:
        colnames += colname.replace(' ', '_') + ' REAL, '
    colnames += columns[len(columns)-1].replace(' ', '_') + ' REAL'
    create_table += colnames + ');'
    
    c.execute(create_table)
    
    # Set indices
    set_primary_index = 'CREATE UNIQUE INDEX Time ON mms1(Time)'
    set_secondary_index = 'CREATE INDEX Spacecraft ON mms1(Spacecraft)'
    c.execute(set_primary_index)
    c.execute(set_secondary_index)

def insertRows(file_path, spacecraft):
    # Insert rows of dataframe to table
    data = pd.read_csv(file_path); # Should point to a spacecraft's data csv
    for rownum in range(1,len(data)):
        insert_row = ('INSERT INTO mms1 VALUES({});'.format(getRowVals(rownum, data, spacecraft)))
        c.execute(insert_row)

def run(spacecraft, level, start_date, end_date):
    # Download specified data
    data_download(spacecraft, level, start_date, end_date)
    
    # Create a table and insert rows from the associated .csv
    file_path = '/Home/colin/pymms/sql' + '_'.join([spacecraft, level, start_date, 'to']) + end_date + '.csv'   
    createTable(file_path, 'mms1')
    insertRows(file_path, 'mms1') 
    
    connection.commit()
    connection.close()
    
# Open connection to SQLite DB
sqlite_file = '/home/colin/pymms/sql/data.db'
connection = sqlite3.connect(sqlite_file)
c = connection.cursor()
    
run('mms1', 'l2', '2015-12-06', '2015-12-06T23:59:59')