## Installation

For development purposes, install the package using
```bash
$ python3 setup.py develop --user
```
This installation will reflect any changes made in the pymms development directory without the need to reinstall the package every single time.

## Download

Here are the instructions to download level 2 data for the MMS spacecraft. The script should be in the pymms repo once Matt accepts my pull request. It uses what we had written before, but, instead of saving all of the data to CSV files, it saves it to an SQLlite DB.

The script server_funcs_consolidated.py in the repo needs to be set up before use. 
Line 773 is the path to where the the SQL .db file will be created, so that needs to be set to your preference. 
Line 777 is the spacecraft whose data will be downloaded. Right now it’s ‘mms1’, but can be set to ‘mms2’, ‘mms3’, or ‘mms4’. 
Line 779 is the start date of the data that is to be downloaded, so that should be set to September of 2015 (2015-09-01). 
Line 780 is the end date, so that should be set to May of 2016 (2016-05-31T23:59:59). Finally, line 781 is the path to where the raw data will be downloaded, so it too should be changed to your preference.

The script is run with “$ python3 server_funcs_consolidated.py”. Once you’ve ran for it 2015-2016, you can go in and change the dates to span 2016-2017 as well.

I think those are the only changes that need to be made. If the script is having trouble making the SQL .db file, you may need to $cd to where you want the file located and create it yourself ("$ sqlite3 data.db” then quit the sqlite3 terminal). The script also needed a package installed when we ran it on your server. I don’t remember the exact name of the package (I think it was GTK?) that needed to be installed, but you should be able to tell if the server you’re running it on starts complaining. Finally, if the server runs out of memory, you should be able to go in decrease the date range that’s being downloaded (so run from September to around January, then February to May) until it starts working.
