from pymms.pymms import MrMMS_SDC_API  as sdc_api
from pyarray import mrarray
import pdb

def maxwellian():
	# Download the data
	sc = 'mms1'
	instr = 'fpi'
	mode = 'brst'
	level = 'l2'
	optdesc = 'des-dist'
	sdc = sdc_api(sc=sc, instr=instr, mode=mode, level=level,
				  optdesc=optdesc,
				  start_date='2017-07-11T22:33:30',
				  end_date='2017-07-11T22:34:40')
	files = sdc.Download()
	
	print(files)

	# Read the data
	dist_vname = '_'.join((sc, optdesc[0:3], 'dist', mode))
	dist = mrarray.from_cdf(files, dist_vname)
	
	pdb.set_trace()
	
	
	
	

if __name__ == 'main':
	maxwellian()