import os
import h5py
from tomoscan.io import HDF5File
from nxtomomill.nexus import NXtomo
from tomwer.core.scan.hdf5scan import HDF5TomoScan

in_nx_tomo_file = "/data/projects/whaitiri/Data/JULY_2022/"
output_dir = "/data/projects/whaitiri/Data/Data_Processing_July2022/NXfiles/"

for dataset in os.listdir(in_nx_tomo_file):

	if (('P28A' in dataset) or ('P28B' in dataset) or ('VCT5' in dataset)) and (not 'txt' in dataset) and (not 'P28A_FT_H_Exp3_3' in dataset):

		path = in_nx_tomo_file + dataset +'/'
		print(path)
		output = output_dir + dataset +'/'
		print(output)
		if not os.path.isdir(output):
			os.mkdir(output)
		for nx in os.listdir(path):
			if ('Scan.nx'in nx):
				nx_path = path + nx
				print(nx_path)
				with HDF5File(nx_path, mode="r") as h5f:
					for entry in h5f.keys():
						print(entry)
						output_file = os.path.join(output, entry.lstrip("/") + "_no_ext.nx" )
						nx_tomo = NXtomo("").load(nx_path, entry, detector_data_as="as_numpy_array")
						nx_tomo.save(file_path=output_file, data_path=entry, overwrite=False)
			else:
				pass
	
