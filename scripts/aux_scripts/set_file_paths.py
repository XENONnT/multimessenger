#!/usr/bin/python
"""
Script to set up a folder structure
Author: Melih Kara kara@kit.edu
"""

import argparse, os
parser = argparse.ArgumentParser()

# assuming a folder structure
# main
 # |___ scripts
 # |___ data
 # |___ imgs
 # |___ logs
 # |___ docs
 
# parser.add_argument("-data_loc", "--data_location", help="Data folder name", default='../data/')
# parser.add_argument("-img_loc", "--image_location", help="Image folder name", default='../imgs/')
# parser.add_argument("-log_loc", "--logs_location", help="Logs folder name", default='../logs/')

# args = parser.parse_args()
# data_path = args.data_location
# img_path = args.image_location
# log_path = args.logs_location

def make_folders(data_path = '../data/',
				 img_path = '../imgs/',
				 log_path = '../logs/'):
	if data_path[-1] != '/': data_path = data_path+'/'
	if img_path[-1]  != '/': img_path = img_path+'/'
	if log_path[-1]  != '/': log_path = log_path+'/'

	print(f"Saving data in    data_path = '{data_path}'")
	print(f"Saving images in  img_path  = '{img_path}'")
	print(f"Keeping logs in   log_path  = '{log_path}'\n")
	os.mkdir(data_path) if not os.path.exists(data_path) else None
	os.mkdir(img_path) if not os.path.exists(img_path) else None
	os.mkdir(log_path) if not os.path.exists(log_path) else None

if __name__ == "__main__":
	print('Called')
	make_folders()

### OLD WAY
# # The notebooks are in ../notebooks/ and scripts are in ../scripts/
# # in order to be able to import from a parent directory I had to play with paths 
# import sys,inspect
# current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parent_dir = os.path.dirname(current_dir) + '/scripts'
# sys.path.insert(0, parent_dir)