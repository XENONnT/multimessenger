# import sys
# sys.modules['admix'] = None
import strax, straxen, wfsim, cutax
import nestpy
import datetime, os, sys

try:csv_name = sys.argv[1]
except: csv_name = 'SN_test_data.csv'
try:run_id = sys.argv[2]
except: run_id = 'SN_test_data'

straxen.print_versions(('strax','straxen','cutax','wfsim'))
path_instructions = '/dali/lgrandi/melih/sn_wfsim/instructions/'
st = cutax.contexts.xenonnt_sim_SR0v0_cmt_v5(output_folder='/dali/lgrandi/melih/sn_wfsim/strax_data')

st.set_config(dict(fax_file=f'{path_instructions}SN_wfsim_instructions_100k.csv'))
st.set_config(dict(fax_config_override=dict(field_distortion_on=False)))
st.set_config(dict(chunk_size=200))


run_id = 'SN_wfsimdata_100k'

truth = st.make(run_id,'truth', progress_bar=False)
evts  = st.make(run_id,'event_info', progress_bar=False)
    
print(f'Finished the job with run id {run_id} at {datetime.datetime.now()}.\nHappy analysis! :D')
