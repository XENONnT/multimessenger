# import sys
# sys.modules['admix'] = None
import straxen, cutax
import datetime, sys, os

try:
    csv_name = sys.argv[1]
except:
    csv_name = 'SN_test_data.csv'

try:
    run_id = sys.argv[2]
except:
    run_id = 'SN_test_data'

straxen.print_versions(('strax','straxen','cutax','wfsim'))
path_instructions = '/dali/lgrandi/melih/sn_wfsim/instructions/'
st = cutax.contexts.xenonnt_sim_SR0v0_cmt_v5(output_folder='/dali/lgrandi/melih/sn_wfsim/strax_data')

instruction_file = os.path.join(path_instructions, csv_name)
st.set_config(dict(fax_file=instruction_file))
st.set_config(dict(fax_config_override=dict(field_distortion_on=False)))
st.set_config(dict(chunk_size=200))

st.make(run_id, 'truth', progress_bar=False)
st.make(run_id, 'event_info', progress_bar=False)
    
print(f'Finished the job with run id {run_id} at {datetime.datetime.now()}.\nHappy analysis! :D')
