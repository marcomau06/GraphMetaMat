import yaml
import os
import shutil

TASK = 'transmission'
ETH_FULL_C_VECTOR = False

def check_conflicting_keys(d1, d2, d1_nm, d2_nm):
    conflicting_keys = []
    for k in d1:
        if k in d2:
            conflicting_keys.append(k)
    if len(conflicting_keys) == 0:
        print(f'no conflicts between {d1_nm} and {d2_nm}.')
        return True
    else:
        print(f'conflicting keys between {d1_nm} and {d2_nm}: {conflicting_keys}')
        return False

SRC_DIR = '/home/derek/Documents/GraphMetaMat/src'
# SRC_DIR = '/workspace/MaterialSynthesis/src'
#SRC_DIR = r'\Desktop\DiscoveryProject\MaterialSynthesis'

pn_args_g = os.path.join(SRC_DIR, 'config_general.yaml')
pn_args_m = os.path.join(SRC_DIR, 'config_model.yaml')
pn_args_d = os.path.join(SRC_DIR, 'config_dataset.yaml')
with open(pn_args_g, 'r') as fp:
    args_g = yaml.safe_load(fp)
with open(pn_args_m, 'r') as fp:
    args_m = yaml.safe_load(fp)
with open(pn_args_d, 'r') as fp:
    args_d = yaml.safe_load(fp)

assert check_conflicting_keys(
    args_g, args_m, 'config_general.yaml', 'config_model.yaml')
assert check_conflicting_keys(
    args_m, args_d, 'config_model.yaml', 'config_dataset.yaml')
assert check_conflicting_keys(
    args_g, args_d, 'config_general.yaml', 'config_dataset.yaml')

args = {}
args.update(args_g)
args.update(args_m)
args.update(args_d)

log_dir = args['log_dir']
device = args['device']

shutil.copyfile(pn_args_g, os.path.join(log_dir, 'config_general.yaml'))
shutil.copyfile(pn_args_m, os.path.join(log_dir, 'config_model.yaml'))
shutil.copyfile(pn_args_d, os.path.join(log_dir, 'config_dataset.yaml'))
