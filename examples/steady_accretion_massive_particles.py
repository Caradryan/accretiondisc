#%%
import subprocess
import os
#import shlex
dir = [
"./Large200_tm008_8e-3/",
"./Large200_tm008_8e-4/",
"./Large200_tm008_8e-5/",
"./Large200_tm008_8e-6/",
"./Large200_tm008_8e-7/",
"./Large200_tm008_8e-8/",
]

for i in range(len(dir)):
    os.popen("mkdir "+dir[i])
#%%
cmd = [
    'python steady_accretion_example.py --save_path ./Large200_tm008_8e-3/ --total_mass 0.008 --total_time 75 --fraction_feed 0.25 --outer_timestep 0.005 --portion_size 8e-3 --save_all',
    'python steady_accretion_example.py --save_path ./Large200_tm008_8e-4/ --total_mass 0.008 --total_time 75 --fraction_feed 0.25 --outer_timestep 0.005 --portion_size 8e-4 --save_all',
    'python steady_accretion_example.py --save_path ./Large200_tm008_8e-5/ --total_mass 0.008 --total_time 75 --fraction_feed 0.25 --outer_timestep 0.005 --portion_size 8e-5 --save_all',
    'python steady_accretion_example.py --save_path ./Large200_tm008_8e-6/ --total_mass 0.008 --total_time 75 --fraction_feed 0.25 --outer_timestep 0.005 --portion_size 8e-6 --save_all',
    'python steady_accretion_example.py --save_path ./Large200_tm008_8e-7/ --total_mass 0.008 --total_time 75 --fraction_feed 0.25 --outer_timestep 0.005 --portion_size 8e-7 --save_all',
    'python steady_accretion_example.py --save_path ./Large200_tm008_8e-8/ --total_mass 0.008 --total_time 75 --fraction_feed 0.25 --outer_timestep 0.005 --portion_size 8e-8 --save_all',
]
#%%
for i, com in enumerate(cmd):
    p = subprocess.Popen(com, stdout=subprocess.PIPE, shell=True)
    #out, err = p.communicate() 
    #file = open(dir[i] + '_log.txt', 'w')
    #file.write(out)
    #file.close()
# %%
