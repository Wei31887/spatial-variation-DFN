import numpy as np
import pandas as pd
import math 
import random
import scipy.stats as st
import numpy.matlib
from scipy.linalg import null_space
import matplotlib.pyplot as plt
from shapely import geometry as geo

from FractureGenerate import Generate
from ModelValidation import ModelProcessing

def random_wells(well_num, region):
    wells = []
    for i in range(well_num):
        rand_x, rand_y = random.uniform(-1, 1), random.uniform(-1, 1)
        center_x = region[0] + rand_x * (region[3]/2)
        center_y = region[1] + rand_y * (region[4]/2)
        tem_well = [[center_x, center_y, region[2] + region[5]/2],\
                    [center_x, center_y, region[2] - region[5]/2]]
        wells.append(tem_well)
    wells = np.array(wells)    
    return wells

""" ----CONSTANT---- """
region = [11, 22.5, 17.5, 8, 8, 35]
wells = random_wells(20, region)
size_dict = {}
SIZE = [1, 4, 8, 10.6667, 16, 20]
for i, s in enumerate(SIZE):
    size_dict[s] = i

""" ----JB---- """
trace_mean = 5.0
trace_loc = 0.12
trace_params = {
    'loc':trace_loc, 
    'scale':trace_mean
}
trace_length = ['expon', trace_params, True, (0.5, 20)]
fracture_shape = 6
aperture = ['tracelength', (-4 * (10**-7), 2*(10**-5), 2*(10**-5))]

# size effect
scale = SIZE[4]
set_name = f'JB_16_heshe'
attitude = [(119, 54), 20.15]
p10 = (2.15, 0.15)
subdomain_scale = 1


# """ ----J1---- """
# trace_mean = 0.76
# trace_loc = 0.2100
# trace_std = 1.2800
# trace_params = {
#     'loc':trace_loc, 
#     'scale':trace_mean,
#     's':trace_std
# }
# trace_length = ['lognorm', trace_params, True, (0.05, 20)]
# fracture_shape = 6
# aperture = ['tracelength', (-8 * (10**-7), 3*(10**-5), 5*(10**-7))]

# # size effect
# scale = SIZE[4]
# set_name = 'J1_16_heshe'
# attitude = [(293, 22), 14.3]
# p10 = (3.37, 0.3)
# subdomain_scale = 1

""" ----J2---- """
# trace_mean = 2.8740
# trace_loc = 0.0
# trace_params = {
#     'loc':trace_loc, 
#     'scale':trace_mean
# }
# trace_length = ['expon', trace_params, True, (0.05, 20)]
# fracture_shape = 6
# aperture = ['tracelength', (-8 * (10**-7), 3*(10**-5), 5*(10**-7))]

# # size effect
# scale = SIZE[4]
# set_name = 'J2_16_heshe'
# attitude = [(59, 60), 26.3]
# p10 = (0.28, 0.15)
# subdomain_scale = 1

""" ----Boundary effect---- """
# trace_length = ['constant', 10, True, (0.5, 20)]
# fracture_shape = 6
# aperture = ['tracelength', (-4 * (10**-7), 2*(10**-5), 2*(10**-5))]

# # size effect
# scale = 10
# set_name = f'boundary_poisson'
# attitude = [(0, 0), 100]
# p10 = (1, 0.050)
# subdomain_scale = 8

if __name__ == '__main__':
    # #wells  = np.array([
    #     [[region[0], region[1], region[2] + region[5]/2],
    #     [region[0], region[1], region[2] - region[5]/2]]
    # ])
    set1 = Generate(set_name, region, wells, attitude, trace_length, \
                    fracture_shape, aperture, p10, scale, subdomain_scale, surface_point=True)
    set1.main_generate('uniform')

    # Verification
    #sample_time = round(1)
    sample_time = 3
    set1_out = ModelProcessing(set1.df, set1.set_name, set1.scale, set1.fracture_num, set1.region)

    wells = random_wells(5, region)
    t = set1_out.p10_validation(wells, sample_time)
    # spacing = set1_out.spacing_sampling(wells, sample_time)
    # df = pd.DataFrame(spacing)
    # df.to_excel(f'output_JB_{size_dict[scale]}.xlsx')
    #set1_out.visualize_model()
    print(f'P10 mean, std:{t[0],t[1]}')

    # Output 
    set1_out.output_fab()




