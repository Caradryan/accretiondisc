"""
Example script for steady disc feeding

We perform a calculation of accretion disc around an SMBH
similar in mass to the one at the MW centre (4e6 Msun)

rin is the ISCO for the given bh
rout is corresponds to the rsink in Gadget our simulations
"""
import numpy as np
import matplotlib.pyplot as plt
import units as unt
from accretiondisc import *
import time 


save_all = True # Save calculation results?
file_name = "test_disc" # file name for calculation results?
plots = True # plot the results? 
save_path = "./"
load = False # load already saved disc?
load_path = "./test_disc.npy"

"""
Initiating the accretion disc given SMBH  parameters
"""
D = Disc(bh_mass=0.8, n_rings=100, mdot=0, mesc=0,
         rin=1.147e-06, rout=0.01, ctime=0.01, 
         disc_alpha = 0.1, h_r_init=0.002, PW=True)

"""
Example of loading a seved disc

Disc object is the first in the save .npy file
reading it as dict to assign each property of the new disc
saved disc values
"""
if load:
   print("loading disc {}".format(load_path))
   all_data =  np.load(load_path, allow_pickle=True)
   last_disc = all_data[0]
   
   for item in last_disc.__dict__:
       setattr(D, item, last_disc.__dict__[item])
else:
    print("Not loading")

tT = 75/100   # Calculation time
fF = 1  # Time for which the disc is fed
tM = 0.02/100 # Total mass to be inserted
DT = 5e-3 # outer timestep

start = time.time()
"""
Initiating disc evolution
"""
(D, sigma_arr, 
 ring_luminosity_from_teff_arr, ring_luminosity_from_mdot_arr,
 nu_arr, tau_arr, bh_mass_arr, mesc_arr, mdot_tot_arr, 
 mass_arr, bh_mdot_arr, temperature_effective_arr, 
 temperature_center_arr, current_time_arr) = do_the_evolution(D, total_time=tT, fraction_of_time_feed=fF, total_mass_to_feed=tM, dt=DT, mass_portion=8e-8, hsml=0.01, r_circ=0.003)
print("\n\n This took ",time.time()  - start)
#%%
if save_all:
    np.save( save_path + file_name,np.array( (D, sigma_arr, 
             ring_luminosity_from_teff_arr, ring_luminosity_from_mdot_arr,
             nu_arr, tau_arr, bh_mass_arr, mesc_arr, mdot_tot_arr, 
             mass_arr, bh_mdot_arr, temperature_effective_arr, 
             temperature_center_arr, current_time_arr), dtype=object), allow_pickle=True)
else:
    print("not saving")
#%%
if save_all:
    print("plotting\n plots saved in {}".format(save_path))
    plot_time = current_time_arr * unt.UnitTime_in_s / unt.year / 1e3
    total_mass = ( (sigma_arr * D.area).sum(1) + mesc_arr + bh_mass_arr - 0.8) * unt.UnitMass_in_g / unt.MS
    plt.figure(1, dpi=300)
    plt.title("Mass growth")
    plt.plot(plot_time, (sigma_arr * D.area).sum(1) * unt.UnitMass_in_g / unt.MS, label="disc")
    plt.plot(plot_time, (bh_mass_arr-0.8) * unt.UnitMass_in_g / unt.MS, label="BH")
    plt.plot(plot_time, mesc_arr * unt.UnitMass_in_g / unt.MS,  label="escaped")
    plt.plot(plot_time, total_mass, label="sum"   )
    plt.legend()  
    plt.ylabel("$M [M_{\odot{}}]$")
    #plt.yscale("log")
    plt.xlabel("t [kyr]")
    plt.tight_layout()
    plt.savefig(save_path+"mass_growth.png", dpi=300)
    
    plt.figure( 2, dpi=300 )
    plt.title("Radiation efficiency")
    plt.plot(plot_time, np.abs(ring_luminosity_from_teff_arr.sum(1) / 9e20 / unt.UnitMass_in_g * DT * unt.UnitTime_in_s / (bh_mdot_arr*DT))) 
    plt.plot([plot_time[0],plot_time[-1]],[0.0625, 0.0625], label="$Ledd$", alpha=1, c="r", ls="-")
    plt.ylabel("$\eta$")
    plt.xlabel("t [kyr]")
    plt.tight_layout()
    plt.savefig(save_path+"mass_growth.png", dpi=300)
    Le = 5.2e44

    plt.figure(3, dpi=300)
    plt.title("Luminosity ")
    plt.plot(plot_time, ring_luminosity_from_mdot_arr.sum(1), label="from $\dot{M}$", alpha=0.5, c="b")
    plt.plot(plot_time, ring_luminosity_from_teff_arr.sum(1), label="from $T_{\\rm eff}$", alpha=1, c="k", ls=":")
    plt.plot([plot_time[0],plot_time[-1]],[Le, Le], label="$Ledd$", alpha=1, c="r", ls="-")
    plt.ylabel("$L [\\rm{erg}\,\\rm{s}^-1]$")
    plt.xlabel("t [kyr]")
    plt.yscale("log")
    plt.tight_layout()
    plt.legend()
    plt.savefig(save_path+"luminosity.png", dpi=300)
    LM = ring_luminosity_from_mdot_arr.sum(1)
    LT = ring_luminosity_from_teff_arr.sum(1)

    plt.figure(4, dpi=300)
    plt.title("Luminosity ratio")
    plt.plot(plot_time, LT/LM, label="$\dot{M}$", alpha=0.5, c="b")
    plt.ylabel("$LT/LM$")
    plt.xlabel("t [kyr]")
    plt.tight_layout()
    plt.legend()
    plt.savefig(save_path+"luminosity_ratio.png", dpi=300)
  
    plt.figure(5, dpi=300)
    plt.title("SMBH accretion")
    plt.plot(plot_time, bh_mdot_arr * unt.UnitMass_in_g /unt.MS / unt.UnitTime_in_s * unt.year, label="$\dot{M}$", alpha=0.5, c="r")
    plt.ylabel("$\dot{M}_{\\rm{BH}}$")
    #plt.yscale("log")
    plt.xlabel("t [kyr]")
    plt.tight_layout()
    plt.legend()  
    plt.savefig(save_path+"SMBH_accretion.png", dpi=300)
else:
    print( "not plotting" )
