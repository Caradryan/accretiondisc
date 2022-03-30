"""
Code to compute diffusion example 

compare w.
"Accretion Power in Astrophysics" Fig. 5.1. (page 83) 
Frank, King, Raine 2002
"""
import numpy as np
import matplotlib.pyplot as plt
import accretiondisc.units as unt
import accretiondisc.disc as ad
from copy import deepcopy
import time 

def acc_power_example(Disc,       # passing the Disc
                          insertion_ring=200,  # Mass insertion ring
                          ctimediff=1,# init courant factor 
                          dt=1e-5,    # init timestep
                          diffTest=False, # boundary condition fudging
                          log=True):      # Additional info 
    """
    A test that perform diffusion with constant viscosity.
    To compare with "Accretion Power in Astrophysics" Fig. 5.1. (page 83) 
    Frank, King, Raine 2002
    OR
    Fig. 1. from Accretion discs in astrophysics
    http://adsabs.harvard.edu/abs/1981ARA%26A..19..137P
    10.1146/annurev.aa.19.090181.001033
    Pringle 1983
    """
    m = 1  #amount of mass inserted
    Disc.sigma[insertion_ring] = m / Disc.area[insertion_ring]
    data = []
    tt=0
    counter  = 0 # counter for saving plotting params
    counter1 = 0 # counter for changed timesteps
    plotVars = [ 0.002, 0.008, # times at which data is stored
                 0.032, 0.128, 0.512 ]
    Disc.nu[:] = 1e-10
    dimensionless_sigma = [] # Storage list
    dimensionless_time = 0 # dimensionless time
    Disc.ctime = ctimediff # courant determination
    while dimensionless_time<0.512: # main cycle
        """
        Copying current Disc
        If timesteps are too large using these to reset the disc
        """
        Disc_c = deepcopy(Disc)            
        Disc.ddt = ad.get_diffusion_time_step_jit_wrap(Disc)      
        diffsteps = np.ceil(dt / np.nanmin(Disc.ddt))
        Disc.ddt = dt / diffsteps
        diffsteps = int(diffsteps)
        fail = False
        dtf = 0
        for dst in range(diffsteps):
            Disc = ad.diffusion_jit_wrap(Disc)
            if ( (Disc.sigma<0).sum()>0 ):
                """
                If inner timestep was too large we reset the disc
                and repeat with smaller inner timestep
                """
                Disc = Disc_c
                dtf = dst / diffsteps
                Disc.ctime *= 0.5
                counter1+=1
                fail = True
                break
        if fail==False:
            """
            Adding data to log and increasing dimensionless time
            """
            tt += dt
            dimensionless_time = 12 * Disc.nu[insertion_ring] * tt / (Disc.rct[insertion_ring] * Disc.rct[insertion_ring])
            disc_mass = ( Disc.sigma * Disc.area )
            bh_acc_mass = ( Disc.bh_mass - 0.8 ) 
            data.append([dimensionless_time, tt, dt, Disc.ctime, counter1, diffsteps, disc_mass, bh_acc_mass])
            Disc.ctime = 0.1 * 0.5**4 # guess for better default ctime
            
        if dimensionless_time>plotVars[counter]:
            """
            Adding data to storage for plotting at predetermined times
            """
            dimensionless_sigma.append(Disc.sigma * Disc.rct[insertion_ring]**2 * np.pi) #/1~/m
            counter+=1
        if log: # output more params
            print( '\r At dimensionless time = {:3e}, ctime = {:3e}, ctimechange = {:3d}, diffsteps = {:3e}, disc_mass = {:03e}, smhb+disc mass = {:03e}'.format(dimensionless_time, Disc.ctime, counter1, diffsteps, (Disc.sigma*Disc.area).sum(), (Disc.bh_mass-0.8) + (Disc.sigma*Disc.area).sum()), end='',flush=True)
        else:
            print( '\r At dimensionless time  = {:3e}'.format(dimensionless_time), end='', flush=True)
    return Disc, dimensionless_sigma, data

save_all = True # Save calculation results?
file_name = "diff_example" # file name for calculation results?
plots = True # plot the results? 
save_path = "./"


start = time.time()  
D = ad.Disc(bh_mass=0.8, n_rings=200, mdot=0, mesc=0,
        rin=1.147e-06, rout=0.01, ctime=0.01, 
        disc_alpha = 0.1, h_r_init=0.002, PW=True)

insertion_distance = 1e-5
insertion_ring_number = np.abs(D.rct - insertion_distance).argmin()
D, dimensionless_sigma, data = acc_power_example(D, insertion_ring_number, ctimediff=0.01, log=True )
print("\n\n This took ",time.time()  - start)

if save_all: 
    np.save(save_path + file_name, np.array((D, dimensionless_sigma, data, insertion_ring_number), dtype=object), allow_pickle=True)


if plots:
    plt.figure(1, dpi=300)
    plt.title("")
    for i in range(len(dimensionless_sigma)):
        plt.plot(D.rct/D.rct[insertion_ring_number], dimensionless_sigma[i])
        plt.xlim(D.rct[0]/D.rct[insertion_ring_number],2.5)  
        plt.ylim(0,6.5)
    plt.xlabel("$R / R_0$")
    plt.text(1.1, 5.5, "$\\tau = 0.002$", fontsize=8)
    plt.text(1.1, 3.2, "$\\tau = 0.008$", fontsize=8)
    plt.text(1.1, 1.5, "$\\tau = 0.032$", fontsize=8)
    plt.text(1.2, 0.8, "$\\tau = 0.128$", fontsize=8)
    plt.text(0.2, 0.51, "$\\tau = 0.512$", fontsize=8)      
    plt.xlabel("$R/R_0$")
    plt.ylabel("$\\pi \\Sigma R_0^2/m$")
    plt.tight_layout()
    plt.savefig(save_path+"diffusion_example.png", dpi=300)