import numpy as np
from . import units as unt 
from numba import jit, njit
from copy import deepcopy 
from dataclasses import dataclass

""""
TODO 
2. make matplotlib optional in main (or remove from main)
4. Consider removing numba dependence
5. transfer some consts from class to Units.py 
"""


@dataclass
class Disc:
    """
    Accretion disc class
    """
    bh_mass: float = 0.8  # black hole mass in code units
    n_rings: int = 100  # number of rings
    mdot: float = 0  # initial bh feeding
    mesc: float = 0  # initialamount of matter escaping the disc
    rin: float = 1.147e-06  # Inner boundary radius
    # intended to be ISCO
    rout: float = 0.01  # outer boundary radius
    ctime: float = 0.1  # Courant factor
    disc_alpha: float = 0.1  # Alpha const. for standart
    # alpha accretion disc
    # alpha~0.1 is appropriate
    h_r_init: float = 0.002  # init. disc Height to Radius ratio
    PW: bool = True  # using Paczyński–Wiita potential
    def __post_init__(self):
        """
        Constast used in the run
        """
        self.GAMMA = 5. / 3.  # Gamma
        self.RING_KAPPA = 0.348  # Thomson scattering opacity in cgs
        """
        Assigning initial parameters
        and
        Declaring variables
        """
        self.current_time = 0  # Time
        self.ddt = 0  # timestep
        self.sigma = np.zeros(self.n_rings)  # Accretion disc's surface density
        self.mass = np.zeros(self.n_rings)  # Accretions disc's mass
        """
        using make_grid() to define simulation grid
        rib - inner boundary of each annulus
        rct - centre of each annulus
        rob - outer boundary of each annulus
        """
        self.rib, self.rct, self.rob, self.area = make_grid(
            self.n_rings, self.rin, self.rout
        )
        self.h_r = (   
            np.ones(self.n_rings) * self.h_r_init
        )  # Filing init.  Height to Radius
        # ratio array with h_r_init
        self.temperature_center = np.ones(self.n_rings) * 10  # Central temperature
        self.temperature_effective = np.zeros(self.n_rings)  # Effective temperature
        self.ring_luminosity_from_mdot = np.zeros(self.n_rings)  # luminosity from mdot
        self.ring_luminosity_from_teff = np.zeros(self.n_rings)  # luminosity from Teff
        self.csound = np.zeros(self.n_rings)  # speed of sound
        self.dsigmadt = np.zeros(self.n_rings)  # change in sigma over dt
        self.dmassdt = np.zeros(self.n_rings)  # change in mass oer dt 
        self.tau = np.zeros(self.n_rings)  # the optical depth
        self.dsigmadt2 = np.zeros((2, self.dsigmadt.shape[0]))
        self.dmassdt2 = np.zeros((2, self.dmassdt.shape[0]))
        self.flux = np.zeros(self.n_rings - 1)  # Surface density flux
        self.mflux = np.zeros(self.n_rings - 1)  # Mass flux
        self.mdot_tot = np.zeros(self.n_rings)
        self.diffusion_switch = False
        self.mass_fed = 0  # mass insertion to disc tracer
        self.omega = np.sqrt( 
            self.bh_mass / (self.rct * self.rct * self.rct)  # angular velocity
        )  
        """
        Turn on PW modification - rteduces to standart if off
        """
        if self.PW:
            self.rg = self.rin / 3  # Swarchilds radius
            print("select rg = ", self.rg)
        else:
            self.rg = 0
            print("select rg = ", self.rg)
        self.omega *= self.rct / (  # angular velocity modification
            self.rct - self.rg
        )  # from P-W potential
        self.nu = (  # initial viscosity
            self.omega * self.disc_alpha * self.rct
            * self.h_r * self.rct * self.h_r
        )
        """
        indices for finite difference
        """
        self.ind0 = np.arange(self.n_rings - 1)
        self.ind1 = np.arange(self.n_rings - 1) + 1
        """
        Factors that remain constant during the run
        """
        self.tau_konst = (  # factor for tau
            unt.UnitMass_in_g / (unt.UnitLength_in_cm * unt.UnitLength_in_cm) 
            * self.RING_KAPPA / 2.
        )
        self.nu_konst = self.disc_alpha * self.rct * self.rct  # factor nu    
        
        self.h_r_konst = (  # factor h_r
            (self.rct - self.rg) * np.sqrt(self.rct) / self.rct             
        )
        self.cs_konst = (  # factor for cs
            np.sqrt(self.GAMMA * unt.kk / (unt.mp * 0.63))                                
            / unt.UnitVelocity_in_cm_per_s 
        )
        self.omega_konst = np.sqrt(  # factor for omega
            1.0 / (self.rct * self.rct * self.rct)
        )
        self.omega_konst *= (  # factor for omega modification from PW
            self.rct / (self.rct - self.rg)
        ) 
        self.temperature_center_konst = (  #factor temperature_center
            (9 * 3 / 4 / 8 
                / (unt.ss * unt.UnitTime_in_s**3 / unt.UnitMass_in_g)) 
            * (self.rct - 1 / 3 * self.rg) ** 2 
            / (self.rct - self.rg) ** 4 * self.rct**-1
        ) ** (1 / 4)
        """
        factor for finite difference diffusion solving 
        """
        """
        Surface density factors
        """
        self.diff_in_fact_1 = ( 
            np.sqrt(self.rct)[self.ind1] ** 3 
            * ((self.rct[self.ind1] - 1 / 3 * self.rg) 
                  / (self.rct[self.ind1] - self.rg) ** 2) 
        ) / np.diff(self.rct) 
        self.diff_in_fact_0 = ( 
            np.sqrt(self.rct)[self.ind0] ** 3 
            * ((self.rct[self.ind0] - 1 / 3 * self.rg) 
                  / (self.rct[self.ind0] - self.rg) ** 2) 
        ) / np.diff(self.rct)
        self.diff_out_fact = np.sqrt(self.rob)[self.ind0] ** (-1) * (
            (self.rob[self.ind0] - self.rg) ** 2 / (self.rob[self.ind0] - 3 * self.rg)
        )
        self.dsigma_konst = 3.0 / (self.rct * (self.rob - self.rib)) 
        """
        Mass factors
        """
        self.mdiff_in_fact_1 = ( 
            (self.rob[self.ind1] - self.rib[self.ind1]) ** (-1) 
            * np.sqrt(self.rct)[self.ind1] 
            * ((self.rct[self.ind1] - 1 / 3 * self.rg) 
                   / (self.rct[self.ind1] - self.rg) ** 2)
        ) / np.diff(self.rct)
        self.mdiff_in_fact_0 = ( 
            (self.rob[self.ind0] - self.rib[self.ind0]) ** (-1) 
            * np.sqrt(self.rct)[self.ind0] 
            * ((self.rct[self.ind0] - 1 / 3 * self.rg) 
                   / (self.rct[self.ind0] - self.rg) ** 2)) / np.diff(self.rct)        
        self.mdiff_out_fact = np.sqrt(self.rob)[self.ind0] ** (-1) * (
            (self.rob[self.ind0] - self.rg) ** 2 / (self.rob[self.ind0] - 3 * self.rg)
        )
        self.dmass_konst = 3.0
        """
        Luminosity from mdot factors
        """
        self.lum_fact_o = ( 
            3 / 4 * unt.GG * unt.UnitMass_in_g
            * ((self.rob - 1 / 3 * self.rg) 
                  / (self.rob - self.rg) ** 2) 
            * self.rob ** (-3/ 2) 
            * (self.rob ** (3 / 2) / (self.rob - self.rg) 
                   - 3 ** (3 / 2) * self.rg**0.5 / 2) 
            * unt.UnitLength_in_cm ** (-2) 
        )
        self.lum_fact_i = (
            3 / 4 * unt.GG * unt.UnitMass_in_g 
            * ((self.rib - 1 / 3 * self.rg) 
                    / (self.rib - self.rg) ** 2) 
            * self.rib ** (-3 / 2) 
            * (self.rib ** (3 / 2) / (self.rib - self.rg) 
                  - 3 ** (3 / 2) * self.rg**0.5 / 2) 
            * unt.UnitLength_in_cm ** (-2)
        )
        self.lum_fact = (
            (self.lum_fact_i + self.lum_fact_o) 
            * (self.rob - self.rib) 
            * unt.UnitLength_in_cm * unt.UnitMass_in_g / unt.UnitTime_in_s 
        )
        """
        Luminosity from temperature_effective factors
        """
        self.lum_fact_teff = (
            4 * np.pi * (self.rob - self.rib) 
            * self.rct * unt.UnitLength_in_cm 
            * unt.UnitLength_in_cm * unt.ss
        )


def make_grid(n_rings, rin, rout):
    """
    F that makes the "grid" of the accretion disc
    The disc is made up of N rings defined by rin and rout   
    """
    """
    Seting up log space grid
    """
    rat12 = (rout / rin) ** (1 / (2 * n_rings))
    r = rin * rat12 ** np.arange(2 * n_rings + 1)
    rib = r[:-2:2]
    rct = r[1::2]
    rob = r[2::2]
    """
    stretching and pushing grid to align rob[0] with isco
    and rin[-1] with rout ( and rsink )
    """
    stretch = (rout - rin) / (rib[-1] - rib[1])
    push = rib[1] * stretch - rin
    rib = rib * stretch - push
    rct = rct * stretch - push
    rob = rob * stretch - push
    deltar = (rob[0] - rib[0]) / 2
    rib += deltar
    rob += deltar
    rct += deltar
    """
    defining surface area of each annulus
    """
    area = np.pi * (rob * rob - rib * rib)
    return rib, rct, rob, area 

@njit()
def wd(normalized_dist):
    """
    Simplified Wendland kernel function
    """
    return (
        (1 - normalized_dist) * (1 - normalized_dist) 
        * (1 - normalized_dist) * (1 - normalized_dist) 
        * (1 + 4 * normalized_dist)
    )

@njit()
def disc_feeding_const_r(
    n_rings, rct, rout, area, mass_portion=8e-08, r_circ=0.003, hsml=0.01
    ):
    """
    Disc feeding function
    Particle mass is distributed split around the r_circ
    """

    """
    Mirroring the disc grid around 0
    """
    double_rct = np.zeros(n_rings * 2) 
    double_rct[:n_rings] = rct[::-1]
    double_rct[n_rings:] = -rct
    """normalized distance to each grid point"""
    normalized_dist = (double_rct - r_circ) / hsml 
    """
    Adding the kernel weighted mass to each grid point
    """
    kernel = wd(np.abs(normalized_dist)) * np.abs(double_rct / rout)
    kernel[np.abs(normalized_dist) > 1] = 0
    sigma_change = ( 
        (kernel[:n_rings][::-1] + kernel[n_rings:]) / kernel.sum() * mass_portion / area
    )
    return sigma_change


def make_empty_arrays(total_steps, n_rings):
    """
    Making empty arrays for saving parameter values on each outer timestep
    """
    sigma_arr = np.zeros((total_steps, n_rings))  # Surface density storage
    mass_arr = np.zeros((total_steps, n_rings))  # Mass storage
    ring_luminosity_from_teff_arr = np.zeros((total_steps, n_rings))  # Luminosity
    # from temperature_effective
    ring_luminosity_from_mdot_arr = np.zeros((total_steps, n_rings))  # Luminosity
    # from mdot
    nu_arr = np.zeros((total_steps, n_rings))  # Viscosity storage
    tau_arr = np.zeros((total_steps, n_rings))  # Optical depth storage
    bh_mass_arr = np.zeros(total_steps)  # BH mass storage
    mesc_arr = np.zeros(total_steps)  # Escaped mass storage
    bh_mdot_arr = np.zeros(total_steps)  # BH mdot storage
    temperature_effective_arr = np.zeros(
        (total_steps, n_rings)
    )  # effective temperature
    temperature_center_arr = np.zeros((total_steps, n_rings))  # central temperature
    current_time_arr = np.zeros(total_steps)
    return (
        sigma_arr,
        mass_arr,
        ring_luminosity_from_teff_arr,
        ring_luminosity_from_mdot_arr,
        nu_arr,
        tau_arr,
        bh_mass_arr,
        mesc_arr,
        bh_mdot_arr,
        temperature_effective_arr,
        temperature_center_arr,
        current_time_arr,
    )


def do_the_evolution(
    Disc,
    total_time=75,
    fraction_of_time_feed=1,
    total_mass_to_feed=0.02,
    dt=5e-3,
    mass_portion=8e-8,
    hsml=0.01,
    r_circ=0.003,
):
    """
    Main driving function for 
    Steady accretion over some period of time
    
    the feeding is discrete to approximate particle accretion
    in Gadget-3
    """
    import time  # to meassure time

    feeding_time = total_time * fraction_of_time_feed  # Determining disc feeding time
    n_steps = feeding_time / dt  # Determine determine outer
    # outer number of steps

    mPerStep = total_mass_to_feed / n_steps  # Determine mass fed
    # over outter step

    nPerStep = int(mPerStep / mass_portion)  # Number of portions of mass
    """
    Declaring storage arrays
    """
    (
        sigma_arr,
        mass_arr,
        ring_luminosity_from_teff_arr,
        ring_luminosity_from_mdot_arr,
        nu_arr,
        tau_arr,
        bh_mass_arr,
        mesc_arr,
        bh_mdot_arr,
        temperature_effective_arr,
        temperature_center_arr,
        current_time_arr,
    ) = make_empty_arrays(int(n_steps / fraction_of_time_feed), Disc.n_rings)
    """
    Outer steps
    """
    for st in range(int(n_steps / fraction_of_time_feed)):
        
        """
        Check if we should feed the disc
        """
        if st <= int(n_steps):
            """
            Inserting mass using a select function
            """
            Disc.sigma += disc_feeding_const_r(
                Disc.n_rings,
                Disc.rct,
                Disc.rout,
                Disc.area,
                mass_portion=mass_portion * nPerStep,
                r_circ=r_circ,
                hsml=hsml,
            )
            Disc.mass_fed += mass_portion * nPerStep

        Disc.ctime = 0.01  # reseting courant number
        Disc = update_params_jit_wrap(Disc)  # updating disc parameters
        Disc.diffusion_switch = True  # turning on diffusion iteration
        Disc.mdot_tot = np.zeros(Disc.n_rings)  # declaring current mdot_tot

        """
        diffusion iteration
        """
        while Disc.diffusion_switch: 
            """
            Copying current Disc
            If timesteps are too large using these to reset the disc
            """
            Disc_c = deepcopy(Disc)            
            Disc.ddt = get_diffusion_time_step_jit_wrap(
                Disc
            )  # determining the inner timestep
            diffsteps = np.ceil(dt / np.nanmin(Disc.ddt))
            Disc.ddt = dt / diffsteps
            diffsteps = int(diffsteps)
            print( 
                "\r At step {:3d}| diffstep {:5d} | total {:5d} | ddt {:3e} ".format(
                    st, diffsteps,int(n_steps / fraction_of_time_feed), Disc.ddt), 
            end="",flush=True)
            for dst in range(diffsteps):
                Disc = diffusion_jit_wrap(Disc)
                if (Disc.sigma < 0).sum() > 0:
                    """
                    If inner timestep was too large we reset the disc
                    and repeat with smaller inner timestep
                    """
                    Disc = Disc_c
                    Disc.ctime *= 0.5
                    break
                Disc = update_params_jit_wrap(Disc)  # Updating disc parameters
                
            """
            Things that need to happen once during each diffusion phase
            """
            if dst == diffsteps - 1:
                """
                Using mean mdot to determine Luminosity
                """
                Disc.ring_luminosity_from_mdot = (
                    Disc.mdot_tot / diffsteps * Disc.bh_mass * Disc.lum_fact
                )
                """
                Using disc parameters to determine temperature_effective 
                And luminosity from temperature_effective
                """
                mask = Disc.tau > 0
                Disc.temperature_effective[mask] = (
                    Disc.temperature_center[mask] ** 4  # here its temperature^4
                    * 4 / 3 / Disc.tau[mask]
                )
                Disc.ring_luminosity_from_teff = (
                    Disc.lum_fact_teff * Disc.temperature_effective
                )  # here its temperature^4
                Disc.ring_luminosity_from_teff[-1] = 0
                Disc.ring_luminosity_from_teff[0] = 0
                Disc.diffusion_switch = False 
                # End of diffusion step; finished with one outer timestep
                
         
            
        """
        Filling storage arrays for this step
        """
        Disc.current_time += dt
        temperature_center_arr[st, :] = Disc.temperature_center
        temperature_effective_arr[st, :] = Disc.temperature_effective ** (1 / 4)
        ring_luminosity_from_teff_arr[st, :] = Disc.ring_luminosity_from_teff
        ring_luminosity_from_mdot_arr[st, :] = Disc.ring_luminosity_from_mdot
        sigma_arr[st, :] = Disc.sigma
        mass_arr[st, :] = Disc.mass
        tau_arr[st, :] = Disc.tau
        nu_arr[st, :] = Disc.nu
        bh_mass_arr[st] = Disc.bh_mass
        bh_mdot_arr[st] = Disc.mdot
        mesc_arr[st] = Disc.mesc
        current_time_arr[st] = Disc.current_time 
    return (
        Disc,
        sigma_arr,
        ring_luminosity_from_teff_arr,
        ring_luminosity_from_mdot_arr,
        nu_arr,
        tau_arr,
        bh_mass_arr,
        mesc_arr,
        mass_arr,
        bh_mdot_arr,
        temperature_effective_arr,
        temperature_center_arr,
        current_time_arr,
    )


@njit()
def diffusion_jit(sigma, area, nu, 
                  diff_in_fact_1, diff_in_fact_0, diff_out_fact, dsigma_konst,
                  mdiff_in_fact_1, mdiff_in_fact_0, mdiff_out_fact, dmass_konst,
                  dsigmadt2, dmassdt2, bh_mass, mdot_tot, ddt, mesc, mdot, ind0, ind1
                  ):
    """
    Determining the change in surface density and mass due to visous diffusion   
    """  
    mass = sigma * area 
    
    """
    Determining Surface density and mass fluxes over boundaries in each direction
    """
    # main diffusion for surface density
    flux = nu[1:] * sigma[1:] * diff_in_fact_1 - nu[:-1] * sigma[:-1] * diff_in_fact_0

    dsigmadt2[0, :-1] = flux * diff_out_fact
    dsigmadt2[1, 1:] = -flux * diff_out_fact
    dsigmadt2 *= dsigma_konst

    # main diffusion for Mass
    mflux = nu[1:] * mass[1:] * mdiff_in_fact_1 - nu[:-1] * mass[:-1] * mdiff_in_fact_0

    dmassdt2[0, :-1] = mflux * mdiff_out_fact
    dmassdt2[1, 1:] = -mflux * mdiff_out_fact
    dmassdt2 *= dmass_konst

    mask2 = dmassdt2[0, :] > 0
    mdot_tot[mask2] += dmassdt2[0, :][mask2]

    sigma = sigma + ddt * np.sum(dsigmadt2, 0)  # updated surface density
    mass = mass + ddt * np.sum(dmassdt2, 0)  # updated mass

    bh_mass += sigma[0] * area[0]  # adding mass from
    # inner boundary to bh
    mesc += sigma[-1] * area[-1]  # adding mass from
    # from outer boundary to
    # "escaped mass"
    mdot = sigma[0] * area[0] / ddt  # BH accretion rate
    """
    removing mass from boundaries
    """
    sigma[0] = 0
    sigma[-1] = 0
    mass[0] = 0
    mass[-1] = 0
    return sigma, mass, bh_mass, mdot, mesc, mdot_tot

def diffusion_jit_wrap(Disc):
    """
    Wraper to update params to use njit()
    """
    A = diffusion_jit(Disc.sigma, Disc.area, Disc.nu, 
                      Disc.diff_in_fact_1, Disc.diff_in_fact_0, Disc.diff_out_fact, Disc.dsigma_konst,
                      Disc.mdiff_in_fact_1, Disc.mdiff_in_fact_0, Disc.mdiff_out_fact, Disc.dmass_konst,
                      Disc.dsigmadt2, Disc.dmassdt2, Disc.bh_mass, Disc.mdot_tot, Disc.ddt, Disc.mesc, Disc.mdot, Disc.ind0, Disc.ind1
                      )
    Disc.sigma, Disc.mass, Disc.bh_mass, Disc.mdot, Disc.mesc, Disc.mdot_tot = A
    return Disc


@njit()
def get_diffusion_time_step_jit(nu, rib, rob, ctime):
    """
    get possible diffusion (inner) timestep
    """
    mask = nu > 0
    ddt = (rob[mask] - rib[mask]) * (rob[mask] - rib[mask]) / nu[mask] * ctime
    return ddt

def get_diffusion_time_step_jit_wrap(Disc):
    return get_diffusion_time_step_jit(Disc.nu, Disc.rib, Disc.rob, Disc.ctime)


@njit()
def update_params_jit(sigma, h_r, temperature_center_konst,
                     tau_konst, nu, nu_konst, omega, omega_konst,
                     bh_mass, h_r_konst, cs_konst):
    """
    Updates parameters to values corresponding to the current sigma
    wrapped to use with njit()
    """
    tau = sigma * tau_konst 
    temperature_center = (tau * nu * sigma * bh_mass) ** (1 / 4) * temperature_center_konst  
    temperature_center[temperature_center < 10] = 10
    omega = bh_mass**0.5 * omega_konst
    csound = np.sqrt(temperature_center) * cs_konst
    h_r = csound * np.sqrt(1.0 / bh_mass) * h_r_konst
    h_r[0] = h_r[1]
    h_r[-1] = h_r[-2]
    nu = omega * h_r * h_r * nu_konst
    nu[0] = 0
    nu[-1] = 0
    return [tau, temperature_center, nu, csound, h_r, omega]

def update_params_jit_wrap(Disc):
    """
    Wraper to update params to use njit()
    """
    A = update_params_jit(  
        Disc.sigma, Disc.h_r, Disc.temperature_center_konst,
        Disc.tau_konst, Disc.nu, Disc.nu_konst,  Disc.omega, Disc.omega_konst,
        Disc.bh_mass, Disc.h_r_konst, Disc.cs_konst)
    Disc.tau, Disc.temperature_center, Disc.nu, Disc.csound, Disc.h_r, Disc.omega = A
    return Disc


#%%
if __name__ == "__main__":
    import time 
    """
    Initiating the Disc
    """
    print("Will it run?")
    D = Disc(
        bh_mass=0.8,
        n_rings=100,
        mdot=0,
        mesc=0,
        rin=1.1474144826096348e-06,
        rout=0.01,
        ctime=0.01,
        disc_alpha=0.1,
        h_r_init=0.002,
        PW=True,
    )
    
    start = time.time()
    
    tT = 75 / 100  # Calculation time
    fF = 1  # Time for which the disc is fed
    tM = 0.02 / 100  # Total mass to be inserted
    DT = 5e-3  # outer timestep
    
    (D, Sigma, LumT, LumM, 
     Nu, Tau, BHMass, esc, dsigTot,
     SigmadM, bh_mdot_arr, temperature_effective_arr, temperature_center_arr, current_time_arr) = do_the_evolution(D, 
                                                       total_time=tT, 
                                                       fraction_of_time_feed=fF, 
                                                       total_mass_to_feed=tM, 
                                                       dt=DT,
                                                       mass_portion=8e-8, 
                                                       hsml=0.01, 
                                                       r_circ=0.003)
    print("\n\n This took ", time.time() - start)
    print("it did :)")
    