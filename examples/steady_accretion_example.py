"""
Example script for steady disc feeding

We perform a calculation of accretion disc around an SMBH
similar in mass to the one at the MW centre (4e6 Msun)

rin is the ISCO for the given bh
rout is corresponds to the rsink in Gadget our simulations
"""
import numpy as np
import matplotlib.pyplot as plt
import accretiondisc.units as unt
import accretiondisc.disc as ad
import time
import argparse

"""
parsing parameters for command line use
"""
parser = argparse.ArgumentParser()
parser.add_argument(
    "--save_all",
    action="store_true",
    help="pass as argument to store the calculation result",
)
parser.add_argument(
    "--file_name", type=str, default="diff_example", help="Calculation result filename"
)
parser.add_argument(
    "--plot", action="store_true", help="pass as argument to store plot the results"
)
parser.add_argument(
    "--save_path", type=str, default="./", help="Path where plots and data is saved"
)
parser.add_argument(
    "--load", action="store_true", help="pass as argument to load a previous result"
)
parser.add_argument(
    "--load_path", type=str, default="./test_disc.npy", help="location of file to load"
)
parser.add_argument(
    "--total_time",
    type=float,
    default=75 / 100,
    help="total calculation time 75 code u ~ 500 kyr",
)
parser.add_argument(
    "--fraction_feed",
    type=float,
    default=1,
    help=" fraction of time for which the disc is fed",
)
parser.add_argument(
    "--total_mass", type=float, default=0.02 / 100, help="mass fed over given time"
)
parser.add_argument(
    "--outer_timestep",
    type=float,
    default=5e-3,
    help="timesteps for storing and checking diffusion timestep",
)

args = parser.parse_args()

print(
    "--save_all ",
    args.save_all,
    "\n--file_name ",
    args.file_name,
    "\n--plot ",
    args.plot,
    "\n--save_path ",
    args.save_path,
    "\n--load ",
    args.load,
    "\n--load_path ",
    args.load_path,
    "\nModel parameters:" "\n--total_time",
    args.total_time,
    "\n--fraction_feed",
    args.fraction_feed,
    "\n--total_mass",
    args.total_mass,
    "\n--outer_timestep",
    args.outer_timestep,
)

"""
Initiating the accretion disc given SMBH  parameters
"""
disc = ad.Disc(
    bh_mass=0.8,  # SMBH mass
    n_rings=100,  # number of rings comprizing the disc
    mdot=0,  # SMBH accretion rate
    mesc=0,  # amount of matter that escaped the disc
    rin=1.147e-06,  # inner boundary, assumed ISCO
    rout=0.01,  # outer boundary
    ctime=0.01,  # courant factor
    disc_alpha=0.1,  # accretion disc alpha (~0.1 is ok)
    h_r_init=0.002,  # height to radius ratio
    PW=True,  # using Paczyńsky-Wiita potential
)  # diffusion reduces to Kepler if False


"""
Example of loading a seved disc

Disc object is the first in the save .npy file
reading it as dict to assign each property of the new disc
saved disc values
"""
if args.load:
    print("loading disc {}".format(args.load_path))
    all_data = np.load(args.load_path, allow_pickle=True)
    last_disc = all_data[0]

    for item in last_disc.__dict__:
        setattr(disc, item, last_disc.__dict__[item])
else:
    print("Not loading")


start = time.time()
"""
Initiating disc evolution
"""
(
    disc,  # disc object
    sigma_arr,  # surface density
    ring_luminosity_from_teff_arr,  # luminosity from Teff
    ring_luminosity_from_mdot_arr,  # luminosity from mdot
    nu_arr,  # viscosity
    tau_arr,  # optical depth
    bh_mass_arr,  # SMBH mass
    mesc_arr,  # escaped mass
    mass_arr,  # mass of the disc (sigma * area)
    bh_mdot_arr,  # SMBH accretion rate
    temperature_effective_arr,  # effective temperature
    temperature_center_arr,  # central temperature
    current_time_arr,  # time in code units
) = ad.do_the_evolution(
    disc,  # disc object
    total_time=args.total_time,  # total time of simulation run
    fraction_of_time_feed=args.fraction_feed,  # fraction of time that the disc is fed
    total_mass_to_feed=args.total_mass,  # total mass fed to the disc
    dt=args.outer_timestep,  # "outer" timestep - intervals between
    # disc feeding, diffusion step determination and data storage
    mass_portion=8e-8,  # size of mass portion (discrete SPH particle)
    hsml=0.01,  # smoothing lenght (minimum in an SPH simulation)
    r_circ=0.003,  # mass insertion radius
)


print("\n\n This took ", time.time() - start)
#%%
if args.save_all:
    np.save(
        args.save_path + args.file_name,
        np.array(
            (
                disc,
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
            ),
            dtype=object,
        ),
        allow_pickle=True,
    )
else:
    print("not saving")
#%%
if args.save_all:
    print("plotting\n plots saved in {}".format(args.save_path))
    plot_time = current_time_arr * unt.UnitTime_in_s / unt.year / 1e3
    tot_mass = (
        ((sigma_arr * disc.area).sum(1) + mesc_arr + bh_mass_arr - 0.8)
        * unt.UnitMass_in_g
        / unt.MS
    )
    plt.figure(1, dpi=300)
    plt.title("Mass growth")
    plt.plot(
        plot_time,
        (sigma_arr * disc.area).sum(1) * unt.UnitMass_in_g / unt.MS,
        label="disc",
    )
    plt.plot(plot_time, (bh_mass_arr - 0.8) * unt.UnitMass_in_g / unt.MS, label="BH")
    plt.plot(plot_time, mesc_arr * unt.UnitMass_in_g / unt.MS, label="escaped")
    plt.plot(plot_time, tot_mass, label="sum")
    plt.legend()
    plt.ylabel("$M [M_{\odot{}}]$")
    # plt.yscale("log")
    plt.xlabel("t [kyr]")
    plt.tight_layout()
    plt.savefig(args.save_path + "mass_growth_example.png", dpi=300)

    plt.figure(2, dpi=300)
    plt.title("Radiation efficiency")
    plt.plot(
        plot_time,
        np.abs(
            ring_luminosity_from_teff_arr.sum(1)
            / 9e20
            / unt.UnitMass_in_g
            * args.outer_timestep
            * unt.UnitTime_in_s
            / (bh_mdot_arr * args.outer_timestep)
        ),
    )
    plt.plot(
        [plot_time[0], plot_time[-1]],
        [0.0625, 0.0625],
        label="$Ledd$",
        alpha=1,
        c="r",
        ls="-",
    )
    plt.ylabel("$\eta$")
    plt.xlabel("t [kyr]")
    plt.tight_layout()
    plt.savefig(args.save_path + "mass_growth_example.png", dpi=300)
    Le = 5.2e44

    plt.figure(3, dpi=300)
    plt.title("Luminosity ")
    plt.plot(
        plot_time,
        ring_luminosity_from_mdot_arr.sum(1),
        label="from $\dot{M}$",
        alpha=0.5,
        c="b",
    )
    plt.plot(
        plot_time,
        ring_luminosity_from_teff_arr.sum(1),
        label="from $T_{\\rm eff}$",
        alpha=1,
        c="k",
        ls=":",
    )
    plt.plot(
        [plot_time[0], plot_time[-1]], [Le, Le], label="$Ledd$", alpha=1, c="r", ls="-"
    )
    plt.ylabel("$L [\\rm{erg}\,\\rm{s}^-1]$")
    plt.xlabel("t [kyr]")
    plt.yscale("log")
    plt.tight_layout()
    plt.legend()
    plt.savefig(args.save_path + "luminosity_example.png", dpi=300)
    LM = ring_luminosity_from_mdot_arr.sum(1)
    LT = ring_luminosity_from_teff_arr.sum(1)

    plt.figure(4, dpi=300)
    plt.title("Luminosity ratio")
    plt.plot(plot_time, LT / LM, label="$\dot{M}$", alpha=0.5, c="b")
    plt.ylabel("$LT/LM$")
    plt.xlabel("t [kyr]")
    plt.tight_layout()
    plt.legend()
    plt.savefig(args.save_path + "luminosity_ratio_example.png", dpi=300)

    plt.figure(5, dpi=300)
    plt.title("SMBH accretion")
    plt.plot(
        plot_time,
        bh_mdot_arr * unt.UnitMass_in_g / unt.MS / unt.UnitTime_in_s * unt.year,
        label="$\dot{M}$",
        alpha=0.5,
        c="r",
    )
    plt.ylabel("$\dot{M}_{\\rm{BH}}$")
    # plt.yscale("log")
    plt.xlabel("t [kyr]")
    plt.tight_layout()
    plt.legend()
    plt.savefig(args.save_path + "SMBH_accretion_example.png", dpi=300)
else:
    print("not plotting")
