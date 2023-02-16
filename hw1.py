# import packages
import numpy as np
import matplotlib.pyplot as plt
import plotting
from matplotlib.lines import Line2D

##########################
#### DEFINE FUNCTIONS #### 
##########################

# define a function for the acceleration using Newton's equations
def accel(x_i, perturb):

    """
    Calculate the acceleration between each object.
    Should return a numpy array.
    
    """

    # big G baby
    G = 39.478

    # get M in solar masses
    M_Sun = 1.
    M_Uranus = 43.66244e-6
    M_Neptune = 51.51389e-6

    # get the magnitude of the distance between each object
    dist_Sun_to_Uranus = np.linalg.norm(x_i[0] - x_i[1])
    dist_Sun_to_Neptune = np.linalg.norm(x_i[0] - x_i[2])
    dist_Uranus_to_Neptune = np.linalg.norm(x_i[1] - x_i[2])

    # get rhat between each object
    # my planets were fucking YEETING out of the solar system
    # but a wonderful little negative sign changed all that :)
    rhat_Sun_to_Uranus = -1*(x_i[0] - x_i[1]) / dist_Sun_to_Uranus
    rhat_Sun_to_Neptune = -1*(x_i[0] - x_i[2]) / dist_Sun_to_Neptune
    rhat_Uranus_to_Neptune = -1*(x_i[1] - x_i[2]) / dist_Uranus_to_Neptune

    # do we want Uranus perturbed by the Sun AND Neptune?
    if perturb == True: 
        # calculate the acceleration and add them together
        accelS = np.array([0, 0, 0])  # we don't give af about the sun!!!!!!!!
        accelU = ((-1*(G * M_Sun) / dist_Sun_to_Uranus**2) * rhat_Sun_to_Uranus) + ((-1*(G * M_Neptune) / dist_Uranus_to_Neptune**2) * rhat_Uranus_to_Neptune)
        accelN = ((-1*(G * M_Sun) / dist_Sun_to_Neptune**2) * rhat_Sun_to_Neptune) + ((-1*(G * M_Uranus) / dist_Uranus_to_Neptune**2) * -1*rhat_Uranus_to_Neptune)
        # the negative in -1*rhat_Uranus_to_Neptune in the last line above is bc we want the vector to go from Neptune to Uranus this time

    # if Uranus isn't perturbed by Neptune, then what will its orbit look like
    # if the only force is from the Sun?
    elif perturb == False:
        accelS = np.array([0, 0, 0])  # sun
        accelU = ((-1*(G * M_Sun) / dist_Sun_to_Uranus**2) * rhat_Sun_to_Uranus)
        accelN = np.array([0, 0, 0])

    # give me my accelerations!!!!!
    a = np.array([accelS, accelU, accelN])

    return a

# define a function that will spit out the initial
# positions and velocities of Neptune, Uranus, and the Sun
def get_xi_vi():
    """
    set up the initial positions and velocities
    of the Sun, Uranus, and Neptune.
    Taken from https://ssd.jpl.nasa.gov/horizons/app.html#/
    
    """

    # Sun
    S_x = -1.346352997255123E+06  # km
    S_y = -4.593198959279832E+04
    S_z = 3.173863158251134E+04
    Sun_pos = np.array([S_x, S_y, S_z])

    S_vx = 2.612301607260599E-03  # km/s
    S_vy = -1.533663676257939E-02
    S_vz = 6.362653892795240E-05
    Sun_vel = np.array([0, 0, 0])

    # Uranus
    U_x = 1.979031148286366E+09  # km
    U_y = 2.174943998266760E+09
    U_z = -1.756090803339267E+07
    Uranus_pos = np.array([U_x, U_y, U_z])

    U_vx = -5.086868282285313E+00  # km/s
    U_vy = 4.265947851062992E+00
    U_vz = 8.178760840892996E-02
    Uranus_vel = np.array([U_vx, U_vy, U_vz])

    # Neptune
    N_x = 4.452697226397768E+09 # km
    N_y = -4.190641405509241E+08
    N_z = -9.398680344155586E+07
    Neptune_pos = np.array([N_x, N_y, N_z])

    N_vx = 4.733080793921328E-01  # km/s
    N_vy = 5.443593445927669E+00
    N_vz = -1.230051082192813E-01
    Neptune_vel = np.array([N_vx, N_vy, N_vz])

    # convert from km to AU and km/s to AU/yr
    km_to_AU = 6.685e-9
    km_per_s_to_AU_per_yr = 0.2108

    # initialize the xi and vi arrays
    x_i = np.array([Sun_pos, Uranus_pos, Neptune_pos]) * km_to_AU
    v_i = np.array([Sun_vel, Uranus_vel, Neptune_vel]) * km_per_s_to_AU_per_yr

    return x_i, v_i

# create function for an N-body integrator
# using the leapfrog method
# specifically the 4th order Yoshida
# integrator
def YoshidaIntegrator(x_i, v_i, dt, perturb):

    """"
    4th order N-body integrator using the leapfrog Yoshida method.

    x_i and v_i are starting position and velocity
    xn_i and vn_i are position and velocity at step n
    x4_i and v4_i are the final positions and velocities
    
    """

    # c1, c2, c3, and c4 are coefficients
    # d1, d2, d3, and d4 are coefficients
    # w0, w1 are in the equations for the coefficients
    # https://en.wikipedia.org/wiki/Leapfrog_integration
    w0 = -2**(-1/3) / (2 - 2**(-1/3))
    w1 = 1 / (2 - 2**(-1/3))

    c1 = w1 / 2
    c4 = c1
    c2 = (w0 + w1) / 2
    c3 = c2
    d1 = w1
    d3 = d1
    d2 = w0

    # equations for the 4th order Yoshida integrator
    # these update position and velocity
    # https://en.wikipedia.org/wiki/Leapfrog_integration
    x1_i = x_i + c1 * v_i * dt
    v1_i = v_i + d1 * accel(x1_i, perturb) * dt
    x2_i = x1_i + c2 * v1_i * dt
    v2_i = v1_i + d2 * accel(x2_i, perturb) * dt
    x3_i = x2_i + c3 * v2_i * dt
    v3_i = v2_i + d3 * accel(x3_i, perturb) * dt
    x4_i = x3_i + c4 * v3_i * dt
    v4_i = v3_i

    return x4_i, v4_i

# define a function that does the integration
def Nbody(tmin, tmax, dt, x_i, v_i, perturb):

    t = tmin  # set the first t
    i = 0  # set the first iteration

    # lets save Uranus's positions and velocities
    xs = []
    vs = []
    time = []

    # do the integration up to t = tmax
    while(t < tmax):

        # do the integration
        x_i, v_i = YoshidaIntegrator(x_i, v_i, dt, perturb)

        # update the time step!
        t += dt
        i += 1
        xs.append(x_i[1])
        vs.append(v_i[1])

        time.append(t)

    # make things into arrays so we don't get yelled at    
    xs = np.array(xs)
    vs = np.array(vs)
    time = np.array(time)
    return xs, vs, time

# main function
if __name__ == "__main__":

    # define variables
    dt = 0.01
    tmin = 0.
    tmax = 100

    # grab the first positions and velocities
    x_i, v_i = get_xi_vi()

    # do the integration!
    xs_perturb, _, time = Nbody(tmin, tmax, dt, x_i, v_i, perturb = True)
    xs_unperturb, _, _ = Nbody(tmin, tmax, dt, x_i, v_i, perturb = False)
    
    # calculate the maximum difference between Uranus perturbed and Uranus unperturbed
    difference = [np.linalg.norm(xs_perturb[i] - xs_unperturb[i]) for i in range(len(xs_perturb))]
    index = np.argmax(difference)
    theta = (difference[index]) / np.linalg.norm(xs_perturb[index])  # get that in arcsec

    # plot plot plot plot plot
    plt.figure(figsize=(7,7))
    plt.plot(time, difference, color='tab:cyan')
    plt.xlabel('time [years]')
    plt.ylabel("difference in orbit of Uranus [AU]")
    
    custom_lines = [Line2D([0], [0], color='white', lw=2),
                    Line2D([0], [0], color='white', lw=2)]

    plt.legend(custom_lines,['Max. difference in AU: %s' % round(difference[index],2), 
                             'Max. difference in arcsec: %s' % round(theta * 206265,2)], 
                             fontsize=10, loc='upper left')
    plt.savefig('hw1.pdf', dpi=200)
    plt.show()