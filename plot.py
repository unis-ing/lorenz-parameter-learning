import glob, json
import h5py as h5
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.cm import get_cmap
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist, squareform

import alphashape
from descartes import PolygonPatch as PolygonPatch

# ==========================================
#              colormap helpers
# ==========================================

# source: https://stackoverflow.com/a/20528097
def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.
    '''
    cdict = {'red': [], 'green': [], 'blue': [], 'alpha': []}

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False), 
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    new_cmap = colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=new_cmap)

    return new_cmap


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    '''
    Truncate cmap to [minval, maxval]
    '''
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))

    return new_cmap


# ==========================================
#        compute period of trajectory
# ==========================================

def get_period(XU, RHO):
    '''
    Outputs
        period           : i.e. winding number
        poincare_section : points in Poincare section (3,n)
        unique           : unique points in the Poincare section
    '''

    poincare_section = get_poincare_section(XU, RHO)

    if poincare_section.size == 0:
        return 0, 0, 0
    else:
        period, idx = compute_period(poincare_section)
        unique      = poincare_section[idx]
        return period, poincare_section, unique

def get_poincare_section(XU, RHO):
    '''
    Returns with shape (n, 2).

    Inputs
        X : shape (3, n)
    '''
    eps = 0.12 # max distance
    z = XU[2]

    return XU[:2, abs(z-(RHO-1)) < eps].T


def compute_period(pts):
    """
        pts : shape = (n,2)
    """
    if pts.shape[0] > 5000:
        return np.inf, np.arange(pts.shape[0])

    D = pdist(pts) # distance vector
    D = squareform(D) # convert to distance matrix
    R = np.nanmax(D) # max distance

    # min distance for two points to be counted separately
    eps = 0.01 * R

    idx = [0] # indices of unique points
    for i in range(1, D.shape[0]):
        distances = D[i, idx]
        is_unique = (distances > eps).all()
        if is_unique:
            idx.append(i)

    period = len(idx) / 2
    if len(idx) % 2 == 1:
        period += 1 / 2

    return period, idx


# ==================================================================
#                       get data for plots
# ==================================================================

def get_plot_data(folder_path):
    '''
    Inputs
        folder_path : folder path (str); should end in '/'

    Outputs
        params  : (n, 3) array with parameters
        errs    : (n, 3) array with base 10 log errors
        periods : (n) array with winding number of solutions
        spirals : (m, 3) array with parameters corresp. to spirals

    '''
    params  = [] # shape (n, 3)
    errs    = [] # shape (n, 3)
    periods = [] # shape (n)
    spirals = [] # shape (m, 3); params corres. to spirals

    search = folder_path + '*'
    for path in glob.glob(search):
        with open(path + '/params.json') as f:
            d = json.load(f)
            SIGMA = d['SIGMA']
            RHO = d['RHO']
            BETA = d['BETA']

        P = [SIGMA, RHO, BETA]
        params.append(P) # store params

        f = h5.File(path + '/data.h5', 'r')
        XU = _, _, z, _, _, _ = np.array(f['XU'])
        G = np.array(f['G'])
        f.close()

        E = abs(P - G[ : , -1])
        E[E == 0] = np.nan
        E = np.log10(E)
        E[np.isnan(E)] = -np.inf
        errs.append(E) # store err

        if np.mean(abs(z - (RHO - 1))[-1000:]) <= 1e-4:
            spirals.append(P) # store spiral params

        pd, _, _ = get_period(XU, RHO)
        periods.append(pd) # store period 

    params  = np.array(params)
    errs    = np.array(errs)
    periods = np.array(periods)
    spirals = np.array(spirals)

    return params, errs, periods, spirals


# ==================================================================
#                  1-param (sigma) plot helpers
# ==================================================================

def make_one_param_plot(params, errs, periods, spirals,
                        params_t, errs_t, periods_t, spirals_t,
                        save, save_t):

    plot_errs = errs[:,0]
    plot_errs_t = errs_t[:,0]

    plot_s = params[:,0]; plot_r = params[:,1]
    plot_s_t = params_t[:,0]; plot_r_t = params_t[:,1]

    spiral_s = spirals[:,0]; spiral_r = spirals[:,1]

    # ======================================
    #         plot regular data
    # ======================================

    # make the centered, discrete colormap
    cmap = get_cmap('seismic', 8)
    MAX = max(plot_errs)
    MIN = min(plot_errs_t[plot_errs_t != -np.inf])

    # shift colormap to specified center
    center = 1
    midpoint = (center - MIN) / (MAX - MIN)
    if midpoint >= 0:
        cmap = shiftedColorMap(cmap, midpoint=midpoint, name='shifted')

    # make figure
    fig, ax = plt.subplots(figsize=(6.5,5))

    # plot the critical rho curve
    foo = SIGMAs = np.linspace(1+8/3+0.001, max(plot_s), 5000)
    BETA = 8/3
    critical_rho = SIGMAs*(SIGMAs+BETA+3)/(SIGMAs-BETA-1)

    # truncate
    max_rho = max(plot_r)
    idx = np.where((0 <= critical_rho) & (critical_rho <= max_rho))
    critical_rho = critical_rho[idx]
    foo = foo[idx]

    ax.plot(critical_rho, foo, 'k--', linewidth=1, label=r'$\rho_c$', zorder=2)

    # fill left of critical rho
    bar = np.linspace(min(plot_r), max(plot_r))
    ax.fill_betweenx(foo, critical_rho, 0, color='gainsboro', zorder=-1)
    ax.fill_between(bar, 0, 1+8/3,  color='gainsboro', zorder=-1) # shade below
    ax.fill_between(bar, 140, max(plot_s),  color='gainsboro', zorder=-1) # shade ablove

    # ==================================================
    # plot spiral region 1
    ind = np.where(spiral_s >= (5/3*(spiral_r-8)-20))
    points = np.vstack((spiral_r[ind], spiral_s[ind])).T
    hull = ConvexHull(points)
    ax.fill(points[hull.vertices,0], points[hull.vertices,1], alpha=0.1, color='fuchsia', hatch='----', zorder=1,
            label='Stable spiral')

    # plot spiral region 2
    ind = np.where(spiral_s <= (5/3*(spiral_r-8)-20))
    points = np.vstack((spiral_r[ind], spiral_s[ind])).T
    hull = ConvexHull(points)
    ax.fill(points[hull.vertices,0], points[hull.vertices,1], alpha=0.1, color='fuchsia', hatch='----', zorder=1)

    # plot low period region 1
    ind = np.where((plot_r >= 5) & (plot_s > 20) & (np.array(periods) <= 8))[0]
    points = np.vstack((plot_r[ind], plot_s[ind])).T
    hull = ConvexHull(points)
    ax.fill(points[hull.vertices,0], points[hull.vertices,1], alpha=0.1, color='dodgerblue', hatch='||||', zorder=1,
             label=r'Period $\leq 8$')

    # plot low period region 2
    ind = np.where((plot_r >= 5) &  (plot_s > 0) & (plot_s <= 20) & (np.array(periods) <= 8))[0]
    points = np.vstack((plot_r[ind], plot_s[ind])).T
    hull = ConvexHull(points)
    ax.fill(points[hull.vertices,0], points[hull.vertices,1], alpha=0.1, color='dodgerblue', hatch='||||', zorder=1)

    # =================================================================
    # insert artificial, low error data point for colorbar formatting

    # plot points w update
    idx = np.where(plot_errs != 1)
    sc = ax.scatter(np.append([plot_r[0]], plot_r[idx]), 
                    np.append([plot_s[0]], plot_s[idx]), 
                    c=np.append([MIN], plot_errs[idx]), cmap=cmap, 
                    edgecolor='k', s=50, alpha=0.9, linewidth=0.75, zorder=2)

    # plot points w no update
    idx = np.where(plot_errs == 1)
    ax.scatter(plot_r[idx], plot_s[idx], color='white', edgecolor='k', s=50, 
               alpha=0.9, linewidth=0.75, zorder=2)

    plt.colorbar(sc, label=r'$\log_{10}|\tilde{\sigma}-\sigma|$')
    plt.xlabel(r'Rayleigh $\rho$')
    plt.ylabel(r'Prandtl $\sigma$')
    plt.legend(loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.3))

    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=300)
    plt.show()

    # ======================================
    #         plot translated data
    # ======================================

    # use the same colormap

    # make figure
    fig, ax = plt.subplots(figsize=(6.5,5))

    # ------------------ copy from prev plot --------------------
    # plot the critical rho curve
    foo = SIGMAs = np.linspace(1+8/3+0.001, max(plot_s), 5000)
    BETA = 8/3
    critical_rho = SIGMAs*(SIGMAs+BETA+3)/(SIGMAs-BETA-1)

    # truncate
    max_rho = max(plot_r)
    idx = np.where((0 <= critical_rho) & (critical_rho <= max_rho))
    critical_rho = critical_rho[idx]
    foo = foo[idx]

    ax.plot(critical_rho, foo, 'k--', linewidth=1, label=r'$\rho_c$', zorder=2)

    # fill left of critical rho
    bar = np.linspace(min(plot_r), max(plot_r))
    ax.fill_betweenx(foo, critical_rho, 0, color='gainsboro', zorder=-1)
    ax.fill_between(bar, 0, 1+8/3,  color='gainsboro', zorder=-1) # shade below
    ax.fill_between(bar, 140, max(plot_s),  color='gainsboro', zorder=-1) # shade ablove

    # ------------------ copy regions from prev plot --------------------
    # plot spiral region 1
    ind = np.where(spiral_s >= (5/3*(spiral_r-8)-20))
    points = np.vstack((spiral_r[ind], spiral_s[ind])).T
    hull = ConvexHull(points)
    ax.fill(points[hull.vertices,0], points[hull.vertices,1], alpha=0.1, color='fuchsia', hatch='----', zorder=1,
            label='Stable spiral')

    # plot spiral region 2
    ind = np.where(spiral_s <= (5/3*(spiral_r-8)-20))
    points = np.vstack((spiral_r[ind], spiral_s[ind])).T
    hull = ConvexHull(points)
    ax.fill(points[hull.vertices,0], points[hull.vertices,1], alpha=0.1, color='fuchsia', hatch='----', zorder=1)

    # plot low period region 1
    ind = np.where((plot_r >= 5) & (plot_s > 20) & (np.array(periods) <= 8))[0]
    points = np.vstack((plot_r[ind], plot_s[ind])).T
    hull = ConvexHull(points)
    ax.fill(points[hull.vertices,0], points[hull.vertices,1], alpha=0.1, color='dodgerblue', hatch='||||', zorder=1,
             label=r'Period $\leq 8$')

    # plot low period region 2
    ind = np.where((plot_r >= 5) &  (plot_s > 0) & (plot_s <= 20) & (np.array(periods) <= 8))[0]
    points = np.vstack((plot_r[ind], plot_s[ind])).T
    hull = ConvexHull(points)
    ax.fill(points[hull.vertices,0], points[hull.vertices,1], alpha=0.1, color='dodgerblue', hatch='||||', zorder=1)

    # ==================================================================
    # insert artificial, high error data point for colorbar formatting

    # plot points w update
    idx = np.where(plot_errs_t != -np.inf)
    sc = ax.scatter(np.append([plot_r_t[0]], plot_r_t[idx]), 
                    np.append([plot_s_t[0]], plot_s_t[idx]), 
                    c=np.append([MAX], plot_errs_t[idx]), cmap=cmap, 
                    edgecolor='k', s=50, alpha=0.9, linewidth=0.75, zorder=2)

    # plot points w 0 error
    idx = np.where(plot_errs_t == -np.inf)
    ax.scatter(plot_r_t[idx], plot_s_t[idx], marker='s', color='white', edgecolor='k', s=40, 
               alpha=0.9, linewidth=0.75, zorder=2,
               label=r'Machine $\varepsilon$')

    plt.colorbar(sc, label=r'$\log_{10}|\tilde{\sigma}-\sigma|$')
    plt.xlabel(r'Rayleigh $\rho$')
    plt.ylabel(r'Prandtl $\sigma$')
    plt.legend(loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.3))

    plt.tight_layout()
    if save_t:
        plt.savefig(save_t, dpi=300)


# ==================================================================
#                       2-param plot helpers
# ==================================================================

def make_two_param_plot(inferred_params, err_param, 
                        params, errs, periods, spirals, 
                        MIN=None, MAX=None, LOW_PD = 3,
                        LEGEND_ON=True, save=None):
    '''
    The axes conventions are: (rho, sigma), (sigma, beta) and (rho, beta)

    Inputs
        inferred_params : string indicating the two axes; should be one of
                          'RS', 'SB', or 'RB'.
        err_param       : string indicating which parameter error(s) will be used to 
                          color the points. Should be either single or double parameter.
        params  : (n, 3) array with parameters
        errs    : (n, 3) array with base 10 log errors
        periods : (n) array with winding number of solutions
        spirals : (m, 3) array with parameters corresp. to spirals
        center : value on which colorbar is centered.
        MIN    : min value on colorbar. if None, use max from errs.
        MAX    : max value on colorbar. if None, use min from errs.
        LOW_PD :
        LEGEND_ON : toggle legend on/off
        save : 
    '''
    assert inferred_params in ['RS', 'SB', 'RB'], 'Needs to be one of "RS", "SB", or "RB".'

    s1 = inferred_params[0]
    s2 = inferred_params[1]
    se = err_param

    p_to_i = {'S':0, 'R': 1, 'B':2}
    p_to_str = {'S': r'$\sigma$', 
                'R': r'$\rho$', 
                'B': r'$\beta$',
                'RS': r'$\Delta\sigma+\Delta\rho$',
                'RB': r'$\Delta\rho+\Delta\beta$',
                'SB': r'$\Delta\sigma+\Delta\beta$'}
    p_to_label = {'S': r'Prandtl $\sigma$', 
                  'R': r'Rayleigh $\rho$', 
                  'B': r'Physical parameter $\beta$'}
    p_to_err_label = {'S': r'$\log_{10}|\tilde{\sigma}-\sigma|$', 
                      'R': r'$\log_{10}|\tilde{\rho}-\rho|$', 
                      'B': r'$\log_{10}|\tilde{\beta}-\beta|$',
                      'RS': r'$\log_{10}(|\Delta\sigma|+|\Delta\rho|)$',
                      'RB': r'$\log_{10}(|\Delta\rho|+|\Delta\beta|)$',
                      'SB': r'$\log_{10}(|\Delta\sigma|+|\Delta\beta|)$'}

    i1 = p_to_i[s1] # index corresp. to first param
    i2 = p_to_i[s2] # index corresp. to second param
    
    plot_p1 = params[:,i1]
    plot_p2 = params[:,i2]
    plot_spirals = spirals[:,[i1,i2]]

    if len(se) == 1:
        center = 1
        ie = p_to_i[se]
        plot_errs = errs[:,ie]

    elif len(se) == 2:
        center = np.log10(20)
        ie1 = p_to_i[se[0]]
        ie2 = p_to_i[se[1]]
        plot_errs = np.log10(10**errs[:,ie1] + 10**errs[:,ie2])

    # make the centered, discrete colormap
    cmap = get_cmap('seismic', 8)
    if MAX == None:
        MAX = max(plot_errs)
    if MIN == None:
        MIN = min(plot_errs)

    # shift colormap to specified center
    midpoint = (center - MIN) / (MAX - MIN)
    if midpoint >= 0:
        cmap = shiftedColorMap(cmap, midpoint=midpoint, name='shifted')

    # =====================
    # make the figure
    # =====================
    if LEGEND_ON:
        fig, ax = plt.subplots(figsize=(6.5,5))
    else:
        fig, ax = plt.subplots(figsize=(6.5,4.5))

    if len(se) == 1:
        plt.title(r'Estimating ({}, {}) (colored by {})'.format(p_to_str[s1], 
                                                                p_to_str[s2],
                                                                p_to_str[se]))
    elif len(se) == 2:
        plt.title(r'Estimating ({}, {})'.format(p_to_str[s1], p_to_str[s2]))

    # plot the critical rho curve
    if 'R' in inferred_params:
        if 'S' in inferred_params:
            foo = SIGMAs = np.linspace(1+8/3+0.001, max(plot_p2), 5000)
            BETA   = 8/3 # dangerous assumption :3c
            critical_rho = SIGMAs*(SIGMAs+BETA+3)/(SIGMAs-BETA-1)

        elif 'B' in inferred_params:
            foo = BETAs = np.linspace(min(plot_p2), max(plot_p2))
            SIGMA = 10 # dangerous assumption #2 :3c
            critical_rho = SIGMA*(SIGMA+BETAs+3)/(SIGMA-BETAs-1)

        # truncate
        max_rho = max(plot_p1)
        idx = ((0 <= critical_rho) & (critical_rho <= max_rho))
        critical_rho = critical_rho[idx]
        foo = foo[idx]

        ax.plot(critical_rho, foo, 'k--', linewidth=1, label=r'$\rho_c$', zorder=2)

        # fill left of critical rho
        if 'S' in inferred_params: 
            bar = np.linspace(min(plot_p1), max(plot_p1))
            ax.fill_betweenx(foo, critical_rho, 0, color='gainsboro', zorder=-1)
            ax.fill_between(bar, 0, 1+8/3,  color='gainsboro', zorder=-1) # shade below
            ax.fill_between(bar, 140, max(plot_p2),  color='gainsboro', zorder=-1) # shade ablove

        elif 'B' in inferred_params:
            bar = np.linspace(min(plot_p1), max(plot_p1))
            ax.fill_betweenx(foo, critical_rho, 0, color='gainsboro', zorder=-1)
            ax.fill_between(bar, 7, max(plot_p2),  color='gainsboro', zorder=-1) # shade ablove

    # ===================================================================
    # compute regions --- this may need to be manually adjusted each time

    if inferred_params == 'RS':
        spiral_s, spiral_r, _ = spirals.T

        # spiral region 1
        idx = np.where(spiral_s >= (5/3*(spiral_r - 8) - 20))
        pts = np.vstack((spiral_r[idx], spiral_s[idx])).T
        hull = ConvexHull(pts)

        ax.fill(pts[hull.vertices,0], pts[hull.vertices,1], 
                alpha=0.1, color='fuchsia', hatch='----', zorder=1,
                label='Stable spiral')

        # spiral region 2
        idx = np.where(spiral_s <= (5/3*(spiral_r - 8) - 20))
        pts = np.vstack((spiral_r[idx], spiral_s[idx])).T
        hull = ConvexHull(pts)

        ax.fill(pts[hull.vertices,0], pts[hull.vertices,1], 
                alpha=0.1, color='fuchsia', hatch='----', zorder=1)

        # low period region 1
        idx = np.where((plot_p1 >= 20) & (plot_p2 > 20) & (np.array(periods) <= LOW_PD))
        pts = np.vstack((plot_p1[idx], plot_p2[idx])).T
        hull = ConvexHull(pts)

        ax.fill(pts[hull.vertices,0], pts[hull.vertices,1], 
                alpha=0.1, color='dodgerblue', hatch='||||', zorder=1,
                label=r'Period $\leq ' + str(LOW_PD) + '$')

        # low period region 2 -- region is too narrow to see on plot
        idx = np.where((plot_p1 >= 5) & (plot_p2 <= 20) & (np.array(periods) <= LOW_PD))
        pts = np.vstack((plot_p1[idx], plot_p2[idx])).T
        hull = ConvexHull(pts)

        ax.fill(pts[hull.vertices,0], pts[hull.vertices,1], 
                alpha=0.1, color='dodgerblue', hatch='||||', zorder=1)

    elif inferred_params == 'RB':

        # spiral region 1
        _, spiral_r, spiral_b = spirals.T
        pts = np.vstack((spiral_r, spiral_b)).T

        # use alphashape to compute something between a convex/concave hull
        hull = alphashape.alphashape(pts, 0.005)
        ax.add_patch(PolygonPatch(hull, fill=True, color='fuchsia', hatch='----', alpha=0.1,
                                  label='Stable spiral'))

        # lower period region 1
        idx = np.where((plot_p1 >= 100) & (plot_p2 > 1) & (np.array(periods) <= LOW_PD))
        pts = np.vstack((plot_p1[idx], plot_p2[idx])).T
        hull = ConvexHull(pts)

        ax.fill(pts[hull.vertices,0], pts[hull.vertices,1], 
                alpha=0.1, color='dodgerblue', hatch='||||', zorder=1,
                label=r'Period $\leq ' + str(LOW_PD) + '$')

    elif inferred_params == 'SB':

        # spiral region 1
        spiral_s, _, spiral_b = spirals.T
        pts = np.vstack((spiral_s, spiral_b)).T

        # use alphashape to compute something between a convex/concave hull
        hull = alphashape.alphashape(pts, 0.01)
        ax.add_patch(PolygonPatch(hull, fill=True, color='fuchsia', hatch='----', alpha=0.1,
                                  label='Stable spiral'))

        # lower period region 1
        idx = np.where((plot_p2 < 10) & (0 < np.array(periods)) & (np.array(periods) <= LOW_PD))
        pts = np.vstack((plot_p1[idx], plot_p2[idx])).T
        hull = ConvexHull(pts)

        ax.fill(pts[hull.vertices,0], pts[hull.vertices,1], 
                alpha=0.1, color='dodgerblue', hatch='||||', zorder=1,
                label=r'Period $\leq ' + str(LOW_PD) + '$')

    # ===============================
    #       make scatter plot
    # ===============================

    # insert artificial points with MIN/MAX error values
    d1 = np.append([plot_p1[0], plot_p1[0]], plot_p1)
    d2 = np.append([plot_p2[0], plot_p2[0]], plot_p2)
    e  = np.append([MIN, MAX], plot_errs)

    # plot points w update
    idx = np.where(e != np.log10(20))
    sc = ax.scatter(d1[idx], d2[idx], c=e[idx], cmap=cmap, 
                    edgecolor='k', s=50, alpha=0.9, linewidth=0.75, zorder=2)

    # plot points w no update
    idx = np.where(e == np.log10(20))
    ax.scatter(d1[idx], d2[idx], color='white', edgecolor='k', s=50, 
               alpha=0.9, linewidth=0.75, zorder=2)

    # plot points w 0 error
    idx = np.where(e == -np.inf)
    ax.scatter(d1[idx], d2[idx], marker='s', color='white', edgecolor='k', s=50, 
               alpha=0.9, linewidth=0.75, zorder=2,
               label='Machine precision')

    plt.colorbar(sc, label=p_to_err_label[se])
    plt.xlabel(p_to_label[s1])
    plt.ylabel(p_to_label[s2])

    if LEGEND_ON:
        plt.legend(loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.3))

    plt.tight_layout()

    if save:
        plt.savefig(save, dpi=300)
    plt.show()
