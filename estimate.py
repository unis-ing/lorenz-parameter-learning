import numpy as np
import time
from scipy.integrate import odeint as odeint


def estimate(lorenz_params, aot_params, mu_params, dt, T, Tr):
	"""
	Arguments:
		lorenz_params -- reference parameters [SIGMA, RHO, BETA]
		aot_params    -- initial parameter estimates [sigma, rho, beta]
		mu_params     -- nudging parameters [mu1, mu2, mu3]
		dt            -- temporal resolution of solution
		T             -- simulation length
		Tr            -- relaxation period

	Outputs:
	    t   -- array of times on which the solution was evaluated
		G   -- array of parameter estimates; shape=(3,n)
		XU  -- solution of coupled Lorenz + nudged system; shape=(6,n)
		XUt -- time derivatives evaluated on t; shape=(6,n)
	"""

	# initialize coupled system
	sol = odeint(func=lorenz,  y0=[60, 60, 10], 
		   		 t=np.arange(0, 5, 1e-3), args=(lorenz_params,))
	X0  = sol[-1]
	U0  = [0.1, 0.1, 0.1]
	XU0 = np.concatenate((X0, U0))

	# =======================================================
	# ------------------ initialize -------------------------
	t = np.arange(0, stop=T, step=dt)
	mu_params = np.array(mu_params)
	G = np.array([aot_params])
	E = np.array([abs(U0 - X0)])
	args = [G, E, 0, 0, 0]

	# =======================================================
	# ---------------- run algorithm ------------------------
	start = time.time()
	sol, infodict = odeint(lorenz_aot, XU0, t, full_output=True,
	                       args=(lorenz_params, mu_params, Tr, args))
	# store solution
	XU = sol.T

	try:
		# reshape parameters
		idx = np.insert(infodict['nfe'], 0, 0) # start indexing from 0
		G   = args[0][idx].T

		# calculate derivatives
		N = int(T / dt)
		mu_params_tiled = np.tile(np.expand_dims(mu_params,1), N)
		X   = XU[:3]
		Xt  = lorenz_system(X, lorenz_params)
		Ut  = aot_system(XU, G, mu_params_tiled)
		XUt = np.vstack((Xt, Ut))
		
		return t, G, XU, XUt

	except IndexError:
		print('Indexing error. Parameter estimates not saved.')
		return t, [aot_params], XU, [[0,0,0]]


# ============================================
# ----------------- equations ----------------
# ============================================

def lorenz_system(X, lorenz_params):
    x, y, z = X
    S, R, B = lorenz_params

    return [S*(y - x),
            R*x - y - x*z,
            x*y - B*z]


def aot_system(XU, aot_params, mu_params):
    x, y, z, u, v, w = XU
    s, r, b = aot_params
    mu1, mu2, mu3 = mu_params

    return [s*(v - u) - mu1*(u - x), 
            r*u - v - u*w - mu2*(v - y), 
            u*v - b*w - mu3*(w - z)]

# ============================================
# ------------ odeint functions  -------------
# ============================================

def lorenz(X, t, lorenz_params):
	'''
	Pass to odeint to solve the Lorenz equations.
	'''
	return lorenz_system(X, lorenz_params)


def lorenz_aot(XU, t, lorenz_params, mu_params, Tr, args):
    '''
    Pass to odeint to solve coupled Lorenz + AOT systems.
    
    Arguments:
        lorenz_params -- list or numpy array of 3 Lorenz parameters
        nudge_params  -- nudging parameters [mu1, mu2, mu3]
        Tr            -- minimum time between parameter updates
        args          -- all other arguments
            [0] -- numpy array of parameters estimates; shape=(n,3)
            [1] -- numpy array of position errors; shape=(n,3)
            [2] -- time at last function call
            [3] -- time elapsed since last update
            [4] -- index of last parameter update
    '''
    x, y, z, u, v, w = XU
    X = XU[:3]; U = XU[3:]

    # unpack parameters
    s, r, b = aot_params = args[0][-1]
    mu1, mu2, mu3 = mu_params
    
    # store current position errs
    err = abs(X - U)
    args[1] = E = np.vstack((args[1], err))

    # update the time variables
    args[3] += t - args[2]
    args[2] =  t

    # initialize next parameter estimates
    s_new, r_new, b_new = s, r, b

    if (err > 0).all(): # stop condition

        if (u != v) and args[3] >= Tr: # enforce waiting period
            i_curr = E.shape[0] - 1 # current index
            i_last = args[4]
            E_curr = E[i_last:].T # shape = (3,n)

            # compute linear fit of position errors for each component
            thresholds = np.empty(3)
            fit_x = np.arange(i_curr-i_last+1)

            for j in range(3):
                if mu_params[j] != 0:
                    with np.errstate(divide='ignore'):
                        fit_y = np.log(E_curr[j])

                    M, B = np.polyfit(fit_x, fit_y, 1)
                    theta = np.exp(M * fit_x[-1] + B)
                    thresholds[j] = theta
                else:
                    thresholds[j] = np.inf

            with np.errstate(invalid='ignore'):
                THRESHOLDS_MET = (err <= thresholds).all()

            if THRESHOLDS_MET:
                args[3] = 0 # reset time_since
                args[4] = i_curr # update i_last

                # update guesses
                s_ = s - mu1 * (u-x) / (v-u)
                r_ = r - mu2 * (v-y) / u
                b_ = b + mu3 * (w-z) / w
                if s_ > 0: s_new = s_
                if r_ > 0: r_new = r_
                if b_ > 0: b_new = b_
    
    # update guesses
    args[0] = np.vstack((args[0], [s_new, r_new, b_new]))
    
    # calculate derivatives
    lorenz_dt = lorenz_system(X, lorenz_params)
    aot_dt    = aot_system(XU, aot_params, mu_params)
    
    return np.concatenate((lorenz_dt, aot_dt))