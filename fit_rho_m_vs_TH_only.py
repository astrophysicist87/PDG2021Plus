import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy.integrate import quad
from scipy.optimize import curve_fit
from scipy.stats import t

mr = 0.5
minMass = 0.5
maxMass = float(sys.argv[3])
xdata = None
ydata = None

#----------------------------------------------------
def rho(mp, TH):
    return np.exp(mp/TH) / (mp**2+mr**2)**1.25

#----------------------------------------------------
def Ntheory(m, TH):
    # Check if 'm' is a single number or an array
    if np.ndim(m) == 0:
        unscaled = np.log(quad(rho, 0, m, args=(TH))[0])
        residuals = ydata - unscaled
        A = np.exp(np.mean(residuals))
        return np.log(A*quad(rho, 0, m, args=(TH))[0])
    
    # Iterate over the array if 'm' is a NumPy array/vector
    unscaled = np.log(np.array([quad(rho, 0, val, args=(TH))[0] for val in m]))
    residuals = ydata - unscaled
    A = np.exp(np.mean(residuals))
    return np.log(np.array([A*quad(rho, 0, val, args=(TH))[0] for val in m]))

#----------------------------------------------------
def Nexp(m, data):
    return np.sum( data[data[:,0] <= m,1] )

#----------------------------------------------------
# load data
data = np.loadtxt(sys.argv[1], comments="#", usecols=range(3,10))

# drop first row (Gamma)
data = data[1:]

particleRows = np.sum(np.abs(data[:,3:]),axis=-1)>0
data[particleRows,1] *= 2

dataFull = data

# drop heavy resonances
data = data[data[:,0] <= maxMass]

# adjust maximum mass if necessary (since it doesn't make sense to fit beyond this)
maxMass = np.min( [np.max(data[:,0]), maxMass] )

# sort by mass
data = data[data[:,0].argsort()]

# only fit at physical hadron masses
# vs. sample uniformly (only if supplied dx > 0);
# if using physical masses, don't overcount the same point in fit
dx = float(sys.argv[2])
xdata = np.linspace(minMass, maxMass, int((maxMass-minMass)/dx)+1) if dx > 0.0 \
        else np.unique(data[:,0])

# don't fit to hadron masses below 0.5 GeV
# (but still include them in the count)
xdata = xdata[xdata >= minMass]

# truncate fit range to [minMass, maxMass]
# (redundant for uniform sampling)
# N.B. - Nexp(...) *must* use full data array to get correct counts
xdata = xdata[(xdata >= minMass) & (xdata <= maxMass)]


ydata = np.log(np.array([Nexp(x, data) for x in xdata]))

popt, pcov = curve_fit(Ntheory, xdata, ydata, bounds=(0, 1.0), p0=[0.17])

#print(popt)

# Extract standard errors from the residuals-scaled covariance matrix
# The square root of the diagonal elements yields the standard error
perr = np.sqrt(np.diag(pcov))

# Calculate the Confidence Intervals using Student's t-distribution
confidence_level = 0.95
n_data = len(ydata)
n_parameters = len(popt)
degrees_of_freedom = n_data - n_parameters

# Compute the critical t-value for a two-tailed test
t_val = t.ppf((1 + confidence_level) / 2.0, df=degrees_of_freedom)

# Calculate upper and lower bounds
ci_lower = popt - t_val * perr
ci_upper = popt + t_val * perr

# Display results
print(f"  - Using [minMass, maxMass] = [{minMass:.4f}, {maxMass:.4f}] ({n_data})")

param_names = ['TH']
for i, name in enumerate(param_names):
    print(f"  - Parameter '{name}':")
    print(f"    Fitted Value: {1000.0 * popt[i]:.4f} +/- {1000.0 * t_val * perr[i]:.4f}")
    #print(f"    Standard Error: {perr[i]:.4f}")
    #print(f"    95% CI: [{ci_lower[i]:.4f}, {ci_upper[i]:.4f}]\n")


