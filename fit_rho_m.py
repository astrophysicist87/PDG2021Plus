import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy.integrate import quad
from scipy.optimize import curve_fit
from scipy.stats import t

mr = 0.5
minMass = 0.5
maxMass = float(sys.argv[3])

#----------------------------------------------------
def rho(mp, A, TH):
    return A*np.exp(mp/TH) / (mp**2+mr**2)**1.25

#----------------------------------------------------
def drho_dTH(mp, A, TH):
    return (-mp/TH**2) * A * np.exp(mp/TH) / (mp**2+mr**2)**1.25

#----------------------------------------------------
def Ntheory(m, A, TH):
    # Check if 'm' is a single number or an array
    if np.ndim(m) == 0:
        return np.log(quad(rho, 0, m, args=(A, TH))[0])
    
    # Iterate over the array if 'm' is a NumPy array/vector
    return np.log(np.array([quad(rho, 0, val, args=(A, TH))[0] for val in m]))

#----------------------------------------------------
def d_Ntheory_dA(m, A, TH):
    return 1./A

#----------------------------------------------------
def d_Ntheory_dTH(m, A, TH):
    # Check if 'm' is a single number or an array
    if np.ndim(m) == 0:
        return quad(drho_dTH, 0, m, args=(A, TH))[0] \
                / quad(rho, 0, m, args=(A, TH))[0]
    
    # Iterate over the array if 'm' is a NumPy array/vector
    return np.array([quad(drho_dTH, 0, val, args=(A, TH))[0] \
                        /quad(rho, 0, val, args=(A, TH))[0] for val in m])
    

#----------------------------------------------------
def Ntheory_jacobian(m, A, TH):
    # Check if 'm' is a single number or an array
    if np.ndim(m) == 0:
        # Set derivatives
        df_dA  = d_Ntheory_dA(m, A, TH)
        df_dTH = d_Ntheory_dTH(m, A, TH)
        # Combine columns into an (N, K) matrix
        return np.column_stack([df_dA, df_dTH])
    
    # Set derivatives
    df_dA  = np.array([d_Ntheory_dA(val, A, TH) for val in m])
    df_dTH = np.array([d_Ntheory_dTH(val, A, TH) for val in m])
    # Combine columns into an (N, K) matrix
    return np.column_stack([df_dA, df_dTH])
    

#----------------------------------------------------
def Nexp(m, data):
    return np.sum(data[data[:,0]<=m,1])

#----------------------------------------------------
# load data
data = np.loadtxt(sys.argv[1], comments="#", usecols=range(3,10))

# drop any massless particles (e.g., Gamma)
data = data[data[:,0] > 0]

# adjust degeneracies to account for antiparticles
particleRows = np.sum(np.abs(data[:,3:]),axis=-1)>0
data[particleRows,1] *= 2

# adjust maximum mass if necessary (since it doesn't make sense to fit beyond this)
maxMass = np.min( [np.max(data[:,0]), maxMass] )

# sort by mass (probably already sorted anyway)
data = data[data[:,0].argsort()]

# only fit at physical hadron masses
# vs. sample uniformly
#xdata = np.unique(data[:,0])
#xdata = np.linspace(minMass, maxMass, int(np.round((maxMass-minMass)/0.01))+1)
dx = float(sys.argv[2])
xdata = np.linspace(minMass, maxMass, int((maxMass-minMass)/dx)+1) if dx > 0.0 \
        else np.unique(data[:,0])


# truncate fit range to [minMass, maxMass]
# (redundant for uniform sampling)
# N.B. - Nexp(...) *must* use full data array to get correct counts
xdata = xdata[(xdata >= minMass) & (xdata <= maxMass)]
ydata = np.log(np.array([Nexp(x, data) for x in xdata]))

#np.set_printoptions(threshold=np.inf)
#print(np.c_[xdata,np.exp(ydata)])

popt, pcov = curve_fit(Ntheory, xdata, ydata, bounds=(0, [10.0, 1.0]), p0=[0.5, 0.17])
#popt, pcov = curve_fit( Ntheory, xdata, ydata, bounds=(0, [10.0, 1.0]), \
#                        method='dogbox', jac=Ntheory_jacobian, p0=[0.5, 0.17])


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

print(f"  - Using [minMass, maxMass] = [{minMass:.4f}, {maxMass:.4f}] ({n_data})")

# 6. Display results
param_names = ['A', 'TH']
for i, name in enumerate(param_names):
    if name == 'A':
        continue
    print(f"  - Parameter '{name}':")
    print(f"    Fitted Value: {1000.0 * popt[i]:.4f} +/- {1000.0 * t_val * perr[i]:.4f}")
    #print(f"    Standard Error: {perr[i]:.4f}")
    #print(f"    95% CI: [{ci_lower[i]:.4f}, {ci_upper[i]:.4f}]\n")


#plotResults = False
#if plotResults:
#    fig = plt.figure(figsize=(4,6), dpi=125)
#    ax = fig.add_subplot(111)
#    
#    xdata = np.linspace(np.amin(data[:,0]),np.amax(data[:,0]),10001)
#    ydata = np.log(np.array([Nexp(x, data) for x in xdata]))
#    plt.plot(xdata, np.exp(ydata), color='blue')
#    plt.plot(xdata, np.exp(Ntheory(xdata,*popt)), color='black')
#    
#    plt.xlim(0.5,3.0)
#    plt.yscale('log')
#    
#    plt.show()


