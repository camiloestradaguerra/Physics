import numpy as np
from odeintw import odeintw
import matplotlib.pyplot as plt

##############
# PARAMETERS #
##############
omega_0 = 1
Omega = 1
gamma = 0.2

#######################################
# RIGHT SIDE OF THE LINDBLAD EQUATION #
#######################################

# For phenomenological Lindblad Equation
def right_part_phen(rho, t):
    a = np.array([[0,0,0],[0,0,0],[1,0,0]])
    ad = np.array([[0,0,1],[0,0,0],[0,0,0]])
    hjc = np.array([[0.5*omega_0, Omega, 0], 
                    [Omega, 0.5*omega_0, 0], 
                    [0, 0, -0.5*omega_0]], 
                   dtype=np.complex128)
    
    return -(1j)*(np.dot(hjc, rho)-np.dot(rho, hjc))+gamma*(np.dot(a, np.dot(rho, ad))-0.5*np.dot(ad, np.dot(a, rho))-0.5*np.dot(rho, np.dot(ad, a)))

# For the true Jaynes-Cummings Lindblad Equation
def right_part(rho, t):
    e1p_k = np.array([[1],[0],[0]])
    e1p_b = np.array([[1, 0, 0]])
    e1m_k = np.array([[0],[1],[0]])
    e1m_b = np.array([[0, 1, 0]])
    e0_k = np.array([[0],[0],[1]])
    e0_b = np.array([[0, 0, 1]])
    hjc = np.array([[0.5*omega_0+Omega, 0, 0], 
                    [0, 0.5*omega_0-Omega, 0], 
                    [0, 0, -0.5*omega_0]], 
                   dtype=np.complex128)
    term1 = gamma*(0.5*np.dot(e0_k, np.dot(e1p_b, np.dot(rho, np.dot(e1p_k, e0_b))))-0.25*np.dot(e1p_k, np.dot(e1p_b, rho))-0.25*np.dot(rho, np.dot(e1p_k, e1p_b)))
    term2 = gamma*(0.5*np.dot(e0_k, np.dot(e1m_b, np.dot(rho, np.dot(e1m_k, e0_b))))-0.25*np.dot(e1m_k, np.dot(e1m_b, rho))-0.25*np.dot(rho, np.dot(e1m_k, e1m_b)))
    return term1+term2

######################
# Initial conditions #
######################
psi_init_phen = np.array([[0.0, 0.0, 0.0],
                     [0.0, 1.0, 0.0],
                     [0.0, 0.0, 0.0]], dtype=np.complex128)

psi_init = np.array([[0.5, 0.0, 0.0],
                     [0.0, 0.5, 0.0],
                     [0.0, 0.0, 0.0]], dtype=np.complex128)

###################
# Temporal scales #
###################
t_5 = np.linspace(0, 5, 101)
t_50 = np.linspace(0, 50, 1001)

######################################
# Solutions of the Lindblad Equation #
######################################
sol_phen_5 = odeintw(right_part_phen, psi_init_phen, t_5)
sol_phen_50 = odeintw(right_part_phen, psi_init_phen, t_50)
sol_5 = odeintw(right_part, psi_init, t_5)
sol_50 = odeintw(right_part, psi_init, t_50)

##########################################################################
# Getting the population elements of the reduced density matrix solution #
##########################################################################
p11_phen_5 = np.array([])
for i in range(0,len(sol_phen_5)):
    p11_phen_5 =np.append(p11_phen_5, sol_phen_5[i,0,0])

p11_phen_50 = np.array([])
for i in range(0,len(sol_phen_50)):
    p11_phen_50 =np.append(p11_phen_50, sol_phen_50[i,0,0])    

p22_phen_5 = np.array([])
for i in range(0,len(sol_phen_5)):
    p22_phen_5 =np.append(p22_phen_5, sol_phen_5[i,1,1])

p22_phen_50 = np.array([])
for i in range(0,len(sol_phen_50)):
    p22_phen_50 =np.append(p22_phen_50, sol_phen_50[i,1,1])    

p33_phen_5 = np.array([])
for i in range(0,len(sol_phen_5)):
    p33_phen_5 =np.append(p33_phen_5, sol_phen_5[i,2,2])    

p33_phen_50 = np.array([])
for i in range(0,len(sol_phen_50)):
    p33_phen_50 =np.append(p33_phen_50, sol_phen_50[i,2,2]) 
   
p11_5 = np.array([])
for i in range(0,len(sol_5)):
    p11_5 =np.append(p11_5, sol_5[i,0,0])

p11_50 = np.array([])
for i in range(0,len(sol_50)):
    p11_50 =np.append(p11_50, sol_50[i,0,0])
    
p22_5 = np.array([])
for i in range(0,len(sol_5)):
    p22_5 =np.append(p22_5, sol_5[i,1,1])

p22_50 = np.array([])
for i in range(0,len(sol_50)):
    p22_50 =np.append(p22_50, sol_50[i,1,1])    

p33_5 = np.array([])
for i in range(0,len(sol_5)):
    p33_5 =np.append(p33_5, sol_5[i,2,2])   

p33_50 = np.array([])
for i in range(0,len(sol_50)):
    p33_50 =np.append(p33_50, sol_50[i,2,2])      

#########
# Plots #
#########

#plt.plot(2*t, p33_phen, '-')
#plt.plot(2*t, p33, '-')
#plt.xlabel(r'\tau')
#plt.ylabel(r'P_{0,g}')



fig, ax = plt.subplots()

ax.plot(2*t_5, p33_phen_5, '-', label=r'$\rho_{0,g}^{ph}$')
ax.plot(2*t_5, p33_5, '--', label=r'$\rho_{0,g}$')
ax.set_xlabel(r'$\tau$')
ax.set_ylabel(r'$P_{0,g}$')
ax.legend(loc=0)
ax.set_xticks(np.arange(0, 10, 2))
ax.set_yticks(np.arange(0, 0.5, 0.1))

axIns = ax.inset_axes([0.57, 0.12, 0.4, 0.4])
axIns.plot(2*t_50, p33_phen_50, '-', label=r'$\rho_{0,g}^{ph}$')
axIns.plot(2*t_50, p33_50, '--', label=r'$\rho_{0,g}$')
axIns.set_xticks(np.arange(0, 110, 20))
axIns.set_yticks(np.arange(0, 1.1, 0.2))
#ax1.set_xlim([-0.1, 10.1])
#ax1.set_ylim([-0.01, 0.41])

#ax2 = plt.axes([0,0,1,1])
# Manually set the position and relative size of the inset axes within ax1
#ip = InsetPosition(ax1, [0.4,0.2,0.5,0.5])
#ax2.set_axes_locator(ip)
# Mark the region corresponding to the inset axes on ax1 and draw lines
# in grey linking the two axes.
#mark_inset(ax1, ax2, loc1=2, loc2=4, fc="none", ec='0.5')

# The data: only display for low temperature in the inset figure.
#Tmax = max(T_D)
#ax2.plot(2*t, , 'x', c='b', mew=2, alpha=0.8,
#         label='Experiment')
# The Einstein fit (not very good at low T).
#ax2.plot(T_E[T_E<=Tmax], CV_E[T_E<=Tmax], c='m', lw=2, alpha=0.5,
#         label='Einstein model')
# The Debye fit.
#ax2.plot(T_D, CV_D, c='r', lw=2, alpha=0.5, label='Debye model')
#ax2.legend(loc=0)
fig.savefig('plot1.png', dpi=300, bbox_inches='tight')

plt.show()

