from OBE2 import *
from OBE3 import *
from OBE4 import *
from QuantumJump import *

#initial init_state
psi0  = np.zeros(9)
psi0[0] = 1.0

'''
#initiate project1 Stimulated Raman transition
project = OpticalBlochEquation3(Omega1=8.86,Omega2=8.71,Delta=31.16,delta=0.021,Gamma1=0,Gamma2=0,GammaC=0,tmax=2.66,init_state=psi0)

#Calculating it's trajectory
project.Trajectory()

#graph potting
project.makePlot(population=True,susceptibility=False,rabi=False,savefig='test.png')

#pyvista plotting
project.makePyvista(rows=3,columns=4,off_screen=False)


#density matrix visualization 
project.DMVis(rows=3,columns=4,t_index=500,Full_Visualize=False,Interactive=False,savefig='testxx.png')
'''


####

'''

#initial project2 STIRAP
project2 = STIRAP(Omega1=2.51,Omega2=2.51,Delta=0.00,delta=0.00,t1=32.25,t2=20.4,tau1=5.91,tau2=5.91,tmax=53.95,init_state=psi0)

#Calculating it's trajectory
project2.Trajectory()

#graph potting
project2.makePlot(population=True, rabi=True,savefig=['poptest.png','stirap.png','rabitest.png'])

#pyvista plotting
project2.makePyvista()

#density matrix visualization 
#project2.DMVis(rows=3,columns=4,t_index=500,Full_Visualize=False, Interactive=False,savefig='dmvisStirap.png')  # need to edit more code

'''

###

'''
#initiate project3 EIT
project3 = EIT(Omega1=0.2,Omega2=0.5,Gamma1=1.0,Gamma2=0.5,delta=0.0,Delta=0.0,tmax=30.0,init_state=psi0,DeltaRange=5.0)

#Calculating it's trajectory
project3.Trajectory()

#Calculating it's Susceptibility
project3.Susceptibility()

#graph potting
project3.makePlot(population=True)

#graph potting
project3.makePlot(susceptibility=True)

#pyvista plotting
project3.makePyvista()

'''

###

'''
#initiate project4 Quantum JUMP
psi00 = (1/2)*np.array([1,1,np.sqrt(2)])

project4 = MCWF(Omega1=5,Omega2=6.36,Delta=0.11,delta=3.5,Gamma1=1.05,Gamma2=1.05,tmax=2.81,init_state=psi00,Nsamples=300)

project4.Trajectory()

project4.makePlot()

#project4.makePyvista()
'''

###

'''
init_psi = np.zeros(4)
init_psi[0] = 1.0 

#initiate project5 qubit dynamics
project5 = OpticalBlochEquation2(Omega=3,Delta=0,Gamma=0,tmax=2.66,init_state=init_psi)

project5.Trajectory()

project5.makePlot()

project5.makePyvista(rows=3,columns=4,off_screen=False,savefig='qubitPyvista.png')

project5.DMVis(rows=3,columns=4,t_index=500,Full_Visualize=False,Interactive=False)
'''

#
init_psi4 = np.zeros(16)
init_psi4[0] = 1.0 

#initiiate project6 4-level dynamics
project5 = OpticalBlochEquation4(Omega1=1,phi1=0,Omega2=1,phi2=0,Omega3=1,phi3=0,Omega4=1,phi4=0,Delta1=0.1,Delta2=0.1,Delta3=0.1,Gamma0=0,Gamma1=0,Gamma2=0,tmax=30,init_state=init_psi4)

project5.Trajectory()

project5.makePlot()

project5.DMVis(rows=3,columns=4,t_index=500,Full_Visualize=False,Interactive=True)

