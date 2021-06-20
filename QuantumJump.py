import numpy as np
from numpy import pi
import matplotlib
import matplotlib.pyplot as plt
import pyvista as pv
from pyvista import examples
from scipy.linalg import kron, eig
import time
import matplotlib.colors as mcolors
matplotlib.rcParams['text.usetex'] = True
from string import ascii_lowercase

def Expt(psi,Op):
    return np.real(np.conj(psi.T).dot(Op.dot(psi)))

class MCWF:
    def __init__(self,Omega1,Omega2,Delta,delta,Gamma1,Gamma2,tmax,init_state,Nsamples):
        self.Omega1 = Omega1
        self.Omega2 = Omega2
        self.Delta = Delta
        self.delta = delta
        self.Gamma1 = Gamma1
        self.Gamma2 = Gamma2
        self.tmax = tmax
        self.init_state = init_state #array type
        self.Nsamples = Nsamples
        self.I3 = np.eye(3)
        self.radius = 3.0
        #define time parameter
        self.numt = 1000
        self.time = np.linspace(0,self.tmax,self.numt)
        self.dt = self.time[1]-self.time[0]
        #define quantum state and sigma matrice (array tyoe)
        self.g1 = np.array([1,0,0])
        self.g2 = np.array([0,1,0])
        self.e = np.array([0,0,1])
        self.sigma1 = np.outer(self.g1,np.conj(self.e.T))
        self.sigma2 = np.outer(self.g2,np.conj(self.e.T))
        self.sigmax = np.array([[0,1,0],[1,0,0],[0,0,0]])
        self.sigmay = np.array([[0,-1j,0],[1j,0,0],[0,0,0]])
        self.sigmaz = np.array([[1,0,0],[0,-1,0],[0,0,0]])
        self.sigma0 = np.array([[0,0,0],[0,0,0],[0,0,1]])
        #define Hamiltonian and effective non Hermitian Hamiltonain (array type)
        self.H =  self.Delta*np.outer(self.g1,np.conj(self.g1.T)) + (self.Delta-self.delta)*np.outer(self.g2,np.conj(self.g2.T)) + 0.5*self.Omega1*(self.sigma1 + np.conj(self.sigma1.T)) + 0.5*self.Omega2*(self.sigma2 + np.conj(self.sigma2.T))
        #define JUMP operators, effective non-Hermitian Hamiltonain and no jump operator (array(type))
        self.C1 = np.sqrt(self.Gamma1)*self.sigma1
        self.OpC1 = np.conj(self.C1.T).dot(self.C1)
        self.C2 = np.sqrt(self.Gamma2)*self.sigma2
        self.OpC2 = np.conj(self.C2.T).dot(self.C2)
        self.nH_eff = self.H - (0.5j)*(np.conj(self.C1.T).dot(self.C1) + np.conj(self.C2.T).dot(self.C2))
        self.C0 = self.I3 - (1j*self.nH_eff*self.dt)
        self.OpC0 = np.conj(self.C0.T).dot(self.C0)
        #define state
        self.state = np.zeros((self.Nsamples,3))
        self.densitymatrixAvr = np.zeros((self.numt,3,3))


    def Initialise(self):
        self.state[:,:] = self.init_state
        self.bloch1 = np.zeros(3)  # Calculated from average density matrix
        self.bloch2 = np.zeros(3)  # Calculated from average density matrix
        self.prob_e = np.zeros((self.Nsamples,self.numt))
        self.prob_g1 = np.zeros((self.Nsamples,self.numt))
        self.prob_g2 = np.zeros((self.Nsamples,self.numt))
        self.prob_eAvr = np.zeros(self.numt)
        self.prob_g1Avr = np.zeros(self.numt)
        self.prob_g2Avr = np.zeros(self.numt)

    def dp1(self,j):
        return np.real(self.dt*Expt(self.state[j,:],self.OpC1))

    def dp2(self,j):
        return np.real(self.dt*Expt(self.state[j,:],self.OpC2))

    def saveTrajectory(self,i,j):
        pg1 = self.state[j,:][0]
        pg2 = self.state[j,:][1]
        pe = self.state[j,:][2]
        self.prob_g1[j,i] = np.conj(pg1)*pg1
        self.prob_g2[j,i] = np.conj(pg2)*pg2
        self.prob_e[j,i] = np.conj(pe)*pe
        self.densitymatrixAvr[i,:,:] += (1/self.Nsamples)*np.outer(self.state[j,:],np.conj(self.state[j,:].T))


    def getNextState(self,j,g1=False,g2=False):
        dp = np.real(self.dp1(j) + self.dp2(j))
        #norm = np.real(self.dt*Expt(self.state[j,:],self.OpC0))
        if(g1):
            self.state[j,:] = np.sqrt(self.dt/self.dp1(j))*self.C1.dot(self.state[j,:])
        elif(g2):
            self.state[j,:] = np.sqrt(self.dt/self.dp2(j))*self.C2.dot(self.state[j,:])
        else:
            self.state[j,:] = (1/np.sqrt(1-dp))*self.C0.dot(self.state[j,:])
            #self.state[j,:] = (1/np.sqrt(norm))*self.C0.dot(self.state[j,:])


    def Trajectory(self):
        self.Initialise()
        for i in range(self.numt):
            for j in range(self.Nsamples):
                dp = self.dp1(j) + self.dp2(j)
                if(np.random.rand()>dp): # no jump
                    self.saveTrajectory(i,j)
                    self.getNextState(j)
                else:
                    if(np.random.rand()<(self.dp1(j)/dp)): #jump to g1
                        self.saveTrajectory(i,j)
                        self.getNextState(j,g1=True)
                    else: #jump to g2
                        self.saveTrajectory(i,j)
                        self.getNextState(j,g2=True)

            #calculating BLoch vector
            DM = self.densitymatrixAvr[i,:,:]
            u1 = np.real(np.trace(self.sigmax*DM))
            v1 = np.real(np.trace(self.sigmay*DM))
            w1 = np.real(np.trace(self.sigmaz*DM))
            #bloch2
            r2 = np.real(np.trace(self.sigma0*DM))
            if(r2==1):
                u2 = r2*np.cos(0.5*pi*r2)
                v2 = r2*np.cos(0.5*pi*r2)
                w2 = r2*np.sin(0.5*pi*r2)
            else:
                u2 = r2*np.cos(0.5*pi*r2)*(np.real(DM[0,2])/np.abs(DM[0,2])) if np.abs(DM[0,2])!=0 else 0
                v2 = r2*np.cos(0.5*pi*r2)*(np.imag(DM[0,2])/np.abs(DM[0,2])) if np.abs(DM[0,2])!=0 else 0
                w2 = r2*np.sin(0.5*pi*r2) if np.abs(DM[0,2])!=0 else 0
            #update vector
            b1array = np.array([u1,v1,w1])
            b2array = np.array([u2,v2,w2])
            self.bloch1 = np.vstack((self.bloch1,b1array))
            self.bloch2 = np.vstack((self.bloch2,b2array))

            #averaging probability over samples
            self.prob_eAvr[i] = DM[2,2]
            self.prob_g1Avr[i] = DM[0,0]
            self.prob_g2Avr[i] = DM[1,1]

        self.bloch1 = self.radius*self.bloch1
        self.bloch2 = self.radius*self.bloch2
        self.bloch1 = np.delete(self.bloch1,0,0)
        self.bloch2 = np.delete(self.bloch2,0,0)

    def makePlot(self):
        lw1 = 0.8
        fig,ax = plt.subplots(3,1,figsize=(6,12))
        fig.suptitle('Monte Carlo Wave-function', fontsize=16)
        #plotting e prob
        for i in range(self.Nsamples):
            ax[0].plot(self.time,self.prob_e[i,:],c='m',lw=lw1)
            if(i == self.Nsamples-1): ax[0].plot(self.time,self.prob_e[i,:],c='m',lw=lw1,label='MCWF')
        ax[0].plot(self.time,self.prob_eAvr,c='b',label=r'Averaged probability in $|e\rangle$')
        ax[0].set_xlabel(r'Time (t)')
        ax[0].set_ylabel(r'Population')
        ax[0].legend()
        #plotting g1 prob
        for i in range(self.Nsamples):
            ax[1].plot(self.time,self.prob_g1[i,:],c='m',lw=lw1)
            if(i == self.Nsamples-1): ax[1].plot(self.time,self.prob_g1[i,:],c='m',lw=lw1,label='MCWF')
        ax[1].plot(self.time,self.prob_g1Avr,c='b',label=r'Averaged probability in $|g_1\rangle$')
        ax[1].set_xlabel(r'Time (t)')
        ax[1].set_ylabel(r'Population')
        ax[1].legend(loc=4)
        #plotting g2 prob
        for i in range(self.Nsamples):
            ax[2].plot(self.time,self.prob_g2[i,:],c='m',lw=lw1)
            if(i == self.Nsamples-1): ax[2].plot(self.time,self.prob_g2[i,:],c='m',lw=lw1,label='MCWF')
        ax[2].plot(self.time,self.prob_g2Avr,c='b',label=r'Averaged probability in $|g_2\rangle$')
        ax[2].set_xlabel(r'Time (t)')
        ax[2].set_ylabel(r'Population')
        ax[2].legend(loc=4)
        plt.show()


    def makePyvista(self):
        #set colour
        dr=[190.0/255.0,30.0/255.0,45.0/255.0]
        dy=[255.0/255.0,213.0/255.0,58.0/255.0]
        dg=[175.0/255.0,169.0/255.0,97.0/255.0] # Durham green
        db=[0,174.0/255.0,239.0/255.0]
        dp=[104.0/255.0,36.0/255.0,109.0/255.0]
        di=[0.0/255.0,42.0/255.0,65.0/255.0] # Durham ink
        dpi=[203.0/255.0,168.0/255.0,177.0/255.0] #  Durham pink
        ds=[218.0/255.0,205.0/255.0,162.0/255.0] # Durham stone
        dsk=[165.0/255.0,200.0/255.0,208.0/255.0] # Durham sky
        ### Generate Bloch sphere by using pv.Spline for Stimulate Raman Transition with decay process
        num=50
        theta = np.linspace(-1 * np.pi, 1 * np.pi, num)
        r=3.0
        phi=0*np.pi/60
        ###xy axis
        z = 0*r * np.cos(theta)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        rpts=np.column_stack((x, y, z))
        spline = pv.Spline(rpts, 1000)
        rxy_tube=spline.tube(radius=0.05)
        ###xz axis
        z = r * np.cos(theta)
        x = r * np.sin(theta)*np.cos(phi-np.pi/2)
        y = r * np.sin(theta)*np.sin(phi-np.pi/2)
        rpts=np.column_stack((x, y, z))
        spline = pv.Spline(rpts, 1000)
        rxz_tube=spline.tube(radius=0.05)
        ###yz axis
        z = r * np.cos(theta)
        x = r * np.sin(theta)*np.cos(phi)
        y = r * np.sin(theta)*np.sin(phi)
        rpts=np.column_stack((x, y, z))
        spline = pv.Spline(rpts, 1000)
        ryz_tube=spline.tube(radius=0.05)
        ### Plot spline on Bloch sphere
        small=pv.Sphere(center=(0, 0, r), radius=0.2)
        big=pv.Sphere(center=(0, 0, 0), radius=r)
        sargs = dict(height=0.25, vertical=True, position_x=0.05, position_y=0.05)
        res=3
        ###Generate time for multi plots
        numt = 1000
        nx=4
        ny=3
        tt = np.linspace(0,numt,nx*ny+1)
        ### Spline plots
        #Get trajectory
        traj1 = self.bloch1
        traj2 = self.bloch2
        #begin pyvista plot
        p = pv.Plotter(shape=(ny,nx), multi_samples=1, window_size=(res*900,res*600))
        p.set_background(dsk, top="white")
        '''UFO = pv.Light(position=(10, 10, 10), focal_point=(0, 0, 0), color='white')
        UFO.positional = True
        UFO.cone_angle = 40
        UFO.exponent = 1
        UFO.intensity = 0.5
        UFO.show_actor()
        p.add_light(UFO)
        '''
        k=1
        for i in range(0,ny):
            for j in range(0,nx):
                p.subplot(i,j)
                #p.add_mesh(small, opacity=1.0, color=dr, smooth_shading=True)
                p.add_mesh(big, opacity=0.4, color="w", specular=0.85, smooth_shading=True)
                #p.add_mesh(tube,smooth_shading=True,color=dpi)
                #p.add_mesh(tube,smooth_shading=True,scalar_bar_args=sargs)
                label=ascii_lowercase[(4*i+j)]
                #p.add_text('$\\vert \\psi\\rangle$',[180.0*res,165.0*res],color=di,font_size=res*14)
                p.add_text(label,[10.0*res,165.0*res],color=di,font_size=res*14)
                p.add_mesh(rxy_tube,opacity=0.1,smooth_shading=True,color=di)
                p.add_mesh(rxz_tube,opacity=0.1,smooth_shading=True,color=di)
                p.add_mesh(ryz_tube,opacity=0.1,smooth_shading=True,color=di)
                # Spline1 for vector1
                points1 = traj1[0:int(tt[k]),:]
                spline1 = pv.Spline(points1, 1000)
                spline1["scalars"] = np.arange(spline1.n_points)
                tubes1=spline1.tube(radius=0.1)
                #Sline2 for vector2
                points2 = traj2[0:int(tt[k]),:]
                spline2 = pv.Spline(points2, 1000)
                spline2["scalars"] = np.arange(spline2.n_points)
                tubes2=spline2.tube(radius=0.1)
                #
                p.add_mesh(tubes1,color=dr,smooth_shading=True,show_scalar_bar=False)
                ept1=pv.Sphere(center=(points1[-1,:]), radius=0.2)
                p.add_mesh(ept1, opacity=1.0, color=dr, smooth_shading=True)
                arrow=pv.Arrow(start=(0.0, 0.0, 0.0), direction=(points1[-1,:]), tip_length=0.25, tip_radius=0.1, tip_resolution=20, shaft_radius=0.05, shaft_resolution=20, scale=np.sqrt(sum(points1[-1,:]**2)))
                p.add_mesh(arrow, opacity=1.0, color=db, smooth_shading=True)
                #
                p.add_mesh(tubes2,color=dy,smooth_shading=True,show_scalar_bar=False)
                ept2=pv.Sphere(center=(points2[-1,:]), radius=0.2)
                p.add_mesh(ept2, opacity=0.5, color=dy, smooth_shading=True)
                arrow=pv.Arrow(start=(0.0, 0.0, 0.0), direction=(points2[-1,:]), tip_length=0.25, tip_radius=0.1, tip_resolution=20, shaft_radius=0.05, shaft_resolution=20, scale=np.sqrt(sum(points2[-1,:]**2)))
                p.add_mesh(arrow, opacity=0.5, color=db, smooth_shading=True)

                k += 1
        #print(points[-1,:])
        p.enable_depth_peeling(10)
        p.link_views()
        p.camera_position = [(-8.5, 8.5, 3.0),
                             (0.0, 0.0, 0.0),
                             (0.0, 0.0, 0.1)]
        #p.camera_position = [(12.0, 0.0, 1.0),(0.0, 0.0, 0.0),(0.1, 0.0, 0.1)]
        #p.show(screenshot='StimRa_HighdeltaGamma.png')
        p.show()
