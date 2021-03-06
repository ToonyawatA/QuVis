import numpy as np
from numpy import pi
from pylab import *
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.widgets import Slider, Button, RadioButtons
import colorsys
import pyvista as pv
from pyvista import examples
from scipy.linalg import kron, eig
import time
matplotlib.rcParams['text.usetex'] = True
from string import ascii_lowercase



##: Durham colour scheme

cDUp = "#7E317B"  # Palatinate Purple
cDUpp =  "#D8ACF4"  # Light purple

cDUb = "#006388"  # Blue
cDUbb = "#91B8BD"  # Mid Blue
cDUbbb = "#C4E5FA"  # Light Blue
cDUbbbb = "#00AEEF"

cDUsky = "#A5C8D0"  # sky blue

cDUo = "#9FA161"  # Olive Green

cDUr = "#AA2B4A"  # Red
cDUrr = "#BE1E2D"
cDUy = "#E8E391" #  Yellow

cDUp = "#C43B8E" # Pink

cDUk = "#231F20"  # Black
cDUkk = "#002A41" # ink

cDUggg = "#CFDAD13"  # Near White/L. Grey
cDUgg = "#968E85"  # Warm Grey
cDUg = "#6E6464"  # midgrey

def make_colormap(seq):
    """
        Args:
            seq: a sequence of floats and RGB-tuples. The floats should be
                increasing and in the interval (0,1).

        Returns:
            a LinearSegmentedColormap
    """
    seq = [(None, ) * 3, 0.0] + list(seq) + [1.0, (None, ) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)


c = mcolors.ColorConverter().to_rgb
rvb = make_colormap([
    c('#b20000'),
    c('#fe7600'), 0.125,
    c('#fe7600'),
    c('#feca00'), 0.25,
    c('#feca00'),
    c('#bcfd00'), 0.375,
    c('#bcfd00'),
    c('#06a133'), 0.5,
    c('#06a133'),
    c('#00f6fd'), 0.625,
    c('#00f6fd'),
    c('#000cfe'), 0.75,
    c('#000cfe'),
    c('#e404fe'), 0.875,
    c('#e404fe'),
    c('#b20000')
])


def getColor(amplitude, phase, maxAmplitude):
    c = rvb(phase / (2. * np.pi))
    scale = amplitude / maxAmplitude
    if scale > 1:
        raise ValueError(
            'Amplitude of the passed complex number is bigger than the'
            ' maximal set amplitudeyter not')
    cc = colorsys.rgb_to_hls(c[0], c[1], c[2])
    c = colorsys.hls_to_rgb(cc[0], cc[1] + (1. - scale) * (1. - cc[1]), cc[2])
    return (c[0], c[1], c[2], 1.0)


def getComplexColor(complexNo, maxAmplitude):
    """
    Get color for a complex numbers

    Represents phase as continous colour wheel, and amplitude as intensity
    of color (zero amplitude = white color), with linear mapping in between.

    Args:
        complexNo (complex float): complex number
        maxAmplitude (float): maximum amplitude in the data set we want to
            represent as colour mapped dots. This is used for normalizing color
            intensity, going from maximal saturation or `maxAmplitude` to
            white color for zero amplitude.

    Returns:
        color as [red, green, blue, alpha]
    """
    angle = np.angle(complexNo)
    if angle < 0:
        angle += 2 * np.pi
    return getColor(np.absolute(complexNo), angle, maxAmplitude)

def white_to_transparency(img):
    """
        Converts white areas of image to transprancy.
    """
    x = np.asarray(img.convert('RGBA')).copy()

    x[:, :, 3] = (255 * (x[:, :, :3] != 255).any(axis=2)).astype(np.uint8)

    return Image.fromarray(x)

def RGBfunc(rho,t_index):
    time_dim=1
    parameter_dim=1
    mdim=3
    #Size of R,G,B
    h_dim = time_dim*(mdim+1)+1
    v_dim = parameter_dim*(mdim+1)+1
    R=0.85*np.ones((v_dim,h_dim))
    G=0.85*np.ones((v_dim,h_dim))
    B=0.85*np.ones((v_dim,h_dim))
    #begin plotting
    peak = np.amax(abs(rho))
    for col in range (0,mdim):
        for row in range (0,mdim):
            R[1+row,1+col]=getComplexColor(rho[col+3*row,t_index],peak)[0]
            G[1+row,1+col]=getComplexColor(rho[col+3*row,t_index],peak)[1]
            B[1+row,1+col]=getComplexColor(rho[col+3*row,t_index],peak)[2]

    return np.dstack((R, G, B))


def Dissipator(sigma):
    d = np.shape(sigma)[0]
    sigma = np.mat(sigma)
    I = np.eye(d)
    I = np.mat(I)
    return np.mat(kron(np.conj(sigma),sigma) - 0.5*kron(I,np.conj(sigma.T)*sigma) - 0.5*kron((np.conj(sigma.T)*sigma).T,I))  


class OpticalBlochEquation3:
    ### init ###
    def __init__(self,Omega1,Omega2,Delta,delta,Gamma1,Gamma2,GammaC,tmax,init_state,DeltaRange=None):
        self.Omega1 = Omega1
        self.Omega2 = Omega2
        self.Delta = Delta
        self.delta = delta
        self.Gamma1 = Gamma1
        self.Gamma2 = Gamma2
        self.GammaC = GammaC
        self.tmax = tmax
        self.Dmax = DeltaRange
        self.init_state = np.mat(init_state.reshape(9,1)) #matrix array
        #define radius
        self.radius = 3.0
        #define quantum state
        self.g1 = np.mat(np.array([1,0,0]))
        self.g2 = np.mat(np.array([0,1,0]))
        self.e = np.mat(np.array([0,0,1]))
        #define matrices
        self.sigma1 = self.g1.T*self.e
        self.sigma2 = self.g2.T*self.e
        self.sigmaC = self.g2.T*self.g2 - self.g1.T*self.g1
        self.sigmax = np.mat(np.array([[0,1,0],[1,0,0],[0,0,0]]))
        self.sigmay = np.mat(np.array([[0,-1j,0],[1j,0,0],[0,0,0]]))
        self.sigmaz = np.mat(np.array([[1,0,0],[0,-1,0],[0,0,0]]))
        self.sigma0 = np.mat(np.array([[0,0,0],[0,0,0],[0,0,1]]))
        self.I3 = np.mat(np.eye(3))
        #define time step and
        self.numt = 1000
        self.time = np.linspace(0,self.tmax,self.numt)
        self.dt = self.time[1]-self.time[0]

    def DensityMatrix(self):
        return self.state.reshape(3,3)

    def LBsuperoperator(self,Delta=None):
        if Delta is None:
            Delta = self.Delta
        
        ### Hamiltonian for three-level with two fields 
        H_a = Delta*self.g1.T*self.g1 + (Delta-self.delta)*self.g2.T*self.g2
        H_af = 0.5*self.Omega1*(self.sigma1 + np.conj(self.sigma1.T)) + 0.5*self.Omega2*(self.sigma2 + np.conj(self.sigma2.T))
        H = H_a + H_af
        #H = 0.5*np.array([[0,0,self.Omega1],[0,-2*Delta,self.Omega2],[np.conj(self.Omega1),np.conj(self.Omega2),-2*(Delta-self.delta)]])

        #define superoperator
        H_eff = -1j*np.mat(kron(self.I3,H) - kron(np.conj(H.T),self.I3))
        L_eff = self.Gamma1*Dissipator(self.sigma1) + self.Gamma2*Dissipator(self.sigma2) + self.GammaC*Dissipator(self.sigmaC)
        S = H_eff+L_eff
        return S

    def LBDiagonalise(self,Delta=None):
        '''
        #define Hamiltonian
        evals, evecs = eig(self.LBsuperoperator())
        evecs = np.mat(evecs)
        '''
        if Delta is None:
            Delta = self.Delta
        #diagonalize Hamiltonian
        evals, evecs = eig(self.LBsuperoperator(Delta))
        evecs = np.mat(evecs)
        return evals, evecs

    def getNextState(self,Delta=None):
        if Delta is None:
            Delta = self.Delta
        self.state = self.LBDiagonalise(Delta)[1]*np.mat(np.diag(np.exp(self.LBDiagonalise(Delta)[0]*self.dt)))*np.linalg.inv(self.LBDiagonalise(Delta)[1])*self.state

    def getState_at_t(self,Delta,t):
        return self.LBDiagonalise(Delta)[1]*np.mat(np.diag(np.exp(self.LBDiagonalise(Delta)[0]*t)))*np.linalg.inv(self.LBDiagonalise(Delta)[1])*self.init_state

    """
    def Initialise_state(self):
        self.state = self.init_state #matrix array
        self.state_arr = np.zeros(9)
    """

    def Initialise(self):
        #define bloch vector and probability array
        self.state = self.init_state #matrix array
        self.state_arr = np.zeros(9) #state array (9*1 array)
        self.bloch1 = np.zeros(3)
        self.bloch2 = np.zeros(3)
        self.probability = np.zeros(3)
    """
    def saveTrajectory_state(self):
        rho = np.squeeze(np.asarray((self.state)))
        self.state_arr = np.column_stack((self.state_arr,rho))
    """

    def saveTrajectory(self):
        #save state 9*1 array
        rho = np.squeeze(np.asarray((self.state)))
        self.state_arr = np.column_stack((self.state_arr,rho))
        #state probability
        pg1 = np.real(self.state[0,0])
        pg2 = np.real(self.state[4,0])
        pe = np.real(self.state[8,0])
        #density matrix
        DM = self.state.reshape(3,3)
        #bloch1
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
            u2 = r2*np.cos(0.5*pi*r2)*(np.real(self.state[2,0])/np.abs(self.state[2,0])) if np.abs(self.state[2,0])!=0 else 0
            v2 = r2*np.cos(0.5*pi*r2)*(np.imag(self.state[2,0])/np.abs(self.state[2,0])) if np.abs(self.state[2,0])!=0 else 0
            w2 = r2*np.sin(0.5*pi*r2) if np.abs(self.state[2,0])!=0 else 0
        parray = np.array([pg1,pg2,pe])
        b1array = np.array([u1,v1,w1])
        b2array = np.array([u2,v2,w2])
        self.bloch1 = np.vstack((self.bloch1,b1array))
        self.bloch2 = np.vstack((self.bloch2,b2array))
        self.probability = np.vstack((self.probability,parray))

    """
    def Trajectory_state(self,Delta=None):
        ###
        if Delta is None :
            Delta = self.Delta
        self.Initialise_state()
        for i in range(self.numt):
            self.saveTrajectory_state()
            self.getNextState(Delta)
        self.state_arr = np.delete(self.state_arr,0,1)
    """

    def Trajectory(self,Delta=None):
        ###
        if Delta is None :
            Delta = self.Delta
        self.Initialise()
        for i in range(self.numt):
            self.saveTrajectory()
            self.getNextState(Delta)
        self.bloch1 = self.radius*self.bloch1
        self.bloch2 = self.radius*self.bloch2
        self.bloch1 = np.delete(self.bloch1,0,0)
        self.bloch2 = np.delete(self.bloch2,0,0)
        self.probability = np.delete(self.probability,0,0)
        self.state_arr = np.delete(self.state_arr,0,1)

    def makePlot(self,population=False,susceptibility=False,rabi=False,savefig=None):
        poplabel = [r'$|g_1\rangle$',r'$|g_2\rangle$',r'$|e\rangle$']
        rabilabel = ['Pulse1','Pulse2']
        chilabel = ['Dispersion','Absorption']
        if(population):
            plt.figure()
            for i in range(3):
                plt.plot(self.time,self.probability[:,i],label=poplabel[i])
            plt.xlabel(r'Time ($t$)')
            plt.ylabel(r'Population')
            plt.legend()
            if savefig is not None:
                plt.savefig(savefig[0],dpi=120)
            plt.show()
        if(susceptibility):
            plt.figure()
            plt.plot(self.detuning,self.rhoba_r,label='Dispersion')
            plt.plot(self.detuning,self.rhoba_i,label='Absorption')
            plt.xlabel(r'Detuning ($\Delta$)')
            plt.ylabel(r'Susceptibility')
            plt.legend()
            if savefig is not None:
                plt.savefig(savefig[1],dpi=120)
            plt.show()
        if(rabi):
            plt.figure()
            plt.plot(self.time,self.omega1,label='Pulse1')
            plt.plot(self.time,self.omega2,label='Pulse2')
            plt.xlabel(r'Time ($t$)')
            plt.ylabel(r'Rabi Frequency')
            plt.legend()
            if savefig is not None:
                plt.savefig(savefig[2],dpi=120)
            plt.show()
     

    def makePyvista(self,t_index=None,rows=1,columns=1,off_screen=False,savefig=None):
        if t_index is None: 
            t_index = self.numt
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
        if(t_index>self.numt):
            raise ValueError('t_index must not exceed %d' %self.numt)
        else:
            tt_index = np.linspace(0,t_index,rows*columns+1)
        ### Spline plots
        #Get trajectory
        traj1 = self.bloch1
        traj2 = self.bloch2
        #begin pyvista plot
        p = pv.Plotter(off_screen,shape=(rows,columns), multi_samples=1, window_size=(res*900,res*600))
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
        for i in range(0,rows):
            for j in range(0,columns):
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
                points1 = traj1[0:int(tt_index[k]),:]
                spline1 = pv.Spline(points1, 1000)
                spline1["scalars"] = np.arange(spline1.n_points)
                tubes1=spline1.tube(radius=0.1)
                #Sline2 for vector2
                points2 = traj2[0:int(tt_index[k]),:]
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
        #p.show(screenshot='test1.png')
        if savefig is not None:
            p.show(screenshot=savefig)
        p.show()
        

    ####### Density matrix visualization #######

    def DMVis(self,DRange=5,rows=1,columns=1,t_index=None,Delta=None,delta=None,Full_Visualize=False,Interactive=False,savefig=None):
        t_index = self.numt if t_index is None else t_index
        Delta = self.Delta if Delta is None else Delta
        delta = self.delta if delta is None else delta
        #generate time for multiplots
        if(t_index>self.numt):
            raise ValueError('t_index must not exceed %d' %self.numt)
        else:
            tt_index = np.linspace(0,t_index,rows*columns+1)
        #define dimension of system
        mdim = 3
        #define color for text
        di=[0.0/255.0,42.0/255.0,65.0/255.0] # Durham ink

        ###for full visualization
        if(Full_Visualize):
            #define size of table
            time_dim = 40 #no. of times to plot across rows
            parameter_dim = 21 # no. of parameter values down columns
            #Delta array (detuning)
            Deltas = np.linspace(Delta-DRange,Delta+DRange,parameter_dim)
            #Size of R,G,B
            h_dim = time_dim*(mdim+1)+1
            v_dim = parameter_dim*(mdim+1)+1
            R=0.85*np.ones((v_dim,h_dim))
            G=0.85*np.ones((v_dim,h_dim))
            B=0.85*np.ones((v_dim,h_dim))
            #begin plotting
            fig, ax=plt.subplots(figsize=(16, 8))
            for v_index in range(0, parameter_dim): # v_index is no. of rows
                '''
                for i in range(npts):
                    rho_vec = self.getState_at_t(Deltas[v_index],t[i])
                    rho[:,i] = np.squeeze(np.asarray(rho_vec))
                    '''
                #self.Trajectory_state(Deltas[v_index])
                self.Trajectory(Deltas[v_index])
                rho = self.state_arr
                peak = np.amax(abs(rho))
                for h_index in range(0, time_dim): # h_index is no. of columns
                    t_index=10*h_index
                    for col in range (0,mdim):
                        for row in range (0,mdim):
                            R[(mdim+1)*v_index+1+row,(mdim+1)*h_index+1+col]=getComplexColor(rho[col+3*row,t_index],peak)[0]
                            G[(mdim+1)*v_index+1+row,(mdim+1)*h_index+1+col]=getComplexColor(rho[col+3*row,t_index],peak)[1]
                            B[(mdim+1)*v_index+1+row,(mdim+1)*h_index+1+col]=getComplexColor(rho[col+3*row,t_index],peak)[2]

            RGB=np.dstack((R, G, B))
            plt.imshow(RGB)
            ax.set_axis_off()
            if savefig is not None:
                plt.savefig(savefig,dpi=120)
            plt.show()

        else:
            #for slider at specific Delta
            if(Interactive):
                self.Trajectory(Delta)
                rho = self.state_arr
                ## plot
                fig, ax = plt.subplots(figsize=(10,10))
                img = ax.imshow(RGBfunc(rho,0))
                axcolor = 'lightgoldenrodyellow'
                ax_slider = plt.axes([0.20, 0.01, 0.65, 0.03], facecolor=axcolor)
                slider = Slider(ax_slider, label='Time', valmin=0.0, valmax=self.tmax, valinit=0, valfmt='%1.1f',valstep=0.05)
                def update(val):
                    ts_index = int((slider.val/self.tmax)*self.numt)
                    ax.imshow(RGBfunc(rho,ts_index))
                    fig.canvas.draw_idle()
                slider.on_changed(update)
                plt.show()
            #for full visualize at specific Delta
            else:
                #Size of R,G,B
                h_dim = (mdim+1)+1
                v_dim = (mdim+1)+1
                R=0.85*np.ones((v_dim,h_dim))
                G=0.85*np.ones((v_dim,h_dim))
                B=0.85*np.ones((v_dim,h_dim))
                #begin plotting
                fig, ax =plt.subplots(rows,columns,figsize=(16, 8))
                self.Trajectory(Delta)
                rho = self.state_arr
                peak = np.amax(abs(rho))
                k=1
                for v_index in range(0,rows):
                    for h_index in range(0,columns): # h_index is no. of columns
                        for col in range (0,mdim):
                            for row in range (0,mdim):
                                R[1+row,1+col]=getComplexColor(rho[col+3*row,int(tt_index[k])],peak)[0]
                                G[1+row,1+col]=getComplexColor(rho[col+3*row,int(tt_index[k])],peak)[1]
                                B[1+row,1+col]=getComplexColor(rho[col+3*row,int(tt_index[k])],peak)[2]
                        
                        RGB=np.dstack((R, G, B))
                        label=ascii_lowercase[(4*v_index+h_index)]
                        ax[v_index,h_index].imshow(RGB) if rows*columns!=1 else ax.imshow(RGB)
                        ax[v_index,h_index].text(0.05, 0.85,label,transform=ax[v_index,h_index].transAxes,color=di,fontsize=20)
                        ax[v_index,h_index].set_axis_off() if rows*columns!=1 else ax.set_axis_off()
                        k += 1
                if savefig is not None:
                    plt.savefig(savefig,dpi=120)
                plt.show()
                


###### STIRAP #########

class STIRAP(OpticalBlochEquation3):
    def __init__(self,Omega1,Omega2,Delta,delta,t1,t2,tau1,tau2,tmax,init_state,Gamma1=0,Gamma2=0,GammaC=0):
        OpticalBlochEquation3.__init__(self,Omega1,Omega2,Delta,delta,Gamma1,Gamma2,GammaC,tmax,init_state)
        self.t1 = t1
        self.t2 = t2
        self.tau1 = tau1
        self.tau2 = tau2
        self.omega1 = np.abs(np.real(self.Omega1*np.exp((-1/(2*self.tau1**2))*(self.time-self.t1)**2)))
        self.omega2 = np.abs(np.real(self.Omega2*np.exp((-1/(2*self.tau2**2))*(self.time-self.t2)**2)))

    def LBsuperoperator(self,omega1,omega2):
        H_a = self.Delta*self.g1.T*self.g1 + (self.Delta-self.delta)*self.g2.T*self.g2
        H_af = 0.5*omega1*(self.sigma1 + np.conj(self.sigma1.T)) + 0.5*omega2*(self.sigma2 + np.conj(self.sigma2.T))
        H = H_a + H_af
        #define superoperator
        H_eff = -1j*np.mat(kron(self.I3,H) - kron(np.conj(H.T),self.I3))
        L_eff = self.Gamma1*Dissipator(self.sigma1) + self.Gamma2*Dissipator(self.sigma2) + self.GammaC*Dissipator(self.sigmaC)
        S = H_eff+L_eff
        return S

    def LBDiagonalise(self,omega1,omega2):
        #define Hamiltonian
        evals, evecs = eig(self.LBsuperoperator(omega1,omega2))
        evecs = np.mat(evecs)
        return evals, evecs

    def getNextState(self,omega1,omega2):
        self.state = self.LBDiagonalise(omega1,omega2)[1]*np.mat(np.diag(np.exp(self.LBDiagonalise(omega1,omega2)[0]*self.dt)))*np.linalg.inv(self.LBDiagonalise(omega1,omega2)[1])*self.state

    def Trajectory(self):
        #define time array
        self.Initialise()
        for i in range(self.numt):
            self.saveTrajectory()
            self.getNextState(self.omega1[i],self.omega2[i])
        self.bloch1 = self.radius*self.bloch1
        self.bloch2 = self.radius*self.bloch2
        self.bloch1 = np.delete(self.bloch1,0,0)
        self.bloch2 = np.delete(self.bloch2,0,0)
        self.probability = np.delete(self.probability,0,0)


######## EIT ###########


class EIT(OpticalBlochEquation3):
    def __init__(self,Omega1,Omega2,Delta,delta,Gamma1,Gamma2,tmax,init_state,DeltaRange,GammaC=0.0):
        OpticalBlochEquation3.__init__(self,Omega1,Omega2,Delta,delta,Gamma1,Gamma2,GammaC,tmax,init_state,DeltaRange)
        # EIT state
        self.state_EIT = np.mat(init_state.reshape(9,1)) #matrix array
        #define detuning step
        self.numd = 1000
        self.detuning = np.linspace(self.Delta-self.Dmax,self.Delta+self.Dmax,self.numd)

    '''
    def LBsuperoperator_EIT(self,Delta=None):
        if Delta is None:
            Delta = self.Delta
        H_a = Delta*self.g1.T*self.g1 + (Delta-self.delta)*self.g2.T*self.g2
        H_af = 0.5*self.Omega1*(self.sigma1 + np.conj(self.sigma1.T)) + 0.5*self.Omega2*(self.sigma2 + np.conj(self.sigma2.T))
        H = H_a + H_af
        #H = 0.5*np.array([[0,0,self.Omega1],[0,-2*Delta,self.Omega2],[np.conj(self.Omega1),np.conj(self.Omega2),-2*(Delta-self.delta)]])
        #define superoperator
        H_eff = -1j*np.mat(kron(self.I3,H) - kron(np.conj(H.T),self.I3))
        L_eff = self.Gamma1*Dissipator(self.sigma1) + self.Gamma2*Dissipator(self.sigma2) + self.GammaC*Dissipator(self.sigmaC)
        S = H_eff+L_eff
        return S

    def LBDiagonalise_EIT(self,Delta=None):
        if Delta is None:
            Delta = self.Delta
        #diagonalize Hamiltonian
        evals, evecs = eig(self.LBsuperoperator_EIT(Delta))
        evecs = np.mat(evecs)
        return evals, evecs

    '''

    def getNextState_EIT(self,Delta=None):
        if Delta is None:
            Delta = self.Delta
        self.state_EIT = self.LBDiagonalise(Delta)[1]*np.mat(np.diag(np.exp(self.LBDiagonalise(Delta)[0]*self.tmax)))*np.linalg.inv(self.LBDiagonalise(Delta)[1])*self.init_state

    def Initialise_EIT(self):
        self.rhoba_r = np.zeros(1)
        self.rhoba_i = np.zeros(1)

    def saveSusceptibility(self):
        rhobar = np.real(self.state_EIT[2])
        rhobai = -1*np.imag(self.state_EIT[2])
        self.rhoba_r = np.append(self.rhoba_r,rhobar)
        self.rhoba_i = np.append(self.rhoba_i,rhobai)

    def Susceptibility(self):
        self.Initialise_EIT()
        for i in range(self.numd):
            self.getNextState_EIT(self.detuning[i])
            self.saveSusceptibility()
        self.rhoba_r = np.delete(self.rhoba_r,0)
        self.rhoba_i = np.delete(self.rhoba_i,0)
