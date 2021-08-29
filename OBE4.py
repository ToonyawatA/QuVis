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
    mdim=4
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
            R[1+row,1+col]=getComplexColor(rho[col+4*row,t_index],peak)[0]
            G[1+row,1+col]=getComplexColor(rho[col+4*row,t_index],peak)[1]
            B[1+row,1+col]=getComplexColor(rho[col+4*row,t_index],peak)[2]

    return np.dstack((R, G, B))


def Dissipator(sigma):
    d = np.shape(sigma)[0]
    sigma = np.mat(sigma)
    I = np.eye(d)
    I = np.mat(I)
    return np.mat(kron(np.conj(sigma),sigma) - 0.5*kron(I,np.conj(sigma.T)*sigma) - 0.5*kron((np.conj(sigma.T)*sigma).T,I))  

class OpticalBlochEquation4:
    ### init ###
    def __init__(self,Omega1,phi1,Omega2,phi2,Omega3,phi3,Omega4,phi4,Delta1,Delta2,Delta3,Gamma0,Gamma1,Gamma2,tmax,init_state,DeltaRange=None):
        self.Omega1 = Omega1
        self.phi1 = phi1 
        self.Omega2 = Omega2
        self.phi2 = phi2 
        self.Omega3 = Omega3
        self.phi3 = phi3 
        self.Omega4 = Omega4 
        self.phi4 = phi4 
        self.Delta1 = Delta1
        self.Delta2 = Delta2
        self.Delta3 = Delta3
        self.Gamma0 = Gamma0
        self.Gamma1 = Gamma1
        self.Gamma2 = Gamma2
        self.tmax = tmax
        self.Dmax = DeltaRange
        self.init_state = np.mat(init_state.reshape(16,1)) #matrix array
        #define radius
        self.radius = 3.0
        #define quantum state
        self.s0 = np.mat(np.array([1,0,0,0]))
        self.s1 = np.mat(np.array([0,1,0,0]))
        self.s2 = np.mat(np.array([0,0,1,0]))
        self.s3 = np.mat(np.array([0,0,0,1]))
        #define matrice
        self.sigma1 = self.s0.T*self.s1
        self.sigma2 = self.s1.T*self.s2
        self.sigma3 = self.s2.T*self.s3
        self.sigma4 = self.s0.T*self.s3
        self.I4 = np.mat(np.eye(4))
        #define time step and
        self.numt = 1000
        self.time = np.linspace(0,self.tmax,self.numt)
        self.dt = self.time[1]-self.time[0]

    def LBsuperoperator(self,Delta1=None,Delta2=None,Delta3=None):
        Delta1 = self.Delta1 if Delta1 is None else Delta1
        Delta2 = self.Delta2 if Delta2 is None else Delta2
        Delta3 = self.Delta3 if Delta3 is None else Delta3
        ### Hamiltonian for three-level with two fields 
        H_a = -1*(Delta1+Delta2+Delta3)*self.s0.T*self.s0 + -1*(Delta2+Delta3)*self.s1.T*self.s1 + -1*(Delta3)*self.s2.T*self.s2
        H_af = 0.5*self.Omega1*(self.sigma1*np.exp(1j*self.phi1) + np.conj(self.sigma1.T*np.exp(1j*self.phi1))) + 0.5*self.Omega2*(self.sigma2*np.exp(1j*self.phi2) + np.conj(self.sigma2.T*np.exp(1j*self.phi2))) + 0.5*self.Omega3*(self.sigma3*np.exp(1j*self.phi3) + np.conj(self.sigma3.T*np.exp(1j*self.phi3))) + 0.5*self.Omega4*(self.sigma4*np.exp(1j*self.phi4) + np.conj(self.sigma4.T*np.exp(1j*self.phi4)))
        H = H_a + H_af
        #define superoperator
        H_eff = -1j*np.mat(kron(self.I4,H) - kron(np.conj(H.T),self.I4))
        L_eff = self.Gamma0*Dissipator(self.sigma1) + self.Gamma1*Dissipator(self.sigma2) + self.Gamma2*Dissipator(self.sigma3)
        S = H_eff+L_eff
        return S

    def LBDiagonalise(self,Delta1=None,Delta2=None,Delta3=None):
        Delta1 = self.Delta1 if Delta1 is None else Delta1
        Delta2 = self.Delta2 if Delta2 is None else Delta2
        Delta3 = self.Delta3 if Delta3 is None else Delta3
        #diagonalize Hamiltonian
        evals, evecs = eig(self.LBsuperoperator(Delta1,Delta2,Delta3))
        evecs = np.mat(evecs)
        return evals, evecs

    def getNextState(self,Delta1=None,Delta2=None,Delta3=None):
        Delta1 = self.Delta1 if Delta1 is None else Delta1
        Delta2 = self.Delta2 if Delta2 is None else Delta2
        Delta3 = self.Delta3 if Delta3 is None else Delta3
        self.state = self.LBDiagonalise(Delta1,Delta2,Delta3)[1]*np.mat(np.diag(np.exp(self.LBDiagonalise(Delta1,Delta2,Delta3)[0]*self.dt)))*np.linalg.inv(self.LBDiagonalise(Delta1,Delta2,Delta3)[1])*self.state

    def Initialise(self):
        #define bloch vector and probability array
        self.state = self.init_state #matrix array
        self.state_arr = np.zeros(16) #state array (9*1 array)
        self.probability = np.zeros(4)

    def saveTrajectory(self):
        #save state 16*1 array
        rho = np.squeeze(np.asarray((self.state)))
        self.state_arr = np.column_stack((self.state_arr,rho))
        #state probability
        ps0 = np.real(self.state[0,0])
        ps1 = np.real(self.state[5,0])
        ps2 = np.real(self.state[10,0])
        ps3 = np.real(self.state[15,0])
        #probability
        parray = np.array([ps0,ps1,ps2,ps3])
        self.probability = np.vstack((self.probability,parray))

    def Trajectory(self,Delta1=None,Delta2=None,Delta3=None):
        ###
        Delta1 = self.Delta1 if Delta1 is None else Delta1
        Delta2 = self.Delta2 if Delta2 is None else Delta2
        Delta3 = self.Delta3 if Delta3 is None else Delta3
        self.Initialise()
        for i in range(self.numt):
            self.saveTrajectory()
            self.getNextState(Delta1,Delta2,Delta3)
        self.probability = np.delete(self.probability,0,0)
        self.state_arr = np.delete(self.state_arr,0,1)

    def makePlot(self,population=False,savefig=None):
        poplabel = [r'$|0\rangle$',r'$|1\rangle$',r'$|2\rangle$',r'$|3\rangle$']
        plt.figure()
        for i in range(4):
            plt.plot(self.time,self.probability[:,i],label=poplabel[i])
        plt.xlabel(r'Time ($t$)')
        plt.ylabel(r'Population')
        plt.legend()
        if savefig is not None:
            plt.savefig(savefig[0],dpi=120)
        plt.show()

    ####### Density matrix visualization #######

    def DMVis(self,DRange=5,rows=1,columns=1,t_index=None,Delta1=None,Delta2=None,Delta3=None,Full_Visualize=False,Interactive=False,savefig=None):
        t_index = self.numt if t_index is None else t_index
        Delta1 = self.Delta1 if Delta1 is None else Delta1
        Delta2 = self.Delta2 if Delta2 is None else Delta2
        Delta3 = self.Delta3 if Delta3 is None else Delta3
        #generate time for multiplots
        if(t_index>self.numt):
            raise ValueError('t_index must not exceed %d' %self.numt)
        else:
            tt_index = np.linspace(0,t_index,rows*columns+1)
        #define dimension of system
        mdim = 4
        #define color for text
        di=[0.0/255.0,42.0/255.0,65.0/255.0] # Durham ink

        ###for full visualization
        if(Full_Visualize):
            #define size of table
            time_dim = 40 #no. of times to plot across rows
            parameter_dim = 21 # no. of parameter values down columns
            #Delta array (detuning)
            Deltas1 = np.linspace(Delta1-DRange,Delta1+DRange,parameter_dim)
            Deltas2 = np.linspace(Delta2-DRange,Delta2+DRange,parameter_dim)
            Deltas3 = np.linspace(Delta3-DRange,Delta3+DRange,parameter_dim)
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
                self.Trajectory(Delta3=Deltas3[v_index])
                rho = self.state_arr
                peak = np.amax(abs(rho))
                for h_index in range(0, time_dim): # h_index is no. of columns
                    t_index=10*h_index
                    for col in range (0,mdim):
                        for row in range (0,mdim):
                            R[(mdim+1)*v_index+1+row,(mdim+1)*h_index+1+col]=getComplexColor(rho[col+4*row,t_index],peak)[0]
                            G[(mdim+1)*v_index+1+row,(mdim+1)*h_index+1+col]=getComplexColor(rho[col+4*row,t_index],peak)[1]
                            B[(mdim+1)*v_index+1+row,(mdim+1)*h_index+1+col]=getComplexColor(rho[col+4*row,t_index],peak)[2]

            RGB=np.dstack((R, G, B))
            plt.imshow(RGB)
            ax.set_axis_off()
            if savefig is not None:
                plt.savefig(savefig,dpi=120)
            plt.show()

        else:
            #for slider at specific Delta
            if(Interactive):
                self.Trajectory(Delta1,Delta2,Delta3)
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
                self.Trajectory(Delta1,Delta2,Delta3)
                rho = self.state_arr
                peak = np.amax(abs(rho))
                k=1
                for v_index in range(0,rows):
                    for h_index in range(0,columns): # h_index is no. of columns
                        for col in range (0,mdim):
                            for row in range (0,mdim):
                                R[1+row,1+col]=getComplexColor(rho[col+4*row,int(tt_index[k])],peak)[0]
                                G[1+row,1+col]=getComplexColor(rho[col+4*row,int(tt_index[k])],peak)[1]
                                B[1+row,1+col]=getComplexColor(rho[col+4*row,int(tt_index[k])],peak)[2]
                        
                        RGB=np.dstack((R, G, B))
                        label=ascii_lowercase[(4*v_index+h_index)]
                        ax[v_index,h_index].imshow(RGB) if rows*columns!=1 else ax.imshow(RGB)
                        ax[v_index,h_index].text(0.05, 0.85,label,transform=ax[v_index,h_index].transAxes,color=di,fontsize=20)
                        ax[v_index,h_index].set_axis_off() if rows*columns!=1 else ax.set_axis_off()
                        k += 1
                if savefig is not None:
                    plt.savefig(savefig,dpi=120)
                plt.show()
                


###### STIRAP ########
    
