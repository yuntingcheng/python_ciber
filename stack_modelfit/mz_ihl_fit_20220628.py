##################################################################
##
##  ihl_fit_20220628.py
##  Jun 28 2022
##  Mike Zemcov
##
##################################################################

import numpy as np
import matplotlib.pyplot as plt
import pickle
import scipy.optimize as opt
#from mpl_toolkits.mplot3d import Axes3D

def logistic(x,a,b,c,d):
    return a / (1. + np.exp(-c * (x-d))) + b

def get_ciber1_Cl(f_IHL,l_in):
    fname = 'ytc/ciber1_Cl/Cl_ciber.pkl'
    with open(fname, "rb") as f:
        Cl_ciber = pickle.load(f)

    r_IHL = f_IHL / (1 - f_IHL)
    l = Cl_ciber['l']
    Clah = (Cl_ciber['Cla'] +\
                            r_IHL**2*Cl_ciber['Clh'] +\
                            2*r_IHL*Cl_ciber['Clha'])
    Clah_shsub = Clah - Cl_ciber['Cla_sh'] * (1+r_IHL)**2
    Dl_native = l*(l+1)*np.median(Clah_shsub, axis=0)/2/np.pi


    whpl = Dl_native > 0.0
    l = l[whpl]
    Dl_nativep = Dl_native[whpl]

    #(a_, b_, c_, d_), _ = opt.curve_fit(logistic,np.log10(l),np.log10(Dl_nativep),method='trf')
    #Dl_fit = logistic(np.log10(l), a_, b_, c_, d_)

    
    Dl = np.power(10,np.interp(np.log10(l_in),np.log10(l),np.log10(Dl_nativep),right=-5))

    
    return l_in,Dl

########################################################################

ax = plt.subplot(111)
n_theory = 100

data = np.loadtxt('ciber1_1p1.txt',comments=';;',delimiter=' ',skiprows=8)

l_data = data[:,0]
Dl_data = data[:,1]
Dl_datadown = data[:,2]
Dl_dataup = data[:,3]

n_data = np.shape(l_data)

Dl_dgl = l_data * (l_data + 1) * l_data**(-3) * 5.76e2 

shotvals = np.linspace(1.,1.5)
ihlvals = np.linspace(0,0.5)
lf = np.zeros([50,50])
chi = np.zeros([50,50])

ishot = 0
for shots in shotvals:
    iihl = 0
    for ihls in ihlvals:
        Dl_shot = l_data * (l_data + 1) * np.ones(n_data) * 1e-7 * shots
        l_ihl,Dl_ihl = get_ciber1_Cl(ihls,l_data)

        Dl_theory = Dl_shot + Dl_dgl + Dl_ihl
        
        errsq = ((Dl_datadown + Dl_dataup) / 2.)**2
        chisq = np.sum((Dl_data - Dl_theory)**2 / (2.*errsq)) / (n_data[0] - 3)
        chi[ishot,iihl] = chisq
        likelihood = np.exp(-chisq / 2.)
        lf[ishot,iihl] = likelihood
        iihl = iihl + 1
    ishot = ishot + 1


l_theory = np.logspace(2,5,n_theory)
lf = lf / np.sum(lf)

lf_marg = np.sum(lf,axis=0)
lf_marg = lf_marg / np.sum(lf_marg)

if 0:
    extent = [np.min(ihlvals), np.max(ihlvals), np.min(shotvals), np.max(shotvals)]
    ax.imshow(np.flipud(lf),extent=extent,aspect=1)

    #X, Y = np.meshgrid(ihlvals,shotvals)
    #ax.plot_surface(X,Y,np.flipud(chi))#,extent=extent,aspect=1)
    ax.set_xlabel(r'$f_{\rm IHL}$')
    ax.set_ylabel(r'Shot Noise Amplitude (arb.)')

if 0:
    ax.plot(ihlvals,lf_marg)
    ax.set_xlabel(r'$f_{\rm IHL}$')
    ax.set_ylabel(r'Likelihood')

whmax = np.where(lf == np.amax(lf))
print(shotvals[whmax[0]])
print(ihlvals[whmax[1]])

lf_marg_copy = lf_marg
whmid = ihlvals[lf_marg_copy == np.amax(lf_marg_copy)]
sep = 0.
limdn = 1.
limup = 0.
while (sep < 0.68):
    maxspot = lf_marg_copy == np.amax(lf_marg_copy)
    sep = sep + lf_marg[maxspot]
    lf_marg_copy[maxspot] = 0.0
    if ihlvals[maxspot] < whmid[0]:
        limdn = ihlvals[maxspot]
    if ihlvals[maxspot] > whmid[0]:
        limup = ihlvals[maxspot]
    
Dl_shot = l_data * (l_data + 1) * np.ones(n_data) * 1e-7*shotvals[whmax[0]]
l_ihl,Dl_ihl = get_ciber1_Cl(ihlvals[whmax[1]],l_data)
l_ihl,Dl_ihldn = get_ciber1_Cl(limdn,l_data)
l_ihl,Dl_ihlup = get_ciber1_Cl(limup,l_data)
Dl_theory = Dl_shot + Dl_dgl + Dl_ihl
Dl_theory_dn = Dl_shot + Dl_dgl + Dl_ihldn
Dl_theory_up = Dl_shot + Dl_dgl + Dl_ihlup

if 0:
    ax.loglog(l_data,Dl_ihl,label='OH+IHL')
    ax.loglog(l_data,Dl_shot,label='Shot') 
    ax.loglog(l_data,Dl_dgl,label='DGL (Fixed)')
    ax.loglog(l_data,Dl_theory,label='Best Fit Theory',color='red')
    ax.loglog(l_data,Dl_theory_dn,linestyle=':',color='red')
    ax.loglog(l_data,Dl_theory_up,linestyle=':',color='red')

    ax.errorbar(l_data,Dl_data,yerr=[Dl_datadown,Dl_dataup],marker='o',linestyle='')
    ax.set_ylim(1e-2,1e4)

    ax.set_xlabel(r'$\ell$')
    ax.set_ylabel(r'$\ell (\ell + 1) C_{\ell} / 2 \pi$ (nW$^{2}$ m$^{-4}$ sr$^{-2}$)')
    plt.legend()

if 1: 
    yerr = (Dl_datadown+Dl_dataup)/2.
    ax.semilogx(l_data,(Dl_data - Dl_theory)/yerr,marker='o',linestyle='')
    ax.errorbar(l_data,(Dl_data - Dl_theory)/yerr,yerr=np.ones(n_data[0]),marker='o',linestyle='')
    print(np.sum(((Dl_data - Dl_theory)**2/yerr**2) ))
    print((n_data[0] - 3) )
    ax.set_xlabel(r'$\ell$')
    ax.set_ylabel(r'$(D_{\ell}^{\rm data} - D_{\ell}^{\rm theory}) / \delta D_{\ell}$')
    
#plt.show()
plt.savefig('ihl_fit_20220628.pdf')

plt.close()

# return
