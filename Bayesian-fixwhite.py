import sys 
sys.path.append('../enterprise/enterprise') 
from pulsar import Pulsar
import glob
from enterprise.signals import signal_base
from enterprise.signals import utils
from enterprise.signals import parameter
from enterprise.signals import white_signals
from enterprise.signals import gp_signals
from enterprise.signals import deterministic_signals
from enterprise_extensions import models
from enterprise_extensions import DPDM


import numpy as np
np.seterr(divide = 'ignore')

from PTMCMCSampler.PTMCMCSampler import PTSampler

import multiprocessing


"""
Searching for DPDM signal using the Frequentist method in any PTA data release.

Xiao Xue (2019.09)

xuexiao@mail.itp.ac.cn

"""








	# End of the function.
#====================================================
if __name__ == '__main__':


	# Importing files.

	datadir = '../ppta/fit1data'
	parfiles = sorted(glob.glob(datadir + '/*.par'))
	timfiles = sorted(glob.glob(datadir + '/*.tim'))
	parfiles.remove(datadir+'/J1125-6014.par')
	timfiles.remove(datadir+'/J1125-6014.tim')

		# I throw away this certain pulsar because it may return a unstable, sometimes
		# very large ln-likelihood value. I think there is a bug somewhere.
	

	psrs = []
	for p, t in zip(parfiles, timfiles):
		psr = Pulsar(p, t)
		psrs.append(psr)






	# The Big Model

	dp = DPDM.dpdm_block(type_ = 'Bayes')
	tm = gp_signals.TimingModel()
	wnb = models.white_noise_block(vary=False)
	dmn = models.dm_noise_block(components=50)                         
	spn = models.red_noise_block(components=50)  

	model = tm + dp + wnb + dmn + spn
	nparams = [] # Get the number of parameters of each pulsar
	signals = []
	for psr in psrs:
		signal = model(psr)
		nparams.append(len(signal.params)-5) # Subtracting common DPDM params
		signals.append(signal)
	PTA = signal_base.PTA(signals)
	ndim = len(PTA.params)
	
	# Get Starting Points
	save1 = np.load('noisepars.npy')
	save2 = np.load('noisepardict.npy')
	Dict = {save2[i]:save1[i] for i in range(len(save1))}

	xs = {par.name:par.sample() for par in PTA.params}
	for parname in Dict.keys():
		if parname in xs.keys():
			xs[parname] = Dict[parname]

	# Use the best fit noise parameters!
	PTA.set_default_params(Dict)

	# Set groups
	N=0
	groups=[]
	for n in nparams:	
		groups.extend([[N,N+1],[N+2,N+3]])
		groups.extend([range(N,N+n)])
		N += n
	groups.extend([range(ndim)])



	x0 = np.hstack([xs[key] for key in sorted(xs.keys())])
	N=0
	for n in nparams:
		x0[N+4]=0
		N += n
	#

	sampler = PTSampler(ndim,PTA.get_lnlikelihood,PTA.get_lnprior,
				cov = np.diag(np.ones(ndim)*0.25), groups=groups, 
				outDir='/home/sdb/xuexiao/PTAchains/FixWhiteBayesian/')
	sampler.sample(x0,100000000,isave=1000)






