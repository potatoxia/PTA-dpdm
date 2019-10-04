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
Searching for DPDM signal using the Bayesian method in any PTA data release.

Xiao Xue (2019.09)

xuexiao@mail.itp.ac.cn

"""





if __name__ == '__main__':
	datadir = '../ppta/fit1data'
	parfiles = sorted(glob.glob(datadir + '/*.par'))
	timfiles = sorted(glob.glob(datadir + '/*.tim'))
	parfiles.remove(datadir+'/J1125-6014.par')
	timfiles.remove(datadir+'/J1125-6014.tim')

	psrs = []
	for p, t in zip(parfiles, timfiles):
		psr = Pulsar(p, t)
		psrs.append(psr)
	
	# Modeling the signals

	dp = DPDM.dpdm_block(type_ = 'Bayes')
	tm = gp_signals.TimingModel()
	wnb = models.white_noise_block(vary=True)
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

	# Get Starting Points and constant parameter values.
	save1 = np.load('noisepars.npy')
	save2 = np.load('noisepardict.npy')
	save3 = np.load('dpdmpars-mlh.npy')
	save4 = np.load('dpdmpardict.npy')
	Dict = {save2[i]:save1[i] for i in range(len(save1))}
	Dict.update({save4[i]:save3[i] for i in range(len(save3))})

	# Use the best fit noise parameters!
	PTA.set_default_params(Dict)


	# Set starting points at the already-known best fit points!
	xs = {par.name:par.sample() for par in PTA.params}
	for parname in Dict.keys():
		if parname in xs.keys():
			xs[parname] = Dict[parname]
	x0 = np.hstack([xs[key] for key in sorted(xs.keys())])

	# Set groups
	groups = [range(ndim),range(ndim-5,ndim)]



	# First sampling (prepare)
	
	sampler = PTSampler(ndim,PTA.get_lnlikelihood,PTA.get_lnprior,
				cov = np.diag(np.ones(ndim)*0.025), groups=groups, 
				outDir='/home/sdb/xuexiao/PTAchains/openBayesian/')
	sampler.sample(x0,10000000,isave=1000)

	
	
