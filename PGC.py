
# Copyright (C) 2020 Joseph Young - see GPLv2_note.txt for full notice

################################################################################
# IMPORTS
################################################################################
from CCMI import CCMI
import numpy as np
from scipy import stats
import copy
################################################################################





################################################################################
# pcoh: COMPUTE PCOH & COH BETWEEN TWO CHANNELS
#   INPUTS:
#       x: Data in shape (channel)x(trial)x(time)
#       time_step: Time step used in sampling (seconds)
#       time_max: Maximum time for data (seconds)
#   OUTPUTS:
#       P: Estimate of partial coherence between channels
#       C: Estimate of coherence between channels
################################################################################
def pcoh(x, time_step, time_max):
    # INITIALIZE
    num_channels = x.shape[0]

    # FFT
    X = np.fft.fft(x,axis=2)

    # POWER SPECTRAL DENSITY MATRIX
    S = np.zeros((num_channels, num_channels,X.shape[2]),dtype=np.complex_)
    for jj in range(num_channels):
        for ii in range(num_channels):
            S[ii,jj,:] = (2*(time_step**2)/time_max)*np.mean(np.squeeze(X[ii,:,:])*np.squeeze(np.conj(X[jj,:,:])),axis=0)

    # COMPUTE COHERENCE
    C = np.zeros(S.shape)
    for ii in range(num_channels):
        for jj in range(num_channels):
            C[ii,jj,:] = (np.abs(S[ii,jj,:])**2)/(np.abs(S[ii,ii,:])*np.abs(S[jj,jj,:]))

    # INVERT S & NORMALIZE TO GET PCOH
    S_inv = np.zeros(S.shape,dtype=np.complex_)
    for ii in range(X.shape[2]):
        S_inv[:,:,ii]= np.linalg.inv(S[:,:,ii])
    P = np.zeros(S.shape)
    for ii in range(num_channels):
        for jj in range(num_channels):
            P[ii,jj,:] = (np.abs(S_inv[ii,jj,:])**2)/(np.abs(S_inv[ii,ii,:])*np.abs(S_inv[jj,jj,:]))

    # RETURN PCOH & COH
    return P, C
################################################################################
# END OF pcoh
################################################################################





################################################################################
# pgc: COMPUTE MIF/PGC BETWEEN TWO CHANNELS
#   INPUTS: Use input order pgc(x,B,f1,f2,f3,ii_specific,jj_specific)
#       x: Data in shape (channel)x(trial)x(time)
#       B: Number of bootstrap iterations
#       f1: Frequency index to compute MIF at for rows
#       f2: (Optional) Frequency index to compute MIF at for columns. If not
#           included, then set equal to f1 (linear interaction).
#       f3: (Optional) Frequency index to condition at for other channels. If
#           not specified, no conditioning occurs.
#       ii_specific: MIF for f1 is only computed for channel index ii_specific
#       jj_specific: MIF for f2 is only computed for channel index jj_specific.
#       NOTE: f3 can be a list of frequency indices
#   OUTPUTS:
#       MIF: Estimate of MIF/PGC between channels, where element is MIF between
#            f1 of row channel and f2 of column channel, all conditioned on
#            other channels at f3 if f3 is specified.
################################################################################
def pgc(*arg):

    # EXTRACT INPUT ARGUMENTS
    x = arg[0]
    B = arg[1]
    f1 = arg[2]

    # INITIALIZE
    num_channels = x.shape[0]
    num_trials = x.shape[1]

    # OTHER ARGS
    # f2
    if len(arg)>3:
        f2 = arg[3]
    else:
        f2 = f1

    #f3
    if len(arg)>4:
        f3 = arg[4]
    else:
        f3 = [-1]

    #ii_range
    if len(arg)>5:
        ii_range = [arg[5]]
    else:
        ii_range = np.arange(num_channels)

    #jj_range
    if len(arg)>6:
        jj_range = [arg[6]]
    else:
        jj_range = np.arange(num_channels)

    # CHECK THAT INPUT FREQUENCIES ARE LISTS
    if type(f1) is not list:
        f1 = [f1]
    if type(f2) is not list:
        f2 = [f2]
    if type(f3) is not list:
        f3 = [f3]

    # FFT
    X = np.fft.fft(x,axis=2)

    # SPLIT REAL & IMAG INTO TWO SEPARATE DIM
    X_sel_f1_ri = np.zeros((num_channels,num_trials,2*len(f1)))
    X_sel_f2_ri = np.zeros((num_channels,num_trials,2*len(f2)))
    X_sel_f3_ri = np.zeros((num_channels,num_trials,2*len(f3)))
    for ii in range(num_channels):
        X_sel_f1_ri[ii,:,:] = np.concatenate((np.real(X[ii,:,:][:,f1]),np.imag(X[ii,:,:][:,f1])),axis=1)
        X_sel_f2_ri[ii,:,:] = np.concatenate((np.real(X[ii,:,:][:,f2]),np.imag(X[ii,:,:][:,f2])),axis=1)
        if f3[0]==-1:
            X_sel_f3_ri[ii,:,:] = 0
        else:
            X_sel_f3_ri[ii,:,:] = np.concatenate((np.real(X[ii,:,:][:,f3]),np.imag(X[ii,:,:][:,f3])),axis=1)

    # ESTIMATE MIF/PGC BETWEEN EACH CHANNEL
    MIF = np.zeros((num_channels,num_channels))
    for ii in ii_range:
        for jj in jj_range:
            # CHECK IF LINEAR INTERACTION
            if f1==f2:

                # COMPUTE FOR LOWER LEFT
                if ii<jj:

                    # EXTRACT RELEVANT DATA
                    ii_data = copy.deepcopy(np.squeeze(X_sel_f1_ri[ii,:,:]))
                    jj_data = copy.deepcopy(np.squeeze(X_sel_f2_ri[jj,:,:]))

                    kk_data = np.zeros((num_trials,2*len(f3)*(num_channels-2)))
                    ll=0

                    for kk in range(num_channels):
                        if (kk!=ii and kk!=jj):
                            kk_data[:,(2*ll*len(f3)):((2*ll*len(f3))+(2*len(f3)))] = copy.deepcopy(np.squeeze(X_sel_f3_ri[kk,:,:]))
                            ll=ll+1

                    # ZSCORE (ONLY ZSCORE NONZERO COLUMNS)
                    ii_col_to_z = np.where(np.sum(ii_data,axis=0)!=0)[0]
                    jj_col_to_z = np.where(np.sum(jj_data,axis=0)!=0)[0]
                    kk_col_to_z = np.where(np.sum(kk_data,axis=0)!=0)[0]

                    ii_data[:,ii_col_to_z] = stats.zscore(ii_data[:,ii_col_to_z],axis=0)
                    jj_data[:,jj_col_to_z] = stats.zscore(jj_data[:,jj_col_to_z],axis=0)
                    kk_data[:,kk_col_to_z] = stats.zscore(kk_data[:,kk_col_to_z],axis=0)

                    # ESTIMATE MIF/PGC
                    MIF[ii,jj] = CCMI(ii_data,
                                      jj_data,
                                      kk_data,
                                      tester = 'Classifier',
                                      metric = 'donsker_varadhan',
                                      num_boot_iter = B,
                                      h_dim = 64, max_ep = 20).get_cmi_est()

                    # MIRROR ONTO UPPER RIGHT
                    MIF[jj,ii] = MIF[ii,jj]
            # NONLINEAR
            else:

                if ii!=jj:

                    # EXTRACT RELEVANT DATA
                    ii_data = copy.deepcopy(np.squeeze(X_sel_f1_ri[ii,:,:]))
                    jj_data = copy.deepcopy(np.squeeze(X_sel_f2_ri[jj,:,:]))

                    kk_data = np.zeros((num_trials,2*len(f3)*(num_channels-2)))
                    ll=0

                    for kk in range(num_channels):
                        if (kk!=ii and kk!=jj):
                            kk_data[:,(2*ll*len(f3)):((2*ll*len(f3))+(2*len(f3)))] = copy.deepcopy(np.squeeze(X_sel_f3_ri[kk,:,:]))
                            ll=ll+1

                    # ZSCORE (ONLY ZSCORE NONZERO COLUMNS)
                    ii_col_to_z = np.where(np.sum(ii_data,axis=0)!=0)[0]
                    jj_col_to_z = np.where(np.sum(jj_data,axis=0)!=0)[0]
                    kk_col_to_z = np.where(np.sum(kk_data,axis=0)!=0)[0]

                    ii_data[:,ii_col_to_z] = stats.zscore(ii_data[:,ii_col_to_z],axis=0)
                    jj_data[:,jj_col_to_z] = stats.zscore(jj_data[:,jj_col_to_z],axis=0)
                    kk_data[:,kk_col_to_z] = stats.zscore(kk_data[:,kk_col_to_z],axis=0)

                    # ESTIMATE MIF/PGC
                    MIF[ii,jj] = CCMI(ii_data,
                                      jj_data,
                                      kk_data,
                                      tester = 'Classifier',
                                      metric = 'donsker_varadhan',
                                      num_boot_iter = B,
                                      h_dim = 64, max_ep = 20).get_cmi_est()

    # IF COMPUTED FOR PAIR OF CHANNELS, ONLY RETURN VALUE FOR CHANNELS
    if len(arg)>6:
        MIF = MIF[ii_range[0], jj_range[0]]

    # RETURN MIF/PGC
    return MIF
################################################################################
# END OF pgc
################################################################################
