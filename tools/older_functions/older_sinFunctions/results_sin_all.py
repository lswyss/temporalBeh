import numpy as np
import matplotlib.pyplot as plt
import pickle
from ..bootstrapTest import *


def data_of_interest(names, interest=[], exclude=[]):
    to_plot = set()
    exclude = set(exclude)
    full_amp = all('127amp' in i or 'amp' not in i for i in interest)
    if full_amp:
        exclude.add('amp')
    for dat in names:
        if any(i in dat for i in interest) and not any(ex in dat for ex in exclude) and all(dat.count('+') <= i.count('+') for i in interest):
            to_plot.add(dat)
    return list(to_plot)


def calc_delta(x, w=2):
    if w % 2 == 0:
        w += 1
    filt = np.concatenate((-np.ones(w // 2), [0], np.ones(w // 2)))
    return np.convolve(x, filt, mode='same') / w

def response_sin(interest_list,exclude,periods =[.5,1,2,3,4,5,10],duration=30,
                          n_boot=1e3,statistic=np.median,
                          measure_compare=None,ind_measure=[],
                          pop_measure=[],baseline=128,
                          conf_interval=95, stat_testing=True, derivative=False,
                          visible=False,t_samp=(-3,40),plot_indiv_trial=False,):


    #keys for measure_compare
    DIFFERENCE = ['diff','difference','subtract']
    RELATIVE = ['relative','rel','divide']
    #significance marker
    def mark_sig(ax,loc,c='grey'):
      ax.scatter([loc,],12,marker='*',color=c)
    #give extra n_boot to measurements
    n_boot_meas = max(n_boot, 3e2)

    fig, ax_all=plt.subplots(nrows=len(periods), ncols=1, sharex=True, sharey=True,figsize=(10,2*len(periods)))


    if len(periods)==1: ax_all = np.array([ax_all])#ax_all[None,:]
    if True: ax_all=ax_all[:,None]
    ax = ax_all

    for ii,interest in enumerate(interest_list):
        if ii==0:
            #step function reference
            #for trial in trial_subset: #so here i want take only the relevant trials then loop through
            name = 'data/LDS_response_LONG.pickle'
            with open(name,'rb') as f:
                result = pickle.load(f)
            # print(result['tau'])
            xp = result['tau']
            ind_t = np.where((xp>=t_samp[0])&(xp<=t_samp[1]))[0] #extracting a subset of data that falls within the specified time range
            #print("keys",result.keys())
            to_plot = data_of_interest(result.keys(),[f'{interest}_{duration}m2h{128}bp'],exclude=exclude)
            #print("to_plot",to_plot)
            if len(to_plot)==0:
                yp_ref=np.zeros((1,xp.size))
            else:
                yp_ref = np.concatenate([result[dat] for dat in to_plot])
                loc = np.argmin(xp**2)
                #for trial in trial_subset: #loop through the subset
                y,rng = bootstrap_traces(yp_ref[:,ind_t],n_boot=n_boot,statistic=statistic,conf_interval=conf_interval)
                for a in ax[:,0]:
                    c='grey'
                    a.plot(xp[ind_t],y,lw=1,color=c,zorder=-2,)
                    print("plot", len(xp[ind_t]), len(y))
                    a.fill_between(xp[ind_t],*rng,alpha=.25,color=c,lw=0,edgecolor='None',zorder=-2)
            ind_t_ref = ind_t.copy()
            xp_ref=xp.copy()
        else:
            continue

        #loop the sin periods
        name = 'data/LDS_response_sinFunc.pickle'
        #name = 'data/LDS_response_sinFunc_v2.pickle'
        if plot_indiv_trial:
            name = 'data/LDS_response_sinFunc_individual.pickle'
        if visible:
            print('visible') # no print
            name = 'data/LDS_response_Vis_sinFunc.pickle'
        with open(name,'rb') as f:
            result = pickle.load(f)
            #print(result.keys())

        for i,p in enumerate(periods):
            xp = result['tau']
            ind_t = np.where((xp>=t_samp[0]-1e-5)&(xp<=t_samp[1]))[0] #SB 03.01.23, added the -1e-5 due to weird float precision issue
         # Start with minutes
            p_name = f'{p}m'
            to_plot = data_of_interest(result.keys(),[f'{interest}_{duration}m_{p_name}'],exclude=exclude)

            # If no results, try seconds
            if len(to_plot) == 0 and (p<1 or p%1>0):
                p_name = f'{int(p*60)}s'
                to_plot = data_of_interest(result.keys(),[f'{interest}_{duration}m_{p_name}'],exclude=exclude)

            if len(to_plot) == 0:
                continue


            print(len(xp), to_plot)
            loc = np.argmin(xp**2)

            if plot_indiv_trial:
                #concatenate sets of worms from diff. experiments
                yp = []
                for dat in to_plot:
                    yp.extend(result[dat]['data'])

                trial_plot = plot_indiv_trial

                ##This is for plotting all trials on one plot (useful for generating AI files)
                # for trial in trial_plot:
                #     # make list of all worms from a given trial
                #     yp_trial = []
                #     for y in yp:
                #         try:
                #             yp_trial.append(y[trial])
                #         except:
                #             continue
                #     yp_trial = np.array(yp_trial)
                #     print(yp_trial.shape)
                #     # calculate and plot trial results
                #     y, rng = bootstrap_traces(yp_trial[:,ind_t],n_boot=n_boot,statistic=statistic,conf_interval=conf_interval)
                #     c = plt.cm.plasma(trial/np.max(trial_plot))
                #     ax[i,0].plot(xp[ind_t],y,lw=1,color=c,zorder=-1,label=f'{interest} {p}m, trial:{trial}, ({yp_trial.shape[0]})',)
                #     ax[i,0].fill_between(xp[ind_t],*rng,alpha=.25,color=c,lw=0,edgecolor='None',zorder=-2)



                num_trials = len(trial_plot)

                for i, trial in enumerate(trial_plot):
                    fig, ax = plt.subplots(figsize=(8, 3))

                    yp_trial = []
                    for y in yp:
                        try:
                            yp_trial.append(y[trial])
                        except:
                            continue
                    yp_trial = np.array(yp_trial)
                    print(yp_trial.shape)

                    y, rng = bootstrap_traces(yp_trial[:, ind_t], n_boot=n_boot, statistic=statistic, conf_interval=conf_interval)
                    c = plt.cm.plasma(trial / np.max(trial_plot))
                    ax.plot(xp[ind_t], y, lw=1, color=c, zorder=-1, label=f'{interest} {p}m, trial: {trial}, ({yp_trial.shape[0]})')
                    ax.fill_between(xp[ind_t], *rng, alpha=.25, color=c, lw=0, edgecolor='None', zorder=-2)

                    # Set appropriate labels, title, etc.
                    ax.set_xlabel('Time (m)')
                    ax.set_ylabel('Activity')
                    ax.set_title(f'Trial {trial}')
                    ax.legend()

                    plt.show()

                    #continue

            else:
                yp = np.concatenate([result[dat]['data'] for dat in to_plot])
                y,rng = bootstrap_traces(yp[:,ind_t],n_boot=n_boot,statistic=statistic,conf_interval=conf_interval)


                if derivative:
                    True #not interesting results :(

                    c=plt.cm.Set1(ii/9)#'cornflowerblue'

                ax[i,0].plot(xp[ind_t],y,lw=1,color=c,zorder=-1,label=f'{interest} {p}m, ({yp.shape[0]})',)
                ax[i,0].fill_between(xp[ind_t],*rng,alpha=.25,color=c,lw=0,edgecolor='None',zorder=-2)
                ax[i,0].plot(xp,result[to_plot[0]]['stim'],c='thistle',zorder=-10)
                time_sig = timeDependentDifference(yp[:,ind_t],yp_ref[:,ind_t_ref],n_boot=n_boot,conf_interval=conf_interval)
                x_sig = xp[ind_t]#np.arange(0,10,1/120)
                y_sig = .1
                j=0
                bott = np.zeros_like(x_sig)+1.5 - (j-1)*y_sig
                ax[i,0].fill_between(x_sig,bott,bott-y_sig*time_sig,
                    facecolor=c,alpha=.4)
                box_keys={'lw':1, 'c':'k'}
                ax[i,0].plot(x_sig,bott,**box_keys)
                ax[i,0].plot(x_sig,bott-y_sig,**box_keys)
            ax[i,0].legend()
        ax[i,0].set_xlim(-3,40)
        ax[i,0].set_ylim(0,1.6)

    ax[-1,0].set_xlabel('time (min)')
    return fig,ax