##
# this python program will
# 1. read raw data for experiment 4
# 2. for each subject, build 10 models, 9 of them have switchpoint in each session, the last one do not have switchpoint
# 3. get model waic for each model
# 4. save model waic for each subject as an excel file

##
import pickle
import gzip
import time

import numpy as np
import scipy as sp
import scipy.io as sio

import pymc3 as pm
import theano
import theano.tensor as tt
import matplotlib.pyplot as plt
import pandas as pd
import random

## get data for single suject
def get_data(sub_id):
    #读取该被试的数据
    subdata=mydata[mydata.Subject==sub_id]
    trialindex=list(subdata['trialindex'])
    rt=list(subdata['RT'])
    conds=list(subdata['Type'])
    print('The trial length of sub%02d is %d' %(sub_id,len(trialindex)))
    return rt, trialindex, conds

## given swtichpoint, build model
def do_sampling_given_switchpoint(sub_id, rt, trialindex, conds, sp):
    # use empirical mean (ignoring condition or time point) as center of prior
    # 以收集到的数据的平均数（忽略条件和时间点）作为先验分布的中心
    mu_obs = np.mean(rt)
    sd_obs = np.std(rt)

    model = pm.Model()
    with model:  # define model as following
        # RTs before switchpoint all come from same distribution
        mu_new = pm.Normal('mu_new', mu=mu_obs, sd=sd_obs * 2, testval=mu_obs)
        mu_before_old_benefit = pm.Normal('mu_before_old_benefit', mu=mu_obs, sd=sd_obs * 2, testval=mu_obs)
        mu_before = pm.math.switch(conds, mu_new - mu_before_old_benefit, mu_new)
        # RTs after switchpoint come from normal distribution where mean depends on
        # condition
        mu_after_old_benefit = pm.Normal('mu_after_old_benefit', mu=mu_obs, sd=sd_obs * 2, testval=mu_obs)
        mu_after = pm.math.switch(conds, mu_new - mu_after_old_benefit, mu_new)
        # define switchpoint base on subject's report
        a = np.arange(sp, sp + 1)
        switchpoint = a.repeat(len(trialindex))
        print('Now set the switchpoint of sub%02d as %d' % (sub_id, sp))
        # if trial is before switchpoint, set mu=mu_before, set mu=mu_after otherwise
        mu = pm.math.switch(trialindex > (switchpoint - 1) * 60, mu_after, mu_before)

        sigma = pm.HalfNormal('sigma', sd=sd_obs * 2, testval=sd_obs * 2)
        # build model for RT
        rt_modelled = pm.Normal('rt_modelled', mu=mu, sd=sigma, observed=rt)

        step = pm.Metropolis()

        trace = pm.sample(40000, step=step, start=model.test_point, chains=8, cores=8)

    return trace[20000::5], model

##
def do_sampling_noswitchpoint(sub_id, rt, trialindex, conds):
    # use empirical mean (ignoring condition or time point) as center of prior
    mu_obs = np.mean(rt)
    sd_obs = np.std(rt)

    model = pm.Model()
    with model:
        mu_new = pm.Normal('mu_new', mu=mu_obs, sd=sd_obs * 2, testval=mu_obs)
        mu_old_benefit = pm.Normal('mu_old_benefit', mu=mu_obs, sd=sd_obs * 2, testval=mu_obs)
        sigma = pm.HalfNormal('sigma', sd=sd_obs * 2, testval=sd_obs * 2)

        mu = pm.math.switch(conds, mu_new - mu_old_benefit, mu_new)

        rt_modelled = pm.Normal('rt_modelled', mu=mu, sd=sigma, observed=rt)

        step = pm.Metropolis()

        trace = pm.sample(40000, step=step, start=model.test_point, chains=8, cores=8) # chains=4,

    return trace[20000::5], model

##
def model_construct(sub_id, model_type, sp):
    filepath = 'experiment4'
    rt, trialindex, conds = get_data(sub_id)
    # log transformation
    logrt = np.log10(rt)
    plt.scatter(trialindex, logrt)
    plt.savefig(filepath + '/scatter_sub{:02d}.png'.format(sub_id))
    print("Now is fitting %s model for sub%02d......" % (model_type, sub_id))
    if model_type == 'nosp':
        trace, model = do_sampling_noswitchpoint(sub_id, logrt, trialindex, conds)
    elif model_type == 'givensp':
        trace, model = do_sampling_given_switchpoint(sub_id, logrt, trialindex, conds, sp)
    with model:
        pm.traceplot(trace)
        plt.savefig(filepath + '/{}_{:02d}_trace_sub{:02d}.png'.format(model_type, sp, sub_id))
        plt.close('all')

        pm.plot_posterior(trace)
        plt.savefig(filepath + '/{}_{:02d}_posterior_sub{:02d}.png'.format(model_type, sp, sub_id))
        plt.close('all')

        # export data
        with gzip.open(filepath + '/tracedata/{}_{:02d}_trace_sub{:02d}.pkl.gz'.format(model_type,sp, sub_id), 'wb') as f:
            pickle.dump((trace, model), f)
        waic = pm.waic(trace, scale='deviance')
    print("The WAIC of %s model is %f" % (model_type, waic.waic))
    print("--------------------------------------------------------")
    return trace, model, waic.waic

##
def run(sub_id, sp_all):
    filepath = 'experiment4'
    tracenp, modelnp, waic_np = model_construct(sub_id, 'nosp', 0)
    # build dataframe, save waic for each switchpoint
    df = pd.DataFrame(columns=['subject', 'model', 'waic'])

    with pd.ExcelWriter(filepath + '/summary_sub' + str(sub_id) + '.xlsx') as writer:
        with modelnp:
            pm.summary(tracenp).to_excel(writer, sheet_name='noswitchpoint')
            df.loc[len(df.index)] = [sub_id, 'noswitchpoint', waic_np]
        # get waic for each switchpoint
        for sp in sp_all:
            tracegp, modelgp, waic_sp = model_construct(sub_id, 'givensp', sp)
            df.loc[len(df.index)] = [sub_id, ('givenswtichpoint_'+str(sp)), waic_sp]
            with modelgp:
                pm.summary(tracegp).to_excel(writer, sheet_name=('givenswtichpoint'+str(sp)))

    # export waic for all models
    df.to_excel('waic_sub'+str(sub_id)+'.xlsx')




if __name__ == '__main__':
    ##
    # get switchpoint for each subject (base on their report)
    tpdata=pd.read_csv('exp4_tp.csv')
    tpdata.subject=(tpdata['subject']).astype(int)
    tpdata.true_transition=(tpdata['true_transition']).astype(int)
    tpdict =tpdata.set_index('subject')['true_transition'].to_dict()
    print(tpdict)
    # get fake switchpoint (randomly assigned)
    ftpdict =tpdata.set_index('subject')['fake_transition'].to_dict()
    print(ftpdict)

    ##
    # import data
    mydata=pd.read_csv('exp4_expdata.csv')
    # delete if acc==0
    mydata.dropna(axis=0,how='any', inplace=True)
    print(mydata)

    ## export waic for all subject in exp4
    for subid in tpdict.keys():
        run(subid, sp_all=[1,2,3,4,5,6,7,8,9]) # get waic for switchpoint in each session

