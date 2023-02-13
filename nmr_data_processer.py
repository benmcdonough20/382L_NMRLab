import pandas as pd
from scipy.optimize import curve_fit
import scipy
import matplotlib
import matplotlib.pyplot as plt
import os
import numpy as np
import pathlib
matplotlib.rcParams.update({'font.size': 22})

def FID_freq_T3():
    def FID_func(x,A,T3,w,offset,phase):
        return A*np.exp(-(x-offset)/T3)*np.cos(w*(x-offset)+phase)
    cwd=os.getcwd()
    ref_dir=os.path.join(cwd,'data','reference_volts.csv')
    ref_df=pd.read_csv(ref_dir,index_col=None,header=None)
    ref_1=ref_df[1].mean()
    ref_2=ref_df[2].mean()
    #init guess offset=600 0.000278, truncate dataframe from 700 0.000478
    data_dir=os.path.join(cwd,'data','Jan_30','FID_resonance','t_26')
    detuning=[]
    input_freq=[]
    for fname in os.listdir(data_dir):
        if '.DS_Store' in fname:
            continue
        freq=int(fname.replace('.csv',''))
        print(freq)
        input_freq.append(freq)
        data_df=pd.read_csv(os.path.join(data_dir,fname),index_col=None,header=None)
        xdata=data_df.loc[700:][0]
        x_offset=data_df.loc[600][0]
        y1data=data_df.loc[700:][1].rolling(20,1).mean().dropna()-ref_1
        # init_guess1=[2.3,8e-4,np.pi*1e3,x_offset,0]
        init_guess1=[4.68001048e+00,1.06813360e-03,8.31020422e+02,x_offset,0]
        # bounds1=[(0,10),(16e-5,4e-3),(0,np.pi*3e-3),(x_offset-1e-4,x_offset+1e-4),(0,2*np.pi)]
        bounds1=[(0,16e-5,0,x_offset-1e-4,0),(10,4e-3,np.pi*2e6,x_offset+1e-4,2*np.pi)]
        popt1,pcov1=curve_fit(FID_func,xdata,y1data,p0=init_guess1,bounds=bounds1)
        print(popt1)
        detuning.append(popt1[2]/(2*np.pi))
        print('Resonance:',freq-popt1[2]/(2*np.pi))
        plt.figure()
        plt.plot(xdata,y1data,label='expt')
        plt.plot(xdata,FID_func(xdata,*popt1),label='fit')
        plt.legend()
        plt.savefig(fname.replace('csv','png'),dpi=300)
        plt.show()
    # plt.figure()
    # plt.plot(input_freq,detuning)
    # plt.show()
    return

def FID_fourier():
    cwd=os.getcwd()
    data_dir=os.path.join(cwd,'data','FID')
    detuning=[]
    input_freq=[]
    for fname in os.listdir(data_dir):
        if '.DS_Store' in fname:
            continue
        freq=int(fname.replace('.csv',''))
        print(freq)
        input_freq.append(freq)
        data_df=pd.read_csv(os.path.join(data_dir,fname),index_col=None,header=None)
        detunings=[]
        T=data_df[0].diff().loc[3]
        for col in [1,2]:
            data_temp=data_df.copy()
            cutoff=data_temp[col].diff().idxmax()+50
            reference=data_temp.loc[:200][col].mean()
            ydata=data_temp.loc[cutoff:].rolling(35,1).mean().dropna()
            first_index=ydata.first_valid_index()
            xdata=data_temp.loc[first_index:]
            yf = scipy.fft.fft(ydata)
            plt.figure()
            plt.plot()
            xf = scipy.fft.fftfreq(len(xdata), T)[:len(xdata)//2]
            print(xf)
            detunings.append(xf[np.argmax(yf)])
        detuning.append(np.mean(detunings))
    plt.figure()
    plt.plot(input_freq,detuning)
    plt.show()
    return

def FID_T3():
    def FID_func(x,A,T3,offset):
        return A*np.exp(-(x-offset)/T3)
    cwd=os.getcwd()
    ref_dir=os.path.join(cwd,'data','reference_volts.csv')
    ref_df=pd.read_csv(ref_dir,index_col=None,header=None)
    ref_1=ref_df[1].mean()
    ref_2=ref_df[2].mean()
    #init guess offset=600 0.000278, truncate dataframe from 700 0.000478
    data_dir=os.path.join(cwd,'data','FID')
    detuning=[]
    input_freq=[]
    for fname in os.listdir(data_dir):
        if '.DS_Store' in fname:
            continue
        freq=int(fname.replace('.csv',''))
        print(freq)
        input_freq.append(freq)
        data_df=pd.read_csv(os.path.join(data_dir,fname),index_col=None,header=None)
        xdata=data_df.loc[700:][0]
        x_offset=data_df.loc[600][0]
        y1data=data_df.loc[700:][1]-ref_1
        init_guess1=[4.68001048e+00,1.06813360e-03,x_offset]
        # bounds1=[(0,10),(16e-5,4e-3),(0,np.pi*3e-3),(x_offset-1e-4,x_offset+1e-4),(0,2*np.pi)]
        bounds1=[(0,16e-5,x_offset-1e-4),(10,4e-3,x_offset+1e-4)]
        popt1,pcov1=curve_fit(FID_func,xdata,y1data,p0=init_guess1,bounds=bounds1)
        print(popt1)
        print('Decay time:',popt1[1])
        # detuning.append(popt1[2]/(2*np.pi))
        # print('Resonance:',freq-popt1[2]/(2*np.pi))
        plt.figure()
        plt.plot(xdata,y1data,label='expt')
        plt.plot(xdata,FID_func(xdata,*popt1),label='fit')
        plt.legend()
        plt.savefig('FID_decay'+fname.replace('csv','png'),dpi=300)
        plt.show()
    # plt.figure()
    # plt.plot(input_freq,detuning)
    # plt.show()
    return

def T1_calc():
    cwd=os.getcwd()
    data_dir=os.path.join(cwd,'ben_ruobin_nmr_feb02/T1_CuSO4')
    files=os.listdir(data_dir)
    plot_data={
        'tau':[],
        'peak':[]
    }
    for file in files:
        # print(file)
        data_df=pd.read_csv(os.path.join(data_dir,file),header=None,index_col=0)
        data_df.index.name='time'
        timestep=data_df.index[1]-data_df.index[0]
        chancutoffs=[]
        reference={1:0,2:0}
        for chan in [1,2]:
            chancutoffs.append(data_df.loc[abs(data_df[chan].diff())>1].tail(1).index.item()+50*timestep)
            # print(data_df)
            reference[chan]=data_df.iloc[:300][chan].mean()
        # print(reference)
        cutoff=max(chancutoffs)
        plot_data['tau'].append(int(pathlib.Path(file).stem))
        chan1_data=data_df[1].rolling(20).mean().loc[cutoff:].copy()-reference[1]
        chan2_data=data_df[2].rolling(20).mean().loc[cutoff:].copy()-reference[2]
        print(file)
        print(data_df.index)
        data_df['vector_summed']=np.sqrt(np.square(chan1_data)+np.square(chan2_data))
        plot_data['peak'].append(np.sqrt(np.square(chan1_data)+np.square(chan2_data)).max())
        # print(chancutoffs)
        # print("======================",data_df.index.tolist()[400])
        # plt.figure()
        # plt.plot(data_df.index,data_df[1],'.r')
        # plt.plot(data_df.index,data_df[2],'.b')
        # plt.plot(data_df.index,data_df['vector_summed'],'.k')

        # plt.axvline(x=data_df['vector_summed'].idxmax(),color='k',linestyle='--')
        # # plt.axvline(x=chancutoffs[1],color='r',linestyle='dotted')
        # plt.show()
    plt.figure()
    plt.plot(plot_data['tau'],plot_data['peak'],'r.')
    plt.xlabel('tau')
    plt.ylabel('peak')
    plt.show()
    return

def FID_glycerin_samplesize():
    def FID_func(x,A,T):
        return A*np.exp(-np.square(x)/T)
    cwd=os.getcwd()
    data_dir=os.path.join(cwd,'Feb09/glycerin_FID')
    files=os.listdir(data_dir)
    plot_data={
    'T2*(s)':[],
    'mass(g)':[],
    'T2_err(s)':[]
    }
    for file in files:
        data_df=pd.read_csv(os.path.join(data_dir,file),header=None,index_col=0)
        data_df.index.name='time'
        timestep=data_df.index[1]-data_df.index[0]
        chancutoffs=[]
        right_cutoff=0.004
        reference={1:0,2:0}
        plot_data['mass(g)'].append(float(pathlib.Path(file).stem.replace('_','.')))
        for chan in [1,2]:
            chancutoffs.append(data_df.loc[abs(data_df[chan].diff())>1].tail(1).index.item()+20*timestep)
            # print(data_df)
            reference[chan]=data_df.iloc[:300][chan].mean()
        yerr=data_df.iloc[:300][1].std()
        print(yerr)
        cutoff=max(chancutoffs)
        chan1_data=data_df[1].loc[cutoff:right_cutoff].copy()-reference[1]
        chan2_data=data_df[2].loc[cutoff:right_cutoff].copy()-reference[2]
        xdata=data_df.loc[cutoff:right_cutoff].index
        data_df['vector_summed']=np.sqrt(np.square(chan1_data)+np.square(chan2_data))
        # data_df['vector_summed']=chan1_data
        init_guess1=[4,2e-03]
        # bounds1=[(0,10),(16e-5,4e-3),(0,np.pi*3e-3),(x_offset-1e-4,x_offset+1e-4),(0,2*np.pi)]
        bounds1=[(0,1e-6),(20,2e-03)]
        popt,pcov=curve_fit(FID_func,xdata,data_df['vector_summed'].dropna(),p0=init_guess1,bounds=bounds1)
        print(popt)
        plot_data['T2_err(s)'].append(np.sqrt(np.diag(pcov))[1])
        plot_data['T2*(s)'].append(popt[1])
        # plt.figure()
        # plt.plot(data_df.index,data_df[1],'.r',label='expt')
        # plt.plot(data_df.index,data_df[2],'.b',label='expt')
        # # plt.plot(xdata,FID_func(xdata,*popt),'.b',label='exp fit')
        # # plt.plot(xdata,data_df['vector_summed'].dropna()+yerr,'-k',label='error range')
        # # plt.plot(xdata,data_df['vector_summed'].dropna()-yerr,'-k',label='error range')
        # # plt.axvline(x=cutoff,color='k',linestyle='dotted')
        # # plt.savefig(os.path.join(cwd,'glycerin_FID_plots',str(pathlib.Path(file).stem)+'.png'),dpi=300)
        # plt.show()
        # plt.close()

        plt.figure()
        plt.plot(xdata,data_df['vector_summed'].dropna(),'.r',label='expt')
        # plt.plot(xdata,chan1_data,'.k',label='expt')
        plt.plot(xdata,FID_func(xdata,*popt),'.b',label='fit')
        # plt.plot(xdata,data_df['vector_summed'].dropna()+yerr,'-k',label='error range')
        # plt.plot(xdata,data_df['vector_summed'].dropna()-yerr,'-k',label='error range')
        # plt.axvline(x=cutoff,color='k',linestyle='dotted')
        plt.xlabel('Time (s)')
        plt.ylabel('$M_T$ (dim)')
        plt.legend()
        plt.savefig(os.path.join(cwd,'glycerin_FID_plots','FID'+str(pathlib.Path(file).stem)+'.png'),bbox_inches='tight',dpi=300)
        plt.close()
    plt.figure()
    plt.plot(plot_data['mass(g)'],np.multiply(plot_data['T2*(s)'],1e6),'.r',markersize=10)
    # plt.errorbar(plot_data['mass(g)'],np.multiply(plot_data['T2*(s)'],1e6),yerr=np.multiply(plot_data['T2_err(s)'],10000),xerr=0.01,linestyle='')
    plt.xlabel('mass (g)')
    plt.ylabel('T2* ($10^{-6}$s)')
    plt.savefig(os.path.join(cwd,'glycerin_FID_plots','T2+_sample_size.png'),bbox_inches='tight',dpi=300)
    plt.close()
    return

# FID_glycerin_samplesize()

def T2_glycerin_conc():
    def T2_func(x,A,T):
        return A*np.exp(-x/T)
    init_guess1=[4,2e-2]
    bounds1=[(0,1e-3),(20,10)]
    cwd=os.getcwd()
    data_dir=os.path.join(cwd,'Feb09/T2_glycerin_conc')
    subdirs=os.listdir(data_dir)
    plot_data={
    'T(s)':[],
    'conc(%wt)':[]
    }
    label_dict={
        'HA':0,
        'A':100,
        'B':80.3,
        'C':64.7,
        'D':42.3,
        'E':29.4
    }
    intervals_dict={
        'HA':0,
        'A':130,
        'B':50,
        'C':130,
        'D':130,
        'E':130    
    }
    peaknum={
        'HA':10,
        'A':10,
        'B':9,
        'C':10,
        'D':10,
        'E':10    
    }
    for subdir in subdirs:
        if subdir not in label_dict:
            continue
        plot_data['conc(%wt)'].append(label_dict[subdir])
        if subdir=='A':
            numpeaks={'A_1':5,
                      'A_2':4,
                      'A_3':3}
            peaks={
                'time':[],
                'height':[],
                'error':[]
            }
            for file in os.listdir(os.path.join(data_dir,subdir)):
                if '.DS_Store' in file:
                    continue
                print(file)
                data_df=pd.read_csv(os.path.join(data_dir,subdir,file),header=None,index_col=0)
                data_df.index.name='time'
                timestep=data_df.index[1]-data_df.index[0]
                max_intervals=[]
                reference={1:0,2:0}
                data_df['1diff']=data_df[1].diff()
                i=0
                while i<len(data_df.index):
                    if abs(data_df['1diff'].iloc[i])>1:
                        max_intervals.append(data_df.index[i])
                        i+=10
                    else:
                        i+=1
                lefts=[elt-130*timestep for elt in max_intervals]      
                rights=[elt+130*timestep for elt in max_intervals]     

                errors={}
                mean_chan_squared={}
                for chan in [1,2]:
                    # print(data_df)
                    reference[chan]=data_df.iloc[:250][chan].mean()
                    squared_chan=np.square(data_df.iloc[:250][chan]-reference[chan])
                    mean_chan_squared[chan]=squared_chan.mean()
                    errors[chan]=squared_chan.var()
                error=((errors[1]+errors[2])/(4*(mean_chan_squared[1]+mean_chan_squared[2])**2))**0.5

                data_df['1_zeroed']=data_df[1]-reference[1]
                data_df['2_zeroed']=data_df[2]-reference[2]
                data_df['vector_summed']=np.sqrt(np.square(data_df['1_zeroed'])+np.square(data_df['2_zeroed']))
                for i in range(numpeaks[pathlib.Path(file).stem]):
                    peakloc=data_df['vector_summed'].loc[rights[i+1]:lefts[i+2]].idxmax()
                    peaks['time'].append(peakloc)
                    peaks['height'].append(data_df['vector_summed'].loc[peakloc])
                    peaks['error'].append(error)
            peak_df=pd.DataFrame.from_dict(peaks)
            peak_df.to_csv(os.path.join(data_dir,'peak_locations',subdir+'_peaks.csv'),index=None)
            popt,pcov=curve_fit(T2_func,np.array(peaks['time']),np.array(peaks['height']),p0=init_guess1,bounds=bounds1)
            plt.figure()
            plt.plot(peaks['time'],peaks['height'],'.r',label='expt')
            plt.errorbar(peaks['time'],peaks['height'],peaks['error'],linestyle='')
            plt.plot(np.linspace(np.min(peaks['time']),np.max(peaks['time']),10000),T2_func(np.linspace(peaks['time'][0],peaks['time'][-1],10000),*popt),'-b',label='fit')
            plt.xlabel('Time (s)')
            plt.ylabel('$M_T$ echo peak (arb u.)')
            plt.legend()
            plt.savefig(os.path.join(cwd,'glycerin_conc_T2',f'{label_dict[subdir]}_peaks.png'),bbox_inches='tight',dpi=300)
            plt.close()

            plt.figure()
            plt.plot(data_df.index,data_df['vector_summed'],'.k')
            plt.xlabel('Time (s)')
            plt.ylabel('$M_T$ (arb u.)')
            # plt.plot(data_df.index,data_df['1_zeroed'],'.r')
            # plt.plot(data_df.index,data_df['2_zeroed'],'.b')
            # plt.vlines(x=lefts,ymin=0,ymax=10,colors='r',linestyle='dashed')
            # plt.vlines(x=rights,ymin=0,ymax=10,colors='b',linestyle='dashed')
            plt.savefig(os.path.join(cwd,'glycerin_conc_T2',f'{label_dict[subdir]}_raw.png'),bbox_inches='tight',dpi=300)
            plt.close()
        else:
            peaks={
                'time':[],
                'height':[],
                'error':[]
            }
            df_list=[]
            for file in os.listdir(os.path.join(data_dir,subdir)):
                if '.DS_Store' in file:
                    continue
                print(file)
                df_list.append(pd.read_csv(os.path.join(data_dir,subdir,file),header=None,index_col=0))
            data_df=pd.concat(df_list)
            data_df=data_df[~data_df.index.duplicated(keep='first')]
            data_df.index.name='time'
            timestep=data_df.index[1]-data_df.index[0]
            max_intervals=[]
            reference={1:0,2:0}
            data_df['1diff']=data_df[1].diff()
            i=0
            while i<len(data_df.index):
                if abs(data_df['1diff'].iloc[i])>3:
                    max_intervals.append(data_df.index[i])
                    i+=10
                else:
                    i+=1
            lefts=[elt-intervals_dict[subdir]*timestep for elt in max_intervals]+[np.inf]      
            rights=[elt+intervals_dict[subdir]*timestep for elt in max_intervals]   
            errors={}
            mean_chan_squared={}
            for chan in [1,2]:
                # print(data_df)
                reference[chan]=data_df.iloc[:250][chan].mean()
                plt.figure()
                plt.plot(data_df.iloc[:250].index,data_df.iloc[:250][chan])
                plt.show()
                plt.close()
                squared_chan=np.square(data_df.iloc[:250][chan]-reference[chan])
                mean_chan_squared[chan]=squared_chan.mean()
                errors[chan]=squared_chan.var()
            error=((errors[1]+errors[2])/(4*(mean_chan_squared[1]+mean_chan_squared[2])**2))**0.5
            data_df['1_zeroed']=data_df[1]-reference[1]
            data_df['2_zeroed']=data_df[2]-reference[2]
            data_df['vector_summed']=np.sqrt(np.square(data_df['1_zeroed'])+np.square(data_df['2_zeroed']))
            plt.figure()
            plt.plot(data_df.index,data_df['vector_summed'],'.k')
            # plt.plot(data_df.index,data_df['1_zeroed'],'.r')
            # plt.plot(data_df.index,data_df['2_zeroed'],'.b')
            # plt.vlines(x=lefts,ymin=0,ymax=10,colors='r',linestyle='dashed')
            # plt.vlines(x=rights,ymin=0,ymax=10,colors='b',linestyle='dashed')
            # plt.vlines(x=max_intervals,ymin=0,ymax=10,colors='b',linestyle='dashed')
            plt.xlabel('Time (s)')
            plt.ylabel('$M_T$ (arb u.)')
            plt.savefig(os.path.join(cwd,'glycerin_conc_T2',f'{label_dict[subdir]}_raw.png'),bbox_inches='tight',dpi=300)
            plt.close()
            if subdir=='B':
                for i in range(peaknum[subdir]):
                    peakloc=data_df['vector_summed'].loc[rights[i+1]:lefts[i+2]].idxmax()
                    peaks['time'].append(peakloc)
                    peaks['height'].append(data_df['vector_summed'].loc[peakloc])
                    peaks['error'].append(error)
                    peak_df=pd.DataFrame.from_dict(peaks)
                    print(peak_df)
                    peak_df.to_csv(os.path.join(data_dir,'peak_locations',subdir+'_peaks.csv'),index=None)
                # bounds1=[(0,10),(16e-5,4e-3),(0,np.pi*3e-3),(x_offset-1e-4,x_offset+1e-4),(0,2*np.pi)]
                popt,pcov=curve_fit(T2_func,np.array(peaks['time']),np.array(peaks['height']),p0=init_guess1,bounds=bounds1)
                print(popt)
                plt.figure()
                plt.plot(peaks['time'],peaks['height'],'.r',label='expt')
                plt.plot(np.linspace(np.min(peaks['time']),np.max(peaks['time']),10000),T2_func(np.linspace(peaks['time'][0],peaks['time'][-1],10000),*popt),'-b',label='fit')
                plt.errorbar(peaks['time'],peaks['height'],peaks['error'],linestyle='')
                plt.xlabel('Time (s)')
                plt.ylabel('$M_T$ echo peak (arb u.)')
                plt.legend()
                plt.savefig(os.path.join(cwd,'glycerin_conc_T2',f'{label_dict[subdir]}_peaks.png'),bbox_inches='tight',dpi=300)
                plt.close()
T2_glycerin_conc()

def plot_trigger_error():
    data_df=pd.read_csv('/Users/Ruobin/Desktop/YALE/Spring 2023/PHYS 382L/NMR/Feb09/T2_glycerin_conc/A/A_1.csv',index_col=0,header=None)
    data_df.index.name='time'
    data_df=data_df.loc[-0.005:0.01]
    plt.figure()
    plt.plot(data_df.index,data_df[1],'.k')
    # plt.axvline(0,linestyle='--',color='b')
    plt.axvline(0.003,linestyle='--',color='b')
    plt.xlabel('Time (s)')
    plt.ylabel('$M_T$ (arb. u)')
    plt.savefig(os.path.join(os.getcwd(),'glycerin_FID_plots','Trigger_error.png'),bbox_inches='tight',dpi=300)
# plot_trigger_error()