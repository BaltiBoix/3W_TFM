import sys
import os
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

import pickle
import bisect
import sklearn
import sklearn.model_selection
from sklearn.metrics import classification_report

class d3w():
    def __init__(self, path3w):
        self.path3w = path3w
        self.df = self.__load_df()
        return

    def __load_df(self):

        d = dict()
        d['origin'] = []
        d['label'] = []
        d['path'] = []
        d['nlines'] = []
        for i in pathlib.Path(self.path3w).iterdir():
            if i.stem.isnumeric():
                print(i)
                label = int(i.stem)
                for fp in i.iterdir():
                    # Considers only csv files
                    if fp.suffix == ".csv":

                        if (fp.stem.startswith("SIMULATED")):
                            d['origin'].append('S')
                        elif fp.stem.startswith("DRAWN"):
                            d['origin'].append('D')
                        else:
                            d['origin'].append('R')
                        
                        d['label'].append(label)
                        d['path'].append(fp)
                        d['nlines'].append(self.file_len(fp)-1)
                        #d['nlines'].append(1)
        return pd.DataFrame(d)
    
    def split(self, real=True, simul=True, drawn=True, test_size=0.2, val_size=0.1):
        
        tmp0_df = self.get_df(real, simul, drawn)
        tmp_df, test_df = sklearn.model_selection.train_test_split(tmp0_df, 
                                                        test_size=test_size, 
                                                        random_state=200560, 
                                                        shuffle=True, 
                                                        stratify=tmp0_df['label'])
        train_df, val_df = sklearn.model_selection.train_test_split(tmp_df, test_size=val_size, 
                                                        random_state=200560, 
                                                        shuffle=True, 
                                                        stratify=tmp_df['label'])
        print('Instances Train: {}  Test: {}  Validation: {}'.format(len(train_df.index), 
                                                                     len(test_df.index), 
                                                                     len(val_df.index)))
        
        return train_df.reset_index(drop=True),\
               test_df.reset_index(drop=True),\
               val_df.reset_index(drop=True)
    
    def file_len(self, filename):
        j = 0
        with open(filename) as f:
            for i, x in enumerate(f):
                if x.strip() == '':
                    j += 1
        return i + 1 - j
    
    def get_df(self, real=True, simul=True, drawn=True):
        sel = []
        if real:
            sel.append('R')
        if simul:
            sel.append('S')
        if drawn:
            sel.append('D')
        if sel:
            return self.df[self.df['origin'].isin(sel)].drop(columns=['origin']).reset_index(drop=True)
    
    @property
    def all(self):
        return self.df.drop(columns=['origin'])
    @property
    def real(self):
        return self.df[self.df['origin']=='R'].drop(columns=['origin']).reset_index(drop=True)
    @property
    def simul(self):
        return self.df[self.df['origin']=='S'].drop(columns=['origin']).reset_index(drop=True)
    @property
    def drawn(self):
        return self.df[self.df['origin']=='D'].drop(columns=['origin']).reset_index(drop=True)

################################################################################################

class CustomDataGen(tf.keras.utils.Sequence):
    '''https://medium.com/analytics-vidhya/write-your-own-custom-data-generator-for-tensorflow-keras-1252b64e41c3'''
    
    def __init__(self, df, X_col, y_col, categories,
                 batch_size,
                 seq_length=15,
                 tmp_path='/tmp'):
        
        self.df = df.copy()
        self.X_col = X_col
        self.y_col = y_col
        self.categories = categories
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.tmp_path = pathlib.Path(tmp_path)
        self.file = None
        self.ts = None
        
        self.n = self.__calc_n()
    
    def __calc_n(self):
        self.df['nbatches'] = np.int32(np.ceil((np.ceil(self.df['nlines'] / 60)-self.seq_length+1)/self.batch_size))
        self.df['ibatch'] = self.df['nbatches'].cumsum() - 1
        return int(self.df['nbatches'].sum())
    
    def plot(self, ifile):
        
        ds = self.__get_ds(self.df['path'][ifile], Norm=False)
        
        fig, axs = plt.subplots(nrows=len(self.X_col)+1, figsize=(10, 12), sharex=True)
        
        fig.suptitle(self.df['path'][ifile])

        for i, vs in enumerate(self.X_col):
            axs[i].plot(ds.index, ds[(vs, 'mean')])
            axs[i].fill_between(ds.index, ds[(vs, 'mean')]-1.96*ds[(vs, 'std')], 
                            ds[(vs, 'mean')]+1.96*ds[(vs, 'std')], 
                            alpha=0.2)
            axs[i].set_ylabel(vs)
            axs[i].grid()
        
        id = np.argsort(ds[(self.y_col, 'mode')])
        
        axs[i+1].scatter([ds.index[i] for i in id], [str(ds[(self.y_col, 'mode')][i]) for i in id], marker='.')
        
        axs[i+1].set_ylabel(self.y_col)
        
        axs[i+1].set_xlabel('minute')

        plt.show()    
    
    def on_epoch_end(self):
            pass
    
    def pkl_path(self, p):
        new_p = self.tmp_path.joinpath(p.parts[-2], p.stem+'.pkl')
        new_p.parent.mkdir(exist_ok=True, parents=True)
        return new_p
    
    def tkl_path(self, p):
        new_p = self.tmp_path.joinpath(p.parts[-2], p.stem+'.tkl')
        new_p.parent.mkdir(exist_ok=True, parents=True)
        return new_p
    
    def __get_ds(self, p, Norm=True):
    
        pkl_p = self.pkl_path(p)
        
        if pkl_p.exists():
            ds = pd.read_pickle(pkl_p)
        else:
        
            dfo = pd.read_csv(p, index_col="timestamp", parse_dates=["timestamp"])

            if np.any(dfo[self.y_col].isna()):
                dfo[self.y_col] = dfo[self.y_col].fillna(method='ffill')
            dfo[self.y_col] = dfo[self.y_col].astype('int')

            flist = []
            flist0 = []
            for f in self.X_col:
                nas = np.sum(dfo[f].isna())
                if nas > 0:
                    if nas < len(dfo.index) * 0.2:
                        dfo[f] = dfo[f].fillna(method='ffill')
                        flist.append(f)
                    else:
                        flist0.append(f)
                else:
                    flist.append(f)

            fdict=dict()
            for f in flist:
                fdict[f] = ['mean','std']

            def mode(series):
                return pd.Series.mode(series)[0]

            fdict[self.y_col] = [mode]

            dfo['minute'] = (dfo.index-dfo.index[0])//np.timedelta64(1,'m')

            ds = dfo.groupby('minute').agg(fdict)

            #ds = ds.iloc[:-1]

            for f in flist0:
                ds[f, 'mean'] = np.NaN
                ds[f, 'std'] = np.NaN
                
            ds.to_pickle(pkl_p)
        
        if Norm:
            ds = self.__Norm(ds)
            if ds.isnull().any().any():
                print(p, pkl_p)
        
        return ds[self.X_col + [self.y_col]]
        
    def __Norm(self, ds, nas_v=0):
        dn = ds.fillna(value=nas_v)
        sc = sklearn.preprocessing.StandardScaler()
        dn = pd.DataFrame(sc.fit_transform(dn), 
                                           index=dn.index, 
                                           columns=dn.columns)
        dn[(self.y_col, 'mode')] = ds[(self.y_col, 'mode')]
        return dn
    
    def __get_dt(self, p):

        tkl_p = self.tkl_path(p)
        
        if tkl_p.exists():
            with open(tkl_p, 'rb') as f:
                self.ts = pickle.load(f)
            return

        ds = self.__get_ds(p)
        
        self.ts = tf.keras.utils.timeseries_dataset_from_array(
            ds.drop(self.y_col, axis=1, level=0),
            ds[self.y_col].iloc[self.seq_length-1:].append(ds[self.y_col].iloc[:self.seq_length-1]).reset_index(drop=True),
            sequence_length=self.seq_length,
            sequence_stride=1,
            sampling_rate=1,
            batch_size=self.batch_size,
            shuffle=False,
            seed=None,
            start_index=None,
            end_index=None
        )        
        
        self.ts = list(self.ts)
        
        with open(tkl_p, 'wb') as f:
            pickle.dump(self.ts, f)

        return
    
    def reset_ts(self):
        for p in self.df['path']:
            self.tkl_path(p).unlink(missing_ok=True)
    
    def __get_output(self, y):
        
        ohe = sklearn.preprocessing.OneHotEncoder(categories=self.categories, sparse= False)
        
        return ohe.fit_transform(y)
    
    def __get_data(self, i, j, p):
        # Generates data containing batch_size samples

        if p != self.file:
            self.__get_dt(p)

        return self.ts[j]
    
    def __getitem__(self, index):
        
        i = bisect.bisect_left(self.df.ibatch, index)
        if i > 0:
            j = index - self.df.ibatch[i-1] - 1
        else:
            j = index
        
        p = self.df.path[i]       
        
        # print(index, i, j, p)
        
        X, y = self.__get_data(i, j, p)        
        
        return X, self.__get_output(y)
    
    def __len__(self):
        return self.n
    
    def get_y(self):
        y = np.empty(0, 'int')
        for p in self.df['path']:
            pkl_p = self.pkl_path(p)
            if pkl_p.exists():
                ds = pd.read_pickle(pkl_p)
            else:
                ds = self.__get_ds(p, Norm = False)
            y = np.append(y, ds[self.y_col].iloc[self.seq_length-1:].astype('int'))
        return y