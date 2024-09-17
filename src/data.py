# -*- coding: utf-8 -*-
"""Contains :class:`DataSource`, which provides the training and testing data for the models.
"""
import pandas as pd
import numpy as np
from numpy.random import Generator, PCG64
from scipy.signal import savgol_filter
import inspect
import pathlib
import sys
import os

sys.path.append(os.path.abspath('..')) # module: qrc_surrogate

from src.helpers import moving_average, normalize, unnormalize, standardize


class DataSource:
    """Provides training and validtion data.

    Contains reactor data from Siemens and benchmark tasks like Narma and short term memory (STM).

    Args:
        nepisodes (int): 
            number of episodes for training
        nvalepisodes (int): 
            number of episodes for validating. If None set to 0.5*nepisodes.
        xtype (str):
            tsetpoint or reactor, smooth, random, smoothfilter, binary, linear, constant
        ytype (str): 
            treactor or reactor, narma, stm, multiply, same
        mermory: 
            order of time delay from x to y. for narma and stm.
        normx, normy (str): 
            none, norm, std
        steps: 
            number of steps per episode. Doesn't apply to reactor data.
        train_equal_val (bool): 
            For debugging. Validation data is the same as the training data.
    """
    # module_path = os.path.abspath(os.path.join('..')) 
    # this_file = pathlib.Path(__file__).absolute()
    module_path = pathlib.Path(__file__).parent.parent.resolve().__str__()
    # reactor data
    file_path = '/data/sprungversuche.h5'
    data_path = module_path + file_path
    # reactor data range
    # reactor_ymin, reactor_ymax = 309, 378 # T_reactor_measured
    reactor_ymin, reactor_ymax = 305, 380 # T_reactor_measured with leeway for rounding errors
    # reactor_xmin, reactor_xmax = 352, 365 # T_setpoint
    reactor_xmin, reactor_xmax = reactor_ymin, reactor_ymax
    
    def __init__(
        self,
        nepisodes = 40, 
        nvalepisodes = 15,
        xtype = 'tsetpoint',
        ytype = 'treactor',
        steps = 80, 
        memory = 1,
        dimx = 1,
        dimy = 1,
        xnorm = 'norm',
        ynorm = 'norm',
        rseed_data = 0,
        train_equal_val = False,
    ) -> None:
        # --
        self.xtype = xtype
        self.ytype = ytype
        self.memory = int(memory) 
        self.nepisodes = max(int(nepisodes), 1) 
        if nvalepisodes is None:
            self.nvalepisodes = max(int(nepisodes * .2), 1)
        else:
            self.nvalepisodes = nvalepisodes
        self.nepisodes_tt = self.nepisodes + self.nvalepisodes 
        self.steps = steps
        self.dimx = dimx
        self.dimy = dimy
        self.rseed_data = rseed_data
        self.train_equal_val = train_equal_val
        # --
        self.xtrain = None
        self.ytrain = None
        self.xval = None
        self.yval = None
        self.xtrain_nonorm = None # before normalization, standardization
        self.ytrain_nonorm = None
        self.xval_nonorm = None
        self.yval_nonorm = None
        self.episodeids = None # list(trainepisodes, valepisodes). for reactor data
        # --
        self.xnorm = xnorm
        self.ynorm = ynorm
        self.xmin = None # of the whole dataset, not just the returned batch
        self.xmax = None 
        self.ymin = None
        self.ymax = None
        self.xmin_nonorm = None # before normalization, standardization. needed for unscaling
        self.xmax_nonorm = None 
        self.ymin_nonorm = None
        self.ymax_nonorm = None
        # --
        if xtype == 'random' and ytype == 'narma':
            raise Warning("""xtype == 'random' and ytype == 'narma' will lead to inf values. 
            Use xtype == 'random_narma' instead. 
            (random samples from 0 to 1, narma_random from 0 to 0.5).""")
        self.run()

    def _fix_input(self):
        """Fix parameters.

        Set memory to 0 if its not needed.
        """
        if (self.xtype == 'reactor') or (self.ytype == 'reactor'):
            self.xtype = 'tsetpoint'
            self.ytype = 'treactor'
        if self.xtype == 'random_narma':
            self.xtype = 'narma_random'
        if self.ytype not in ['narma', 'narma_random', 'narma_smooth', 'stm','narma2']:
            self.memory = 0
        return self

    def get_hyperparameters(self):
        """Return a dictionary of passable parameters and their current values.

        Use to recreate instance

        .. code-block:: python

            params = model.get_hyperparameters()
            model_copy = Model(**params)
        """
        params = inspect.getfullargspec(self.__init__).args[1:]
        dictstring = ''.join(f'\'{p}\': self.{p}, ' for p in params)
        dictstring = '{' + dictstring + '}'
        return eval(dictstring)
    
    def get_dims(self):
        return self.xmin, self.xmax, self.ymin, self.ymax
    
    def get_dims_nonorm(self):
        return self.xmin_nonorm, self.xmax_nonorm, self.ymin_nonorm, self.ymax_nonorm
    
    def run(self):
        """Generate episodes, train-test-split, and normalize.
        """
        self._fix_input()
        x, y = self._generate_x_and_y()
        self.xtrain_nonorm, self.ytrain_nonorm, self.xval_nonorm, self.yval_nonorm = self._train_val_split(x=x, y=y)
        xnorm, ynorm = self._normalize_data(x, y)
        self.xtrain, self.ytrain, self.xval, self.yval = self._train_val_split(x=xnorm, y=ynorm)
        if self.train_equal_val:
            self.nvalepisodes = self.nepisodes
            self.nepisodes_tt = self.nepisodes + self.nvalepisodes
        return self
    
    def get_data(self):
        """Get train-test-split.
        
        Returns:
            xtrain, ytrain, xval, yval (list[np.array]): shape(episodes, steps, dim)
        """
        return self.xtrain, self.ytrain, self.xval, self.yval
    
    def _generate_x_and_y(self):
        """Generate x and y.

        Returns:
            x: list[np.array] (episodes, steps, dim_x)
            y: list[np.array] (episodes, steps, dim_y)
        """
        # x
        self.xmin = [0 for _ in range(self.dimx)]
        self.xmax = [1 for _ in range(self.dimx)]
        # generate extra
        if self.ytype == 'stm':
            self.steps += self.memory
        if self.xtype == 'binary':
            x = self.x_binary()
        elif self.xtype == 'random':
            x = self.x_random()
        elif self.xtype ==  'constant':
            x = self.x_const()
        elif self.xtype ==  'linear':
            x = self.x_linear()
        elif self.xtype ==  'smooth':
            x = self._x_smoothfilter()
        elif self.xtype == 'tsetpoint':
            self.dimx = 1
            self.xmin, self.xmax = [self.reactor_xmin], [self.reactor_xmax]
            x, y = self._xy_reactor()
        elif self.xtype in ['narma_random']:
            self.xmax = [0.5 for _ in range(self.dimx)]
            self.dimx = self.dimy
            x = self._x_narma_random()
        elif self.xtype == 'narma_smooth': 
            self.xmax = [0.5 for _ in range(self.dimx)]
            self.dimx = self.dimy
            x = self._x_narma_smooth()
        elif self.xtype == 'narma_seq': 
            self.xmax = [0.5 for _ in range(self.dimx)]
            self.dimx = self.dimy
            x = self._x_narma_seq()
        else:
            raise Warning(f'Invalid x_type {self.xtype}')
        # y
        if self.ytype == 'stm':
            # for x shifted by 1
            self.dimy = self.dimx
            self.ymin = self.xmin
            self.ymax = self.xmax
            y = [np.roll(xe, self.memory).reshape(self.steps, self.dimy) for xe in x]
            # remove extra steps
            self.steps -= self.memory
            y = [ye[:self.steps] for ye in y]
            x = [xe[:self.steps] for xe in x]
        elif self.ytype == 'same':
            self.dimy = self.dimx
            self.ymin = self.xmin
            self.ymax = self.xmax
            y = [xe for xe in x]
        elif self.ytype == 'multiply':
            # for product of x * x shifted by 1
            self.dimy = self.dimx
            self.ymin = [dx**2 for dx in self.xmin]
            self.ymax = [dx**2 for dx in self.xmax]
            y = [np.multiply(np.roll(xe, 1), xe).reshape(self.steps, self.dimy) for xe in x]
        elif self.ytype == 'treactor':
            self.dimy = 1
            self.ymin, self.ymax = [self.reactor_ymin], [self.reactor_ymax]
            if self.xtype != 'tsetpoint': # y already set
                _, y = self._xy_reactor()
        elif self.ytype in ['narma']: 
            if self.xtype != self.ytype: # y already set
                y = self._y_narma(n=self.memory, xs=x)
            self.ymin = [min(0, np.min(col)) for col in np.vstack(y).T]
            self.ymax = [max(1, np.max(col)) for col in np.vstack(y).T]
        elif self.ytype in ['narma2']: 
            if self.xtype != self.ytype: # y already set
                y = self._y_narma2(n=self.memory, xs=x)
            self.ymin = [min(0, np.min(col)) for col in np.vstack(y).T]
            self.ymax = [max(1, np.max(col)) for col in np.vstack(y).T]
        else:
            raise Warning(f'Invalid y_type {self.ytype}')
        return x, y
    
    def _normalize_data(self, x, y):
        """Input x y and return normalized x y.
        Can be normalized, standardized, or nothing (original input).
        
        Args, Returns:
            x: list[np.array] (episodes, steps, dim_x)
            y: list[np.array] (episodes, steps, dim_y)
        """
        # normalize x
        self.xmin_nonorm, self.xmax_nonorm = self.xmin, self.xmax
        match self.xnorm:
            case 'norm':
                if np.allclose(self.xmin, np.zeros(np.shape(self.xmin))) and np.allclose(self.xmax, np.full(np.shape(self.xmax), 1)):
                    xnorm = x
                else:
                    xnorm = normalize(x, self.xmin, self.xmax)
                    self.xmin = [0 for _ in range(self.dimx)]
                    self.xmax = [1 for _ in range(self.dimx)]
                assert np.all(np.array([np.max(xe) for xe in xnorm]) <= np.asarray(self.xmax)), f'{[np.max(xe) for xe in x]}'
                assert np.all(np.array([np.min(xe) for xe in xnorm]) >= np.asarray(self.xmin)), f'{[np.min(xe) for xe in x]}'
            #    if (self.xmin_nonorm == None) and (self.xmax_nonorm == None):  # Updatesd version..
            #        self.xmin_nonorm = [np.min(col) for col in np.vstack(x).T]
            #        self.xmax_nonorm = [np.max(col) for col in np.vstack(x).T]
            #    
#
#                if np.allclose(self.xmin, np.zeros(np.shape(self.xmin))) and np.allclose(self.xmax, np.full(np.shape(self.xmax), 1)):
#                    xnorm = x
#                else:
#                    xnorm = normalize(x, self.xmin, self.xmax)
#                    self.xmin = [0 for _ in range(self.dimx)]
#                    self.xmax = [1 for _ in range(self.dimx)]
#                assert np.all(np.array([np.max(xe) for xe in xnorm]) <= np.asarray(self.xmax)), f'{[np.max(xe) for xe in x]}'
#                assert np.all(np.array([np.min(xe) for xe in xnorm]) >= np.asarray(self.xmin)), f'{[np.min(xe) for xe in x]}'
            case 'std':
                if (self.xmin_nonorm == None) and (self.xmax_nonorm == None):
                    self.xmin_nonorm = [np.min(col) for col in np.vstack(x).T]
                    self.xmax_nonorm = [np.max(col) for col in np.vstack(x).T]
                
                xnorm = standardize(x)
                self.xmin = [np.min(col) for col in np.vstack(xnorm).T]
                self.xmax = [np.max(col) for col in np.vstack(xnorm).T]
            case 'scale':
                self.xmin_nonorm = [np.min(col) for col in np.vstack(x).T]
                self.xmax_nonorm = [np.max(col) for col in np.vstack(x).T]
                xnorm = normalize(x,self.xmin_nonorm, self.xmax_nonorm)
                self.xmin = [np.min(col) for col in np.vstack(xnorm).T]
                self.xmax = [np.max(col) for col in np.vstack(xnorm).T]
                
            case 'none':
                xnorm = x
                self.xmin_nonorm, self.xmax_nonorm = self.xmin, self.xmax
            case _:
                raise Warning('Invalid normx')
        # normalize y
        self.ymin_nonorm, self.ymax_nonorm = self.ymin, self.ymax
        match self.ynorm:
            case 'norm':
#                if (self.ymin_nonorm == None) and (self.ymax_nonorm == None):
#                    self.ymin_nonorm = [np.min(col) for col in np.vstack(y).T]
#                    self.ymax_nonorm = [np.max(col) for col in np.vstack(y).T]
#                
#                if np.allclose(self.ymin, np.zeros(np.shape(self.ymin))) and np.allclose(self.ymax, np.full(np.shape(self.ymax), 1)):
#                    ynorm = y
#                else:
#                    ynorm = normalize(y, self.ymin, self.ymax)
#                    self.ymin = [0 for _ in range(self.dimy)]
#                    self.ymax = [1 for _ in range(self.dimy)]
#                assert np.all(np.array([np.max(ye) for ye in ynorm]) <= np.asarray(self.ymax)), f'{[np.max(ye) for ye in y]}'
#                assert np.all(np.array([np.min(ye) for ye in ynorm]) >= np.asarray(self.ymin)), f'{[np.min(ye) for ye in y]}'
                if np.allclose(self.ymin, np.zeros(np.shape(self.ymin))) and np.allclose(self.ymax, np.full(np.shape(self.ymax), 1)):
                    ynorm = y
                else:
                    ynorm = normalize(y, self.ymin, self.ymax)
                    self.ymin = [0 for _ in range(self.dimy)]
                    self.ymax = [1 for _ in range(self.dimy)]
                assert np.all(np.array([np.max(ye) for ye in ynorm]) <= np.asarray(self.ymax)), f'{[np.max(ye) for ye in y]}'
                assert np.all(np.array([np.min(ye) for ye in ynorm]) >= np.asarray(self.ymin)), f'{[np.min(ye) for ye in y]}'

 
            case 'std':
                if (self.ymin_nonorm == None) and (self.ymax_nonorm == None):
                    self.ymin_nonorm = [np.min(col) for col in np.vstack(y).T]
                    self.ymax_nonorm = [np.max(col) for col in np.vstack(y).T]
                ynorm = standardize(y)
                self.ymin = [np.min(col) for col in np.vstack(ynorm).T]
                self.ymax = [np.max(col) for col in np.vstack(ynorm).T]
            case 'scale':
                self.ymin_nonorm = [np.min(col) for col in np.vstack(y).T]
                self.ymax_nonorm = [np.max(col) for col in np.vstack(y).T]
                ynorm = normalize(y,self.ymin_nonorm, self.ymax_nonorm)
                self.ymin = [np.min(col) for col in np.vstack(ynorm).T]
                self.ymax = [np.max(col) for col in np.vstack(ynorm).T]
            case 'none':
                ynorm = y
            case _:
                raise Warning('Invalid normy')
        assert len(self.xmin) == self.dimx
        assert len(self.ymax) == self.dimy
        return xnorm, ynorm
    
    def _train_val_split(self, x, y):
        """Split x and y into train and val sets.

        Args:
            x: list[np.array] (episodes, steps, dim_x)
            y: list[np.array] (episodes, steps, dim_y)

        Returns: 
            xtrain, ytrain, xval, yval: list[np.array] (episodes, steps, dim)
        """
        # rng = Generator(PCG64(seed=self.rseed))
        # train = rng.integers(low=0, high=self.n_episodes, size=self.n_train)
        xtrain = x[self.nvalepisodes:]
        ytrain = y[self.nvalepisodes:]
        if self.train_equal_val:
            xval = xtrain
            yval = ytrain
        else:
            xval = x[:self.nvalepisodes]
            yval = y[:self.nvalepisodes]
        return xtrain, ytrain, xval, yval
    
    def _xy_reactor(self):
        """Load raw reactor data from Siemens.

        Returns:
            x: list[np.array] (episodes, steps, dim_x)
            y: list[np.array] (episodes, steps, dim_y)
        """
        rng = Generator(PCG64(seed=self.rseed_data))
        # load and format dataframe
        df = pd.read_hdf(self.data_path, key='df')
        df = df.drop(['T_jout ValueY', 'm_M ValueY', 'm_P ValueY', 'UA ValueY', 'Q_Reac ValueY', 'm_M_setpoint ValueY'], axis=1)
        df = df.astype({"episode": int})
        # get episodes number of episodes
        episodes = df['episode'].unique() # list of episodes
        if self.nepisodes_tt > len(episodes):
            before = self.nepisodes_tt
            self.nepisodes = int(len(episodes) * .8)
            self.nvalepisodes = int(len(episodes) - self.nepisodes)
            self.nepisodes_tt = self.nepisodes + self.nvalepisodes
            print(f'''
            Warning: There are only {len(episodes)} episodes in the dataset. 
            I am setting the number of training episodes + testing episodes from {before} to {self.nepisodes_tt}
            ''')
        # method1, with duplicates
        # samples = rng.integers(low=0, high=np.max(episodes)+1, size=self.nepisodes_tt) # list of indices
        # method2, no duplicates
        samples = rng.choice(episodes, size=self.nepisodes_tt, replace=False, shuffle=True)
        # method3, no duplicates
        # rng.shuffle(episodes, axis=0)
        # samples = episodes[:self.nepisodes_tt]
        x, y = [], []
        # print(df.head())
        self.episodeids = samples
        for i in samples:
            df_e = df.loc[df['episode'] == i]
            x.append(np.asarray(df_e['T_R_setpoint ValueY']).reshape(df_e.shape[0], self.dimx))
            y.append(np.asarray(df_e['T_R ValueY']).reshape(df_e.shape[0], self.dimy))
        return x, y

    def _x_narma_smooth(self):  
        """Smooth input for NARMA task.

        https://arxiv.org/pdf/2211.02612.pdf

        Returns:
            x: list[np.array] (episodes, steps, dim_x)
        """
        rng = Generator(PCG64(seed=self.rseed_data))
        xs = [] 
        a, b, c, p = 2.11, 3.73, 4.11, 100
        for _ in range(self.nepisodes_tt):
            # xe = np.zeros(shape=(self.steps, 1))
            # for t in range(self.steps): 
            #     xe[t] = 0.1 * (rng.normal(loc=1., scale=0.01) * (
            #         np.sin(2*np.pi * a * t / p * rng.normal(loc=1., scale=0.1)) \
            #         * np.sin(2*np.pi*b*t/p *rng.normal(loc=1., scale=0.1)) \
            #         * np.sin(2*np.pi*c*t/p *rng.normal(loc=1., scale=0.1)) \
            #         + rng.normal(loc=1., scale=0.01)))
            t_start = int(rng.integers(1000))
            tsteps = np.arange(t_start, t_start + self.steps, 1, dtype=int)
            # xe = 0.1 * (np.sin(2 * np.pi * a * tsteps / p) \
            #     * np.sin(2 * np.pi * b * tsteps / p) \
            #     * np.sin(2 * np.pi * c * tsteps / p) \
            #     + 1)
            xe = np.zeros(shape=(self.steps, 1))
            for tstep, t in enumerate(tsteps):
                xe[tstep] = 0.1 * (np.sin(2 * np.pi * a * t / p) \
                    * np.sin(2 * np.pi * b * t / p) \
                    * np.sin(2 * np.pi * c * t / p) \
                    + 1)
            xs.append(xe)
        return xs
    
    def _x_narma_seq(self):  
            """Smooth input for NARMA task, but always begins at t =0 

            https://arxiv.org/pdf/2211.02612.pdf

            Returns:
                x: list[np.array] (episodes, steps, dim_x)
            """
            rng = Generator(PCG64(seed=self.rseed_data))
            xs = [] 
            a, b, c, p = 2.11, 3.73, 4.11, 100
            for ep in range(self.nepisodes_tt):
                # xe = np.zeros(shape=(self.steps, 1))
                # for t in range(self.steps): 
                #     xe[t] = 0.1 * (rng.normal(loc=1., scale=0.01) * (
                #         np.sin(2*np.pi * a * t / p * rng.normal(loc=1., scale=0.1)) \
                #         * np.sin(2*np.pi*b*t/p *rng.normal(loc=1., scale=0.1)) \
                #         * np.sin(2*np.pi*c*t/p *rng.normal(loc=1., scale=0.1)) \
                #         + rng.normal(loc=1., scale=0.01)))
                t_start = self.steps*ep
                tsteps = np.arange(t_start, t_start + self.steps, 1, dtype=int)
                # xe = 0.1 * (np.sin(2 * np.pi * a * tsteps / p) \
                #     * np.sin(2 * np.pi * b * tsteps / p) \
                #     * np.sin(2 * np.pi * c * tsteps / p) \
                #     + 1)
                xe = np.zeros(shape=(self.steps, 1))
                for tstep, t in enumerate(tsteps):
                    xe[tstep] = 0.1 * (np.sin(2 * np.pi * a * t / p) \
                        * np.sin(2 * np.pi * b * t / p) \
                        * np.sin(2 * np.pi * c * t / p) \
                        + 1)
                xs.append(xe)
            return xs


    def _x_narma_random(self): 
        """Random input for NARMA task drawn from (0, 0.5).

        https://www.nature.com/articles/s41378-021-00313-7

        Returns:
            x: list[np.array] (episodes, steps, dim_x)
        """
        rng = Generator(PCG64(seed=self.rseed_data))
        xs = [rng.random((self.steps, self.dimx))*0.5 for _ in range(self.nepisodes_tt)]
        return xs  

    def _y_narma(self, xs, n: int = 5):
        """Nonlinear autoregressive moving average task.

        https://www.nature.com/articles/s41378-021-00313-7
        https://arxiv.org/pdf/2211.02612.pdf

        Args:
            xs (list[np.array]): (episodes, steps, dim_x)
            n (int): correlation to past ~ memory length.
        
        Returns:
            y (list[np.array]): (episodes, steps, dim_y)
        """   
        # y 
        a, b, c, d = 0.3, 0.05, 1.5, 0.1
        ys = []
        for xe in xs: # loop over episodes
            # xe[:n] = 0
            xe = np.vstack([np.zeros((n, self.dimx)), xe])
            ye = np.zeros(shape=(self.steps+n, self.dimy))
            # https://arxiv.org/pdf/2211.02612.pdf
            for t in range(n, self.steps+n): # (steps, dimx)
                try:
                    ye[t] = \
                        (a * ye[t-1]) \
                        + (b * ye[t-1, :] * np.sum(ye[t-n:t], axis=0)) \
                        + (c * xe[t-1] * xe[t-n]) \
                        + d
                except Exception as e:
                    print(e)
                    print(
                        f"""
                        {ye[t]} = 
                        (a * {ye[t-1]}) 
                        + (b * {ye[t-1, :]} * {np.sum(ye[t-n:t], axis=0)}) 
                        + (c * {xe[t-1]} * {xe[t-n]}) 
                        + d
                        sum: {ye[t-n:t]}
                        """
                    )
            ys.append(ye[n:])
        assert np.shape(xs[0])[0] == self.steps, f'{np.shape(xs[0])}, {self.steps}'
        assert np.shape(ys[0])[0] == self.steps, f'{np.shape(ys[0])}, {self.steps}'
        return ys
    
    def _y_narma2(self, xs, n: int = 2):
        """Nonlinear autoregressive moving average task. Taken from tutorial.


        Args:
            xs (list[np.array]): (episodes, steps, dim_x)
            n (int): correlation to past ~ memory length.
        
        Returns:
            y (list[np.array]): (episodes, steps, dim_y)
        """   
        # y 
        a, b, c, d = 0.4, 0.4, 0.6, 0.1
        ystart=0.195
        ys = []
        for xe in xs: # loop over episodes
            # xe[:n] = 0
            xe = np.vstack([np.zeros((n, self.dimx)), xe])
            ye = np.zeros(shape=(self.steps+n, self.dimy))
            ye[0:2,:]=ystart
            # https://arxiv.org/pdf/2211.02612.pdf
            for t in range(n, self.steps+n): # (steps, dimx)
                try:
                    ye[t] = \
                        (a * ye[t-1]) \
                        + (b * ye[t-1, :] * ye[t-2, :]) \
                        + (c * xe[t-1]**3) \
                        + d
                except Exception as e:
                    print(e)
                    print(
                        f"""
                        {ye[t]} = 
                        (a * {ye[t-1]}) 
                        + (b * {ye[t-1, :]} * {np.sum(ye[t-n:t], axis=0)}) 
                        + (c * {xe[t-1]} * {xe[t-n]}) 
                        + d
                        sum: {ye[t-n:t]}
                        """
                    )
            ys.append(ye[n:])
        assert np.shape(xs[0])[0] == self.steps, f'{np.shape(xs[0])}, {self.steps}'
        assert np.shape(ys[0])[0] == self.steps, f'{np.shape(ys[0])}, {self.steps}'
        return ys

    def _x_smoothfilter(self) -> list:
        """By Andreas Burger.

        A Savitzky-Golay filter and moving average applied to a
        random sequence of floats uniformly picked from (0,1).

        Returns:
            list: (episodes, steps, dim_x)
        """
        w1 = 10
        w2 = 10
        rng = Generator(PCG64(seed=self.rseed_data))
        return [np.asarray([
            savgol_filter(
                moving_average(
                    rng.random(self.steps+w1-1), 
                    w=w1), 
                window_length=w2, 
                polyorder=2) 
            for _ in range(self.dimx)]).reshape((self.steps, self.dimx)) 
            for _ in range(self.nepisodes_tt)]
    
    def x_binary(self):
        """Random sequence of {0, 1}.

        Returns:
            list: (episodes, steps, dim_x)
        """
        rng = Generator(PCG64(seed=self.rseed_data))
        return [np.asarray([rng.integers(2, size=(self.steps))
                for _ in range(self.dimx)]).reshape((self.steps, self.dimx))
                for _ in range(self.nepisodes_tt)]
    
    def x_random(self):
        """Random sequence uniformly in (0, 1).

        Returns:
            list: (episodes, steps, dim_x)
        """
        rng = Generator(PCG64(seed=self.rseed_data))
        return [np.asarray([rng.random(size=(self.steps))
                for _ in range(self.dimx)]).reshape((self.steps, self.dimx))
                for _ in range(self.nepisodes_tt)]
        
    def x_const(self):
        """Constant, randomly in (0, 1).

        Returns:
            list: (episodes, steps, dim_x)
        """
        rng = Generator(PCG64(seed=self.rseed_data))
        return [np.ones((self.steps, self.dimx)) * rng.random()
                for _ in range(self.nepisodes_tt)]

    def x_linear(self):
        """Line from x(t=0)=0 to x(t=max)=1.

        Returns:
            list: (episodes, steps, dim_x)
        """
        return [
            np.asarray([
                np.linspace(start=0, stop=1, num=self.steps)
            for _ in range(self.dimx)]).reshape((self.steps, self.dimx))
        for _ in range(self.nepisodes_tt)]