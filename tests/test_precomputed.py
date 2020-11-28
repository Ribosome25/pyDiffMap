# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 13:15:38 2020

@author: ruibzhan
"""

import pytest
import pydiffmap as dm
import numpy as np


class TestPrecomputed(object):
#    @pytest
    def test_precomputed_kernel_and_distance(self):
        #np.random.seed(7)
        data = np.random.randn(20, 10)  # dimension: 20 x 10
        corr = np.corrcoef(data, rowvar=True)
        
        dist_matr = 1 - corr
        eigvs = np.linalg.eigvals(1 - corr)
        #%% right one
        dfmap = dm.diffusion_map.DiffusionMap.from_sklearn(n_evecs=2, metric='correlation', alpha = 0.5, epsilon = 1, k = 32)
        # Init a kernel.Kernel obj and reuse the cls(__init__). Return a Dm obj with attrb kernel = that kernel. 
        dm_xy = dfmap.fit_transform(data)
        precompuL = dfmap.kernel_matrix.toarray()
        precompuDist = dfmap.local_kernel.scaled_dists.toarray()
        assert(np.allclose(precompuDist, dist_matr))
        #%% pre comp kernel
        
        dfmap2 = dm.diffusion_map.DiffusionMap.from_sklearn(n_evecs=2,metric = 'correlation',
                                                           alpha = 0.5, epsilon = 1, k = 32,
                                                           kernel_type='precomputed')
        dm_xy2 = dfmap2.fit_transform_from_precomputed_kernel(precompuL)
        
        dfmap20 = dm.diffusion_map.DiffusionMap.from_sklearn(n_evecs=2,metric = 'correlation',
                                                           alpha = 0.5, epsilon = 1, k = 32,
                                                           kernel_type='precomputed')
        dm_xy20 = dfmap2.fit_transform(precompuL)
        #%% pre-comp distance
        dfmap3 = dm.diffusion_map.DiffusionMap.from_sklearn(n_evecs=2,metric = 'precomputed',
                                                           alpha = 0.5, epsilon = 1, k = 32,
                                                           kernel_type='gaussian')
        dm_xy3 = dfmap3.fit_transform(precompuDist)
        L_3 = dfmap3.kernel_matrix.toarray()
        assert(np.allclose(L_3, precompuL))
        #%%
        def _check_equal(x, y):
            for ii in range(x.shape[1]):
                assert(np.allclose(x[:, ii], y[:, ii]) or np.allclose(x[:, ii], - y[:, ii]))
            return True
        
        assert(_check_equal(dm_xy, dm_xy2))
        assert(_check_equal(dm_xy, dm_xy20))
        assert(_check_equal(dm_xy, dm_xy3))

#%%
