"""
INN trainer
----------------------------------------------------------------------------------
Copyright (C) 2024  Chanwook Park
 Northwestern University, Evanston, Illinois, US, chanwookpark2024@u.northwestern.edu
"""

import jax
import jax.numpy as jnp
from jax.nn.initializers import uniform
from jax import config
config.update("jax_enable_x64", True)
import copy
from functools import partial
from typing import (Any, Callable, Iterable, List, Optional, Sequence, Tuple,
                    Union)
from jax.scipy.interpolate import RegularGridInterpolator
from .Interpolator import GaussianInterpolator, LinearInterpolator, NonlinearInterpolator

class INN_linear:
    def __init__(self, grid, config):
        """ 1D linear interpolation
        --- input --- 
        grid: (J,) 1D vector of the grid
        values: (J,) 1D vector of nodal values
        """
        self.grid = grid
        self.config = config
        if isinstance(grid, list):
            self.interpolate = [LinearInterpolator(grid_idm) for grid_idm in grid]
        else:
            self.interpolate = LinearInterpolator(grid)
        
    @partial(jax.jit, static_argnames=['self'])
    def get_Ju_idata_imd_idm_ivar(self, x_idata_idm, u_imd_idm_ivar_nds):
        """ compute interpolation for a single mode, 1D function
        --- input ---
        x_idata_idm: scalar, jnp value / this can be any input
        u_imd_idm_ivar_nds: (J,) jnp 1D array
        --- output ---
        Ju_idata_imd_idm_ivar: scalar, 1D interpolated value
        """
        Ju_idata_imd_idm_ivar = self.interpolate(x_idata_idm, u_imd_idm_ivar_nds)
        return Ju_idata_imd_idm_ivar
    get_Ju_idata_imd_idm_vars = jax.vmap(get_Ju_idata_imd_idm_ivar, in_axes = (None,None,0)) # input: scalar, (var,J) / output: (var,)
    get_Ju_idata_imd_dms_vars = jax.vmap(get_Ju_idata_imd_idm_vars, in_axes = (None,0,0)) # input: (dim,), (dim,var,J) / output: (dim,var)
    get_Ju_idata_mds_dms_vars = jax.vmap(get_Ju_idata_imd_dms_vars, in_axes = (None,None,0)) # input: (dim,), (M,dim,var,J) / output: (M,dim,var)

    @partial(jax.jit, static_argnames=['self', 'interpolate'])
    def get_Ju_idata_imd_idm_ivar_with_interp(self, x_idata_idm, interpolate, u_imd_idm_ivar_nds):
        return interpolate(x_idata_idm, u_imd_idm_ivar_nds)
    get_Ju_idata_imd_idm_vars_with_interp = jax.vmap(get_Ju_idata_imd_idm_ivar_with_interp, in_axes=(None,None,None,0))
    get_Ju_idata_mds_idm_vars_with_interp = jax.vmap(get_Ju_idata_imd_idm_vars_with_interp, in_axes=(None,None,None,0))

    ## CP decomposition
    # def get_Ju_idata_imd(self, x_idata_dms, u_imd_dms_vars_nds):
    #     # x_idata_dms: (dim,)
    #     # x_dms_nds: (dim, nnode)
    #     # u_imd_dms_vars_nds: (dim, var, nnode)
        
    #     Ju_idata_imd_dims_vars = self.get_Ju_idata_imd_dms_vars(x_idata_dms, u_imd_dms_vars_nds) # output: (dim, var)
    #     Ju_idata_imd = jnp.prod(Ju_idata_imd_dims_vars, axis=0)
    #     return Ju_idata_imd # (var,)
    # get_Ju_idata_mds = jax.vmap(get_Ju_idata_imd, in_axes = (None,None,0)) # output: (mds,var)

    # def get_Ju_idata(self, x_idata_dms, u_mds_dms_vars_nds):
    #     Ju_idata_mds = self.get_Ju_idata_mds(x_idata_dms, u_mds_dms_vars_nds) # (mds,var)
    #     Ju_idata = jnp.sum(Ju_idata_mds, axis=0) # returns (var,)
    #     return Ju_idata

    # # @partial(jax.jit, static_argnames=[]) # jit necessary
    # # @jax.jit
    # @partial(jax.jit, static_argnames=['self'])
    # def forward(self, params, x_idata):
    #     """ Prediction function
    #         run one forward pass on given input data
    #         --- input ---
    #         params: u_mds_dms_vars_nds, (nmode, dim, var, nnode)
    #         x_dms_nds: nodal coordinates (dim, nnode)
    #         x_idata: x_idata_dms (dim,)
    #         --- return ---
    #         predicted output (var,)
    #     """
    #     pred = self.get_Ju_idata(x_idata, params)
    #     return pred
    # v_forward = jax.vmap(forward, in_axes=(None,None, 0)) # returns (ndata,)
    # vv_forward = jax.vmap(v_forward, in_axes=(None,None, 0)) # returns (ndata,)

    def tucker(self, G, factors):
        """ serior computation of tucker decomposition 
        --- input ---
        G: core tensor, (M, M, ..., M) 
        factors: factor matrices, (dim, M)"""
        for factor in factors:
            G = jnp.tensordot(G, factor, axes=[0,0])
        return jnp.squeeze(G)
    v_tucker = jax.vmap(tucker, in_axes=(None,None,0)) # returns (var,)

    def tucker_list(self, G, factors_dms):
        """Tucker contraction for per-dimension ranks.

        factors_dms is a list of arrays with shape (R_d, var).
        """
        preds = []
        for ivar in range(factors_dms[0].shape[1]):
            pred_ivar = G
            for factor in factors_dms:
                pred_ivar = jnp.tensordot(pred_ivar, factor[:, ivar], axes=[0, 0])
            preds.append(jnp.squeeze(pred_ivar))
        return jnp.stack(preds)

    def get_tucker_factors_list(self, x_idata, params_dms):
        factors_dms = []
        for idm, (interpolate, params_idm) in enumerate(zip(self.interpolate, params_dms)):
            factors_idm = self.get_Ju_idata_mds_idm_vars_with_interp(
                x_idata[idm],
                interpolate,
                params_idm,
            )
            factors_dms.append(factors_idm)
        return factors_dms


    def forward(self, params, x_idata):
        """ Prediction function
            run one forward pass on given input data
            --- input ---
            params: u_mds_dms_vars_nds, (nmode, dim, var, nnode)
            x_dms_nds: nodal coordinates (dim, nnode)
            x_idata: x_idata_dms (dim,)
            --- return ---
            predicted output (var,)
        """
        if self.config['TD_type']=='CP':
            pred = self.get_Ju_idata_mds_dms_vars(x_idata, params) # output: (M,dim,var)
            pred = jnp.prod(pred, axis=1) # output: (M,var)
            pred = jnp.sum(pred, axis=0) # output: (var,)
        if self.config['TD_type']=='Tucker':
            
            G = params[0]
            if isinstance(params[1], list):
                factors = self.get_tucker_factors_list(x_idata, params[1])
                pred = self.tucker_list(G, factors)
            else:
                factors = self.get_Ju_idata_mds_dms_vars(x_idata, params[1]) # output: (M,dim,var)
                # for factor in pred.transpose(1,0,2): # factor: (M,var)
                #     G = jnp.tensordot(G, factor, axes=[0,0])
                factors = factors.transpose(2,1,0) # (var,dim,M)
                # print(factors.shape)
                pred = self.v_tucker(G, factors)

        return pred

    v_forward = jax.vmap(forward, in_axes=(None,None, 0)) # returns (ndata,)
    vv_forward = jax.vmap(v_forward, in_axes=(None,None, 0)) # returns (ndata,)
        





class INN_nonlinear(INN_linear):
    def __init__(self, grid, config):
        super().__init__(grid, config) # prob being dropout probability

        if isinstance(grid, list):
            length_scale = self.config['MODEL_PARAM'].get('length_scale')
            self.interpolate = []
            for idm, grid_idm in enumerate(grid):
                config_idm = copy.deepcopy(self.config)
                if isinstance(length_scale, list):
                    config_idm['MODEL_PARAM']['length_scale'] = length_scale[idm]
                self.interpolate.append(NonlinearInterpolator(grid_idm, config_idm))
        else:
            self.interpolate = NonlinearInterpolator(grid, self.config)


class INN_gaussian(INN_linear):
    def __init__(self, grid, config):
        super().__init__(grid, config)

        sigma_factor = self.config['MODEL_PARAM'].get('sigma_factor', 1.2)

        if isinstance(grid, list):
            self.interpolate = []
            for grid_idm in grid:
                self.interpolate.append(GaussianInterpolator(grid_idm, sigma_factor))
        else:
            self.interpolate = GaussianInterpolator(grid, sigma_factor)


## MLP

def relu(x):
    return jnp.maximum(0, x)

@partial(jax.jit, static_argnames=['activation'])
def forward_MLP(params, activation, x_idata):
    # per-example predictions
    activations = x_idata
    for w, b in params[:-1]:
        outputs = jnp.dot(w, activations) + b
        if activation == 'relu':
            activations = relu(outputs)
        elif activation == 'sigmoid':
            activations = jax.nn.sigmoid(outputs)
    final_w, final_b = params[-1]
    return jnp.dot(final_w, activations) + final_b
v_forward_MLP = jax.vmap(forward_MLP, in_axes=(None,None, 0)) # returns (ndata,)
vv_forward_MLP = jax.vmap(v_forward_MLP, in_axes=(None,None, 0)) # returns (ndata,)