import os

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.modules import Module
from functools import partial
from typing import Callable, Tuple, Union, Tuple, Union, Any

def import_abspy(name="models", path="classification/"):
    import sys
    import importlib
    path = os.path.abspath(path)
    assert os.path.isdir(path)
    sys.path.insert(0, path)
    module = importlib.import_module(name)
    sys.path.pop(0)
    return module

build = import_abspy(
    "models", 
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../classification/"),
)
selective_scan_flop_jit: Callable = build.vmamba.selective_scan_flop_jit
VSSM: nn.Module = build.vmamba.VSSM
Backbone_VSSM: nn.Module = build.vmamba.Backbone_VSSM

supported_ops={
    "aten::silu": None, # as relu is in _IGNORED_OPS
    "aten::neg": None, # as relu is in _IGNORED_OPS
    "aten::exp": None, # as relu is in _IGNORED_OPS
    "aten::flip": None, # as permute is in _IGNORED_OPS
    "prim::PythonOp.SelectiveScanFn": selective_scan_flop_jit, # latter
    "prim::PythonOp.SelectiveScanOflex": selective_scan_flop_jit, # latter
    "prim::PythonOp.SelectiveScanCore": selective_scan_flop_jit, # latter
    "prim::PythonOp.SelectiveScan": selective_scan_flop_jit, # latter
}

def mmengine_flop_count(model: nn.Module = None, input_shape = (3, 224, 224), show_table=False, show_arch=False, _get_model_complexity_info=False):
    from mmengine.analysis.print_helper import is_tuple_of, FlopAnalyzer, ActivationAnalyzer, parameter_count, _format_size, complexity_stats_table, complexity_stats_str
    from mmengine.analysis.jit_analysis import _IGNORED_OPS
    from mmengine.analysis.complexity_analysis import _DEFAULT_SUPPORTED_FLOP_OPS, _DEFAULT_SUPPORTED_ACT_OPS
    from mmengine.analysis import get_model_complexity_info as mm_get_model_complexity_info
    
    # modified from mmengine.analysis
    def get_model_complexity_info(
        model: nn.Module,
        input_shape: Union[Tuple[int, ...], Tuple[Tuple[int, ...], ...],
                        None] = None,
        inputs: Union[torch.Tensor, Tuple[torch.Tensor, ...], Tuple[Any, ...],
                    None] = None,
        show_table: bool = True,
        show_arch: bool = True,
    ):
        if input_shape is None and inputs is None:
            raise ValueError('One of "input_shape" and "inputs" should be set.')
        elif input_shape is not None and inputs is not None:
            raise ValueError('"input_shape" and "inputs" cannot be both set.')

        if inputs is None:
            device = next(model.parameters()).device
            if is_tuple_of(input_shape, int):  # tuple of int, construct one tensor
                inputs = (torch.randn(1, *input_shape).to(device), )
            elif is_tuple_of(input_shape, tuple) and all([
                    is_tuple_of(one_input_shape, int)
                    for one_input_shape in input_shape  # type: ignore
            ]):  # tuple of tuple of int, construct multiple tensors
                inputs = tuple([
                    torch.randn(1, *one_input_shape).to(device)
                    for one_input_shape in input_shape  # type: ignore
                ])
            else:
                raise ValueError(
                    '"input_shape" should be either a `tuple of int` (to construct'
                    'one input tensor) or a `tuple of tuple of int` (to construct'
                    'multiple input tensors).')

        flop_handler = FlopAnalyzer(model, inputs).set_op_handle(**supported_ops)
        # activation_handler = ActivationAnalyzer(model, inputs)

        flops = flop_handler.total()
        # activations = activation_handler.total()
        params = parameter_count(model)['']

        flops_str = _format_size(flops)
        # activations_str = _format_size(activations)
        params_str = _format_size(params)

        if show_table:
            complexity_table = complexity_stats_table(
                flops=flop_handler,
                # activations=activation_handler,
                show_param_shapes=True,
            )
            complexity_table = '\n' + complexity_table
        else:
            complexity_table = ''

        if show_arch:
            complexity_arch = complexity_stats_str(
                flops=flop_handler,
                # activations=activation_handler,
            )
            complexity_arch = '\n' + complexity_arch
        else:
            complexity_arch = ''

        return {
            'flops': flops,
            'flops_str': flops_str,
            # 'activations': activations,
            # 'activations_str': activations_str,
            'params': params,
            'params_str': params_str,
            'out_table': complexity_table,
            'out_arch': complexity_arch
        }
    
    if _get_model_complexity_info:
        return get_model_complexity_info

    model.eval()
    analysis_results = get_model_complexity_info(
        model,
        input_shape,
        show_table=show_table,
        show_arch=show_arch,
    )
    flops = analysis_results['flops_str']
    params = analysis_results['params_str']
    # activations = analysis_results['activations_str']
    out_table = analysis_results['out_table']
    out_arch = analysis_results['out_arch']
    
    if show_arch:
        print(out_arch)
    
    if show_table:
        print(out_table)
    
    split_line = '=' * 30
    print(f'{split_line}\nInput shape: {input_shape}\t'
          f'Flops: {flops}\tParams: {params}\t'
        #   f'Activation: {activations}\n{split_line}'
    , flush=True)
    # print('!!!Only the backbone network is counted in FLOPs analysis.')
    # print('!!!Please be cautious if you use the results in papers. '
    #       'You may need to check if all ops are supported and verify that the '
    #       'flops computation is correct.')


def fvcore_flop_count(model: nn.Module, inputs=None, input_shape=(3, 224, 224), show_table=False, show_arch=False):
    from fvcore.nn.parameter_count import parameter_count as fvcore_parameter_count
    from fvcore.nn.flop_count import flop_count, FlopCountAnalysis, _DEFAULT_SUPPORTED_OPS
    from fvcore.nn.print_model_statistics import flop_count_str, flop_count_table
    from fvcore.nn.jit_analysis import _IGNORED_OPS
    from fvcore.nn.jit_handles import get_shape, addmm_flop_jit
    
    if inputs is None:
        assert input_shape is not None
        if len(input_shape) == 1:
            input_shape = (1, 3, input_shape[0], input_shape[0])
        elif len(input_shape) == 2:
            input_shape = (1, 3, *input_shape)
        elif len(input_shape) == 3:
            input_shape = (1, *input_shape)
        else:
            assert len(input_shape) == 4

        inputs = (torch.randn(input_shape).to(next(model.parameters()).device),)


    model.eval()

    Gflops, unsupported = flop_count(model=model, inputs=inputs, supported_ops=supported_ops)
    
    flops_table = flop_count_table(
        flops = FlopCountAnalysis(model, inputs).set_op_handle(**supported_ops),
        max_depth=100,
        activations=None,
        show_param_shapes=True,
    )

    flops_str = flop_count_str(
        flops = FlopCountAnalysis(model, inputs).set_op_handle(**supported_ops),
        activations=None,
    )

    if show_arch:
        print(flops_str)

    if show_table:
        print(flops_table)
    
    print(Gflops.items())

    params = fvcore_parameter_count(model)[""]
    flops = sum(Gflops.values())
    print("GFlops: ", flops, "Params: ", params, flush=True)
    return params, flops


# ==============================


def build_model_vssm(depths=[2, 2, 9, 2], embed_dim=96):
    model = VSSM(depths=depths, dims=embed_dim, d_state=16, dt_rank="auto", ssm_ratio=2.0, mlp_ratio=0.0, downsample="v1")
    def forward_backbone(self: VSSM, x):
        x = self.patch_embed(x)
        for layer in self.layers:
            x = layer(x)
        return x

    model.forward = partial(forward_backbone, model)
    try:
        del model.norm
    except:
        pass
    try:
        del model.head
    except:
        pass
    try:
        del model.classifier
    except:
        pass
    model.cuda().eval()
    return model
    

def vssm_flops(core="fvcore"):
    _flops_count = fvcore_flop_count
    if core.startswith("mm"):
        _flops_count = mmengine_flop_count
    build_vmamba = build_model_vssm
    _flops_count(build_vmamba(depths=[2, 2, 9, 2], embed_dim=96), input_shape=(3, 224, 224))
    _flops_count(build_vmamba(depths=[2, 2, 27, 2], embed_dim=96), input_shape=(3, 224, 224))
    _flops_count(build_vmamba(depths=[2, 2, 27, 2], embed_dim=128), input_shape=(3, 224, 224))
    # 4.46 + 22.1, 9.11 + 43.6, 15.2 + 75.2


def mmdet_mmseg_vssm():
    from mmengine.model import BaseModule
    # from mmdet.registry import MODELS as MODELS_MMDET
    # from mmseg.registry import MODELS as MODELS_MMSEG

    # @MODELS_MMSEG.register_module()
    # @MODELS_MMDET.register_module()
    class MM_VSSM(BaseModule, Backbone_VSSM):
        def __init__(self, *args, **kwargs):
            BaseModule.__init__(self)
            Backbone_VSSM.__init__(self, *args, **kwargs)


def mmseg_flops(config=None, input_shape=(3, 512, 2048)):
    from mmengine.config import Config
    from mmengine.runner import Runner

    cfg = Config.fromfile(config)
    cfg["work_dir"] = "/tmp"
    runner = Runner.from_cfg(cfg)
    model = runner.model.cuda()
    
    fvcore_flop_count(model, input_shape=input_shape)




    
if __name__ == '__main__':
    if False:
        print("fvcore flops count for vssm ====================", flush=True)
        vssm_flops()
        print("mmengine flops count for vssm ====================", flush=True)
        vssm_flops("mm") # same as fvcore


    fvcore_flop_count(model, input_shape=input_shape)


    
