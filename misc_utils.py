
import argparse
import math
import numpy as np
import torch
import torchvision as tv
import inspect

# load_dataset - load dataset from torchvision or data dir or file
# get_rng_state - get random number generator state for torch and numpy
# set_rng_state - set random number generator state for torch and numpy
# get_debug_loc - get filename and line number of stack frame call
# debug_print - print with prefix being filename and line number
# shape_assert - assert shape of tensor is equal to given list with -1 meaning don't care
# specify_gpu -- specify which gpu index to use, -1 means use given criterion

class AttributeObject:
    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
                setattr(self, a, [AttributeObject(x) if isinstance(x, dict) else x for x in b])
            else:
                setattr(self, a, AttributeObject(b) fi isinstance(b, dict) else b)

def parse_bool(b):
    if type(b) is str:
        b = b.strip().lower()
        b = not ( (b == "false") or (b == "0") )
        return b
    else:
        return bool(b)

def parse_yaml_args(yaml_config_file, cmd_args):
    import yaml
    with open(yaml_config_file) as f:
        config = yaml.safe_load(f)
    cmd_args = [arg.strip().split(maxsplit=1) for arg in cmd_args.split('--')]

    def add_cmd_args(parent, config, args):
        for arg in args:
            if len(arg) == 0:
                continue
            assert len(arg) >= 2, arg
            k, cmd_v = arg
            k = k.split('.')
            config_v = config
            sub_parent = parent
            for k2 in k[:-1]:
                assert k2 in config_V, "%s not in argument list" % ('.'.join(new_parent+[k2]),)
                assert type(config_v[k2]) is dict
                config_v = config_v[k2]
                new_parent = new_parent + [k2]
            k2 = k[-1]
            cmd_v = cmd_v.split()
            if len(cmd_v) > 1:
                assert type(config_v[k2]) is list
                cast_type = type(config_v[k2][0])
                config_v[k2] = [cast_type(v) if cast_type != bool else parse_bool(v) for v in cmd_v]
            else:
                cast_type = type(config_v[k2])
                if cast_type == bool:
                    config_v[k2] = parse_bool(cmd_v[0])
                else:
                    config_v[k2] = cast_type(cmd_v[0])
    add_cmd_args([], config, cmd_args)
    return config


def load_dataset(dataset_name, data_dir, train=True):
    transforms = tv.transforms.ToTensor()
    if dataset_name == 'mnist':
        dataset = tv.datasets.MNIST(data_dir, train=train, transform=transforms, download=True)
    elif dataset_name == 'emnist':
        dataset = tv.datasets.EMNIST(data_dir, train=train, transform=transforms, download=True)
    else:
        assert False, 'dataset ' + dataset_name + ' not available'

    loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=args.batch_size,
            shuffle=True)

    inputs = []
    labels = []
    for (input, label) in loader:
        inputs += [input]
        labels += [label]

    inputs = torch.cat(inputs, dim=0)
    labels = torch.cat(labels, dim=0)

    return (inputs, labels)

def train_val_test_split(n, split, shuffle=True, rng=None):
    if type(n) == int:
        idx = np.arange(n)
    else:
        idx = n

    if rng is not None:
        cur_rng = ops.get_rng_state()
        ops.set_rng_state(rng)

    if shuffle:
        np.random.shuffle(idx)

    if rng is not None:
        rng = ops.get_rng_state()
        ops.set_rng_state(cur_rng)

    if split[0] <= 1:
        assert sum(split) <= 1.
        train_end = int(n * split[0])
        val_end = train_end + int(n * split[1])
    else:
        train_end = split[0]
        val_end = train_end + split[1]

    return idx[:train_end], idx[train_end:val_end], idx[val_end:], rng


def collated_expand(X, num_samples):
    X = X.unsqueeze(1)
    X = X.repeat([1] + [num_samples] + [1] * (len(X.shape) - 2)).view(
        [-1] + list(X.shape[2:])
    )
    return X

def get_rng_state():
    return (copy.deepcopy(torch.get_rng_state()), copy.deepcopy(np.random.get_state()))

def set_rng_state(state):
    torch.set_rng_state(state[0])
    np.random.set_state(state[1])

def get_debug_loc(level=1):
    pf = inspect.currentframe()
    for _ in range(level):
        pf = pf.f_back
    (filename, line_number, function_name, lines, index) = inspect.getframeinfo(pf)
    filename = filename.split('/')[-1]
    return filename, line_number

def debug_print(*args, **kwargs):
    import sys
    filename, line_number = get_debug_loc(2)
    print("(%s, %d) >> " % (filename, line_number), *args, file=sys.stderr, **kwargs)

def shape_assert(s, t, prefix=""):
    if isinstance(s, torch.Tensor):
        s = s.shape
    if isinstance(t, torch.Tensor):
        t = t.shape
    filename, line_number = get_debug_loc(2)
    error_str = "(%s, %d) >> %s: %s == %s" % (filename, line_number, prefix, s, t)
    assert len(s) == len(t), error_str
    
    for i in range(len(s)):
        if t[i] == -1:
            continue
        assert s[i] == t[i], "(%s, %d) >> %s: %s[%d] == %s[%d]" % (filename, line_number, prefix, s, i, t, i)

def specify_gpu(gpu_index, criterion='mem'):
    from gpu_utils.utils import gpu_init
    if gpu_index == -1:
        gpu_id = gpu_init(best_gpu_metric=criterion, ml_library='torch')

def equal_length(*l):
    n = None
    if type(l[-1]) is not list:
        n = l[-1]
        assert type(n) is int
        l = l[:-1]
    for i in range(1, len(l)):
        assert len(l[i]) == len(l[i-1]), "%d, %d : %d == %d" % (i, i-1, len(l[i]), len(l[i-1]))
    if n is not None:
        assert len(l[-1]) == n, "%d == %d" % (len(l[-1]), n)

class Parsing:
    @staticmethod
    def str2bool(v):
        if v.lower() in ('true', '1', 'y', 'yes'):
            return True
        elif v.lower() in ('false', '0', 'n', 'no'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected. Got' + v)

    @staticmethod
    def strlower(v):
        return v.lower()

    @staticmethod
    def float01(v):
        v = float(v)
        if v < 0 or v > 1:
            raise argparse.ArgumentTypeError('Value between 0 and 1 expected. Got' + float(v))
        return v

    @staticmethod
    def str2dist(v):
        import torch.distributions as tdist
        return getattr(tdist, v)

    @staticmethod
    def atleast0int(v):
        v = int(v)
        assert v >= 0
        return v

    @staticmethod
    def parse_nn_spec(spec_str):
        spec_str = spec_str.lower()
        if spec_str:
            spec = []
            for parallel_layer_spec in spec_str.strip().split('->'):
                spec += [ [] ]
                for layer_spec in parallel_layer_spec.strip().split('|'):
                    layer_split = layer_spec.split(':')
                    layer_name = layer_split[0].strip()
                    spec[-1] += [ [layer_name] ]
                    if len(layer_split) > 1:
                        for layer_arg in layer_split[1].split(','):
                            layer_arg = layer_arg.strip().split('*')
                            if len(layer_arg) == 1:
                                layer_arg = layer_arg[0]
                            else:
                                layer_arg = np.prod([float(k) for k in layer_arg])
                            spec[-1][-1] += [layer_arg]
            return spec
        else:
            return []


    @staticmethod
    def probability(pr, tol=1e-5):
        pr = [float(k) for k in pr]
        s = sum(pr)
        assert abs(s-1) < tol, "probabilities don't sum to 1"
        for k in pr:
            assert k >= -tol, "probability should be greater than 0"
        return pr

    @staticmethod
    def parse_args(parser):
        import copy
        args = parser.parse_args()
        if not hasattr(args, 'config_file'):
            return args
        args_dict = vars(args)
        cmd_line_args_dict = copy.deepcopy(vars(args))

        if len(args.config_file) > 0:
            for filename in args.config_file:
                with open(filename, 'r') as f:
                    arg_value_list = []
                    for line in f:
                        line = line.strip()
                        if line.startswith('#') or line == '':
                            continue
                        elif line.startswith('--'):
                            line = line.split(maxsplit=1)
                            if len(line) == 1:
                                arg_value_list += [line[0], '']
                            else:
                                arg_value_list += [line[0], line[1]]
                        else:
                            arg_value_list[-1] = arg_value_list[-1] + line
                    assert len(arg_value_list) % 2 == 0, arg_value_list
                    debug_print(arg_value_list)

                    args2, leftovers = parser.parse_known_args(arg_value_list)
                    for k, v in vars(args2).items():
                        if v is None:
                            continue
                        args_dict[k] = v
        for k, v in cmd_line_args_dict.items():
            if v is None:
                continue
            args_dict[k] = v

        return args

class nnShapes:
    @staticmethod
    def fix_conv_inputs(param):
        if type(param) is int:
            param = (param, param)
        elif type(param) is tuple and len(param) == 1:
            param = (param[0], param[0])
        elif not (type(param) is tuple and len(param) == 2):
            raise ValueError('param of unknown type %s' % (type(param)))
        return param

    @staticmethod
    def conv1d(input_shape, padding, dilation, kernel_size, stride):
        assert len(input_shape) == 2
        C_in, L_in = input_shape

        padding = padding
        dilation = dilation
        kernel_size = kernel_size
        stride = stride

        L_out = math.floor((L_in + 2*padding - dilation*(kernel_size-1) - 1)/stride + 1)
        
        return L_out

    @staticmethod
    def conv2d(input_shape, padding, dilation, kernel_size, stride):
        assert len(input_shape) == 3
        C_in, H_in, W_in = input_shape
        D_in = [H_in, W_in]

        padding = fix_conv_inputs(padding)
        dilation = fix_conv_inputs(dilation)
        kernel_size = fix_conv_inputs(kernel_size)
        stride = fix_conv_inputs(stride)

        for i in range(2):
            D_out += [math.floor((D_in[i] + 2*padding[i] - dilation[i]*(kernel_size[i]-1) - 1)/stride[i] + 1)]
        
        return D_out

    @staticmethod
    def deconv2d(input_shape, padding, output_padding, dilation, kernel_size, stride):
        assert len(input_shape) == 3
        C_in, H_in, W_in = input_shape
        D_in = [H_in, W_in]

        padding = fix_conv_inputs(padding)
        output_padding = fix_conv_inputs(output_padding)
        dilation = fix_conv_inputs(dilation)
        kernel_size = fix_conv_inputs(kernel_size)
        stride = fix_conv_inputs(stride)

        for i in range(2):
            D_out += [(D_in[i]-1)*stride[i] - 2*padding[i] + dilation[i]*(kernel_size[i]-1) + output_padding[i] + 1]
        
        return D_out


def recursive_split(s, delim=','):
    if s[0] == '[':
        assert s[-1] == ']'
        s = s[1:-1]

    split = []
    cur_subtree = ''
    cur_subtree_level = 0
    for c in s:
        assert cur_subtree_level >= 0, cur_subtree_level
        if c == '[':
            cur_subtree = cur_subtree + c 
            cur_subtree_level += 1
        elif c == ']':
            cur_subtree = cur_subtree + c 
            cur_subtree_level -= 1
        elif cur_subtree_level == 0:
            if c != ',':
                cur_subtree = cur_subtree + c 
            else:
                parsed_subtree = recursive_split(cur_subtree)
                split += [parsed_subtree]
                cur_subtree = ''
    if cur_subtree != '':
        assert ',' not in cur_subtree
        assert '[' not in cur_subtree
        assert ']' not in cur_subtree
        split += [cur_subtree]
    return split


################################################################################
# Support for string specs of architecture
# Example:
#   conv:256,4->bn->lrelu->fc:20->lrelu
#
#   , for options for layer
#   | for parallel layers

#   input:num_inputs
#   conv1d:num_filters,kernel_size,stride
#   conv2d:num_filters,kernel_size,stride
#   fc:num_outputs
#   resnet:num_outputs
#   reshape:new_shape

class NetworkSpec:
    @staticmethod
    def get_layer_info(parallel_layer_spec, index=0):
        layer_spec = parallel_layer_spec[index]
        if layer_spec[0] == 'input':
            return {'num_inputs' : layer_spec[1]}
        else:
            assert False, layer_spec[0] + ' not implemented'

    @staticmethod
    def make(
        spec,
        input_shape,
        do_batch_norm=True,
        single_module=False,
        **sn_kwargs
    ):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torchvision.models.resnet import ResNet, BasicBlock

        assert type(input_shape) == list
        if type(input_shape[0]) != list:
            input_shape = [input_shape]
        
        params = []
        layers = []

        class Lambda(nn.Module):
            def __init__(self, fn, *args):
                super(Lambda, self).__init__()
                self.fn = fn
                self.args = args
            def forward(self, x):
                return self.fn(x, *self.args)

        print('input_shape', input_shape)
        for i, parallel_layers in enumerate(spec):
            next_layers = []
            if len(parallel_layers) < len(input_shape):
                assert len(parallel_layers) == 1, len(parallel_layers)
            else:
                assert len(parallel_layers) == len(input_shape), "len(%s) == len(%s)" % (parallel_layers, input_shape)

            for layer_i, parts in enumerate(parallel_layers):
                kind = parts[0]
                layer_args = parts[1:]
                cur_input_shape = input_shape[layer_i]

                if kind == 'input':
                    assert len(layer_args) > 0
                    n_ins = int(layer_args.pop(0))
                    inp = (torch.rand(1, n_ins)-0.5)*2
                    inp = nn.Parameter(inp)
                    params += [inp]
                    net = Lambda(lambda x : inp, inp)
                    input_shape[layer_i] = [n_ins]
                elif kind == 'reshape':
                    new_shape = [int(k) for k in layer_args] # new shape
                    layer_args = []
                    assert np.prod(new_shape) == np.prod(input_shape)
                    input_shape[layer_i] = new_shape
                    net = Lambda(lambda x : x.view([-1]+new_shape))
                elif kind == 'resnet':
                    assert len(layer_args) > 0
                    n_outs = int(layer_args.pop(0))
                    net = ResNet(BasicBlock, [int(k) for k in layer_args], num_classes=n_outs, **sn_kwargs)
                    input_shape[layer_i] = [n_outs]
                elif kind == 'avgpool1d':
                    kernel_size = int(layer_args.pop(0))
                    stride = int(layer_args.pop(0))
                    net = nn.AvgPool1d(kernel_size, stride=stride)
                elif kind == 'avgpool2d':
                    kernel_size = int(layer_args.pop(0))
                    stride = int(layer_args.pop(0))
                    net = nn.AvgPool2d(kernel_size, stride=stride)
                elif kind == '+':
                    net = Lambda(lambda x : torch.cat(x, dim=1))
                elif kind == 'relu':
                    net = nn.ReLU()
                elif kind == 'lrelu':
                    leak = 0.2 if not layer_args else float(layer_args.pop(0))
                    net = nn.LeakyReLU(negative_slope=leak)
                elif kind == 'tanh':
                    net = nn.Tanh()
                elif kind == 'sigmoid':
                    net = nn.Sigmoid()
                elif kind == 'fc':
                    n_outs = int(layer_args.pop(0))
                    net = nn.Linear(np.prod(input_shape[layer_i]), n_outs, **sn_kwargs)
                    input_shape[layer_i] = [n_outs]
                elif kind == 'lstm_cell':
                    assert len(input_shape[layer_i]) == 2 # seq_len, input_size
                    hidden_size = int(layer_args.pop(0))
                    bias = True if not layer_args else Parsing.str2bool(layer_args.pop(0))
                    net = nn.LSTMCell(
                            input_shape[layer_i][1], 
                            hidden_size, 
                            bias=bias,
                            )
                    input_shape[layer_i] = [hidden_size]
                elif kind == 'lstm':
                    assert len(input_shape[layer_i]) == 2 # seq_len, input_size
                    hidden_size = int(layer_args.pop(0))
                    num_layers = 1 if not layer_args else int(layer_args.pop(0))
                    bias = True if not layer_args else Parsing.str2bool(layer_args.pop(0))
                    dropout = 0. if not layer_args else float(layer_args.pop(0))
                    bidirectional = False if not layer_args else Parsing.str2bool(layer_args.pop(0))
                    net = nn.LSTM(
                            input_shape[layer_i][1], 
                            hidden_size, 
                            num_layers=num_layers,
                            bias=bias,
                            batch_first=True,
                            dropout=dropout,
                            bidirectional=bidirectional
                            )
                    input_shape[layer_i] = [input_shape[layer_i][0], (2 if bidirectional else 1)*hidden_size]
                elif kind == 'conv1d':
                    n_filts = int(layer_args.pop(0)) # out_channels
                    kernel_size = 5 if not layer_args else int(layer_args.pop(0))
                    stride = 2 if not layer_args else int(layer_args.pop(0))
                    if len(input_shape[layer_i]) != 2:
                        raise ValueError("need to reshape_2d before conv. Current shape is %s" % (input_shape[layer_i]))
                    in_channels = input_shape[layer_i][0]
                    padding = 0
                    dilation = 1

                    L_out = nnShapes.conv1d(input_shape[layer_i], padding, dilation, kernel_size, stride)
                    net = nn.Conv1d(
                            in_channels,
                            n_filts,
                            kernel_size,
                            stride=stride,
                            padding=padding,
                            dilation=dilation,
                            **sn_kwargs)
                    input_shape[layer_i] = [n_filts, L_out]
                elif kind in ('deconv2d', 'conv2d'):
                    n_filts = int(layer_args.pop(0)) # out_channels
                    kernel_size = 5 if not layer_args else int(layer_args.pop(0))
                    stride = 2 if not layer_args else int(layer_args.pop(0))
                    if len(input_shape[layer_i]) != 3:
                        raise ValueError("need to reshape_3d before conv. Current shape is %s" % (input_shape[layer_i]))
                    in_channels = input_shape[layer_i][0]
                    padding = 0
                    dilation = 1

                    if kind == 'conv2d':
                        H_out, W_out = nnShapes.conv2d(input_shape[layer_i], padding, dilation, kernel_size, stride)
                        net = nn.Conv2d(
                                in_channels,
                                n_filts,
                                kernel_size,
                                stride=stride,
                                padding=padding,
                                dilation=dilation,
                                **sn_kwargs)
                    else:
                        output_padding = 0
                        H_out, W_out = nnShapes.deconv2d(input_shape[layer_i], padding, output_padding, dilation, kernel_size, stride)
                        net = nn.ConvTranspose2d(
                                in_channels,
                                n_filts,
                                kernel_size,
                                stride=stride,
                                padding=padding,
                                output_padding=output_padding,
                                dilation=dilation,
                                **sn_kwargs)

                    input_shape[layer_i] = [n_filts, H_out, W_out]
                elif kind == 'bn1d':
                    assert len(input_shape[layer_i]) <= 2
                    if do_batch_norm:
                        net = nn.BatchNorm1d(input_shape[layer_i][0])
                elif kind == 'bn1d*':
                    assert len(input_shape[layer_i]) <= 2
                    net = nn.BatchNorm1d(input_shape[layer_i][0])
                elif kind == 'bn2d':
                    assert len(input_shape[layer_i]) == 3
                    if do_batch_norm:
                        net = nn.BatchNorm2d(input_shape[layer_i][0])
                elif kind == 'bn2d*':
                    assert len(input_shape[layer_i]) == 3
                    net = nn.BatchNorm2d(input_shape[layer_i][0])
                elif kind == 'flatten':
                    input_shape[layer_i] = [np.prod(input_shape[layer_i])]
                    net = Lambda(lambda x, y : x.view([-1, y]), input_shape[layer_i][0])
                elif kind == 'mul':
                    factor = float(layer_args.pop(0))
                    net = Lambda(lambda x : x * factor)
                elif kind == 'add':
                    amt = float(layer_args.pop(0))
                    net = Lambda(lambda x : x + amt)
                elif kind == 'scale_pm1_to_01':
                    net = Lambda(lambda x : x / 2 + 0.5)
                elif kind == 'scale_01_to_pm1':
                    net = Lambda(lambda x : x * 2 - 1)
                elif kind == 'softmax':
                    dim = -1 if not layer_args else int(layer_args.pop(0))
                    assert dim < len(input_shape[layer_i])
                    assert dim >= -1
                    net = torch.nn.Softmax(dim=dim)
                elif kind == 'block_weight':  # eg block_weight-10_2.5-2_1-126_0
                    mul_vec = []
                    start = 0
                    while layer_args:
                        length, scale = layer_args.pop(0).split('_')
                        length = int(length)
                        assert length > 0
                        scale = float(scale)
                        if scale != 0:
                            mul_vec += [torch.ones(length, dtype=torch.float32)*scale]
                        else:
                            mul_vec += [torch.ones(length, dtype=torch.float32)]
                    mul_vec = torch.cat(mul_vec)
                    mul_vec = nn.Parameter(mul_vec)
                    params += [mul_vec]
                    net = Lambda(lambda x, mul_vec : x * mul_vec[None, :, (None,)*(len(x.shape)-2)], mul_vec)
                else:
                    raise ValueError("unknown op '{}'".format(kind))
                assert not layer_args, kind
                if single_module:
                    return net
                next_layers += [net]
            layers += [next_layers]
            print(parallel_layers, input_shape)
        assert type(input_shape) is list
        assert len(input_shape) == 1

        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.params = nn.ParameterList(params)
                self._layers_ = nn.ModuleList([item for sublist in layers for item in sublist])
                self.layers = layers
                self.layer_names = spec.split(',')

            def forward(self, x):
                if type(x) is not list:
                    x = [x]

                for i, l in enumerate(layers):
                    next_x = []
                    for j, p in enumerate(l):
                        if isinstance(l, nn.LSTM):
                            out, _ = p(x[j])
                            next_x += [out]
                        else:
                            next_x += [p(x[j])]
                    x = next_x
                assert type(x) is list and len(x) == 1
                return x[0]

        model = Model()
        print('Num model params', sum((p.numel() for p in model.parameters())))
        return model, input_shape[0]

    @staticmethod
    def make_parallel(
        spec,
        input_shape,
        batch_size,
        do_batch_norm=True,
        verbose=False,
        **sn_kwargs
    ):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torchvision.models.resnet import ResNet, BasicBlock

        assert type(input_shape) == list
        if type(input_shape[0]) != list:
            input_shape = [input_shape]
        
        params = []
        layers = []

        class Lambda(nn.Module):
            def __init__(self, fn, *args):
                super(Lambda, self).__init__()
                self.fn = fn
                self.args = args
            def forward(self, x):
                return self.fn(x, *self.args)

        def batch_norm_fun(permutation):
            def fun(x, batch_norm_layer):
                x = x.permute(permutation)
                x = batch_norm_layer(x)
                x = x.permute(permutation)
                return x

        if verbose:
            print('input_shape', input_shape)
        for i, parallel_layers in enumerate(spec):
            assert len(parallel_layers) == 1
            parts = parallel_layers[0]

            kind = parts[0]
            layer_args = parts[1:]

            if kind == 'input':
                assert len(layer_args) > 0
                n_ins = int(layer_args.pop(0))
                inp = (torch.rand(1, batch_size, n_ins)-0.5)*2
                inp = nn.Parameter(inp)
                params += [inp]
                net = Lambda((lambda x, y : y), inp)
                input_shape = [n_ins]
            elif kind == 'avgpool1d':
                kernel_size = int(layer_args.pop(0))
                stride = int(layer_args.pop(0))
                net = nn.AvgPool2d(kernel_size=(1, kernel_size), stride=(1, stride))
            elif kind == 'avgpool2d':
                kernel_size = int(layer_args.pop(0))
                stride = int(layer_args.pop(0))
                net = nn.AvgPool3d(kernel_size=(1, kernel_size, kernel_size), stride=(1, stride, stride))
            elif kind == 'relu':
                net = nn.ReLU()
            elif kind == 'lrelu':
                leak = 0.2 if not layer_args else float(layer_args.pop(0))
                net = nn.LeakyReLU(negative_slope=leak)
            elif kind == 'tanh':
                net = nn.Tanh()
            elif kind == 'sigmoid':
                net = nn.Sigmoid()
            elif kind == 'fc':
                n_outs = int(layer_args.pop(0))
                net = nn.Linear(np.prod(input_shape), n_outs, **sn_kwargs)
                input_shape = [n_outs]
            elif kind == 'conv1d':
                n_filts = int(layer_args.pop(0)) # out_channels
                kernel_size = 5 if not layer_args else int(layer_args.pop(0))
                stride = 2 if not layer_args else int(layer_args.pop(0))
                if len(input_shape) != 2:
                    raise ValueError("need to reshape_2d before conv. Current shape is %s" % (input_shape))
                in_channels = input_shape[0]
                padding = 0
                dilation = 1

                L_out = nnShapes.conv1d(input_shape, padding, dilation, kernel_size, stride)
                net = nn.Conv2d(
                        in_channels,
                        n_filts,
                        kernel_size=(1, kernel_size),
                        stride=(1, stride),
                        padding=padding,
                        dilation=dilation,
                        **sn_kwargs)
                input_shape = [n_filts, L_out]
            elif kind in ('deconv2d', 'conv2d'):
                n_filts = int(layer_args.pop(0)) # out_channels
                kernel_size = 5 if not layer_args else int(layer_args.pop(0))
                stride = 2 if not layer_args else int(layer_args.pop(0))
                if len(input_shape) != 3:
                    raise ValueError("need to reshape_3d before conv. Current shape is %s" % (input_shape))
                in_channels = input_shape[0]
                padding = 0
                dilation = 1

                if kind == 'conv2d':
                    H_out, W_out = nnShapes.conv2d(input_shape, padding, dilation, kernel_size, stride)
                    net = nn.Conv3d(
                            in_channels,
                            n_filts,
                            kernel_size=(1, kernel_size, kernel_size),
                            stride=(1, stride, stride),
                            padding=padding,
                            dilation=dilation,
                            **sn_kwargs)
                else:
                    output_padding = 0
                    H_out, W_out = nnShapes.deconv2d(input_shape, padding, output_padding, dilation, kernel_size, stride)
                    net = nn.ConvTranspose3d(
                            in_channels,
                            n_filts,
                            kernel_size=(1, kernel_size, kernel_size),
                            stride=(1, stride, stride),
                            padding=padding,
                            output_padding=output_padding,
                            dilation=dilation,
                            **sn_kwargs)

                input_shape = [n_filts, H_out, W_out]
            elif kind == 'bn1d':
                if do_batch_norm:
                    if len(input_shape) == 1:
                        net = Lambda(batch_norm_fun([2, 1]), nn.BatchNorm1d(input_shape[0]))
                    else:
                        assert len(input_shape) == 2
                        net = Lambda(batch_norm_fun([2, 1]), nn.BatchNorm2d(input_shape[0]))
            elif kind == 'bn1d*':
                if len(input_shape) == 1:
                    net = Lambda(batch_norm_fun([2, 1]), nn.BatchNorm1d(input_shape[0]))
                else:
                    assert len(input_shape) == 2
                    net = Lambda(batch_norm_fun([2, 1]), nn.BatchNorm2d(input_shape[0]))
            elif kind == 'bn2d':
                assert len(input_shape) == 3
                if do_batch_norm:
                    net = Lambda(batch_norm_fun([2, 1]), nn.BatchNorm3d(input_shape[0]))
            elif kind == 'bn2d*':
                assert len(input_shape) == 3
                net = Lambda(batch_norm_fun([2, 1]), nn.BatchNorm3d(input_shape[0]))
            elif kind == 'reshape':
                new_shape = [int(k) for k in layer_args] # new shape
                layer_args = []
                if -1 not in new_shape:
                    assert np.prod(new_shape) == np.prod(input_shape), '%s, %s' % (new_shape, input_shape)
                else:
                    assert new_shape.count(-1) == 1
                    assert -np.prod(new_shape) <= np.prod(input_shape)
                input_shape = new_shape
                new_shape = [1, batch_size] + new_shape
                net = Lambda(lambda x : x.view(new_shape))
            elif kind == 'flatten':
                input_shape = [np.prod(input_shape)]
                net = Lambda(lambda x, y : x.view([-1, batch_size, y]), input_shape[0])
            elif kind == 'mul':
                factor = float(layer_args.pop(0))
                net = Lambda(lambda x : x * factor)
            elif kind == 'add':
                amt = float(layer_args.pop(0))
                net = Lambda(lambda x : x + amt)
            elif kind == 'scale_pm1_to_01':
                net = Lambda(lambda x : x / 2 + 0.5)
            elif kind == 'scale_01_to_pm1':
                net = Lambda(lambda x : x * 2 - 1)
            elif kind == 'softmax':
                net = torch.nn.Softmax(dim=-1)
            else:
                raise ValueError("unknown op '{}'".format(kind))
            layers += [net]
            if verbose:
                print(input_shape)
        assert type(input_shape) is list

        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.params = nn.ParameterList(params)
                self._layers_ = nn.ModuleList([item for item in layers])
                self.layers = layers
                self.layer_names = [layer[0] for layer in spec]

            def forward(self, x):
                for i, l in enumerate(layers):
                    x = l(x)
                return x

        model = Model()
        if verbose:
            print('Num model params', sum((p.numel() for p in model.parameters())))
        return model, input_shape
