from __future__ import print_function

import sys
import struct
import numpy as np
import net_pb2

import keras
from keras.models import Sequential, save_model, load_model
from keras.layers import Dense, Dropout
from google.protobuf import json_format


def error_msg(msg):
    print(msg)
    sys.exit(-1)

def convert_ndarray_to_list(ndarray):
    if ndarray.dtype.name != 'float32':
       error_msg('params must in float32 format, but got %s'. arr.dtype.name) 
    arr = list()
    if ndarray.ndim == 1:
        arr = ndarray.tolist()
    elif ndarray.ndim == 2:
        for row in ndarray:
            arr.extend(row.tolist())
    else:
        error_msg("unsopported ndarry dim %d" % arr.ndim)
    return arr

def parse_activation_type(act):
    if act == 'relu': 
        return net_pb2.NodeProto.RELU
    elif act == 'sigmoid':  
        return net_pb2.NodeProto.SIGMOID
    elif act == 'tanh':  
        return net_pb2.NodeProto.TANH
    elif act == 'softmax':
        return net_pb2.NodeProto.SOFTMAX
    else:
        error_msg('activation %s is not supported' % act)

def convert_keras_model_to_net(model, xnet_model):
    layers = model.layers
    ref_count = {}
    def add_new_node(key):
        number = 0
        if key in ref_count:
            number = ref_count[key]
        else:
            ref_count[key] = 0
        ref_count[key] = ref_count[key] + 1
        return number

    for layer in layers:
        layer_name = layer.name
        class_name = layer.__class__.__name__
        in_dim, out_dim = layer.input_shape[1], layer.output_shape[1]
        xnet_node = xnet_model.nodes.add()
        print(class_name, in_dim, out_dim)
        if class_name == 'Dense':
            xnet_node.name = 'fully_connect%d' % add_new_node('fully_connect')
            xnet_node.node_type = net_pb2.NodeProto.FULLY_CONNECT
            xnet_node.fully_connect_param.weight.data_type = net_pb2.TensorProto.FLOAT
            xnet_node.fully_connect_param.weight.shape.extend([out_dim, in_dim])
            xnet_node.fully_connect_param.weight.float_data.extend(
                convert_ndarray_to_list(layer.kernel.get_value().T))
            if layer.use_bias != None:
                xnet_node.fully_connect_param.bias.data_type = net_pb2.TensorProto.FLOAT
                xnet_node.fully_connect_param.bias.shape.extend([out_dim])
                xnet_node.fully_connect_param.bias.float_data.extend(
                    convert_ndarray_to_list(layer.bias.get_value()))
            if layer.activation != None and layer.activation.__name__ != 'linear':
                xnet_node = xnet_model.nodes.add()
                act = layer.activation.__name__
                xnet_node.name = '%s%d' % (act, add_new_node(act))
                xnet_node.node_type = parse_activation_type(act)
        elif class_name == 'Activation':
            act = layer.activation.__name__
            xnet_node.name = '%s%d' % (act, add_new_node(act))
            xnet_node.node_type = parse_activation_type(act)
        else:
            error_msg('error, layer %s %s is supported' % (layer_name, class_name))

if __name__ == '__main__':
    usage = '''Usage: convert keras sequential model to xnet model
               eg: convert_keras_model.py keras_model_file out_net_file'''
    if len(sys.argv) != 3:
        error_msg(usage)
    
    model = load_model(sys.argv[1]) 
    model.summary()
    if not isinstance(model, Sequential):
        error_msg('model %s is not Sequential model' % sys.argv[1])
    
    xnet_model = net_pb2.NetProto()
    xnet_model.version = 1;
    xnet_model.doc = 'converted from keras model'
    convert_keras_model_to_net(model, xnet_model)
    with open(sys.argv[2], 'wb') as fid:
        fid.write(xnet_model.SerializeToString())
    
    json_string = json_format.MessageToJson(xnet_model)
    #print(json_string)

