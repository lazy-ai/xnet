/* Created on 2017-12-03
 * Author: Binbin Zhang
 */

#include <fstream>

#include "xnet.h"

std::string Node::NodeTypeToString(NodeProto_NodeType type) {
    switch(type) {
        case NodeProto::FULLY_CONNECT: return "<FullyConnect>";
        case NodeProto::QUANTIZE_FULLY_CONNECT: return "<QuantizeFullyConnect>";
        case NodeProto::RELU: return "<ReLU>";
        case NodeProto::SIGMOID: return "<Sigmoid>";
        case NodeProto::TANH: return "<Tanh>";
        case NodeProto::SOFTMAX: return "<Softmax>";
        default: return "<Unknown>";
    }
}

void FullyConnect::FromProtoFunc(const NodeProto &proto) {
    CHECK(proto.has_fully_connect_param());
    const FullyConnectParameter &param = proto.fully_connect_param();
    has_bias_ = false;
    weight_.FromProto(param.weight());
    if (param.has_bias()) { 
        bias_.FromProto(param.bias());
        has_bias_ = true;
    }
}

void QuantizeFullyConnect::FromProtoFunc(const NodeProto &proto) {
    CHECK(proto.has_quantize_fully_connect_param());
    const QuantizeFullyConnectParameter &param = proto.quantize_fully_connect_param();
    has_bias_ = false;
    weight_.FromProto(param.weight());
    if (param.has_bias()) { 
        bias_.FromProto(param.bias());
        has_bias_ = true;
    }
    w_scale_ = param.w_scale();
    w_zero_point_ = param.w_zero_point();
}

void XNet::Info() {
    for (int i = 0; i < nodes_.size(); i++) 
        nodes_[i]->Info();
}

void XNet::ClearNodes() {
    for (int i = 0; i < nodes_.size(); i++) 
        delete nodes_[i];
}

void XNet::Read(std::string model_file) {
    NetProto net_proto;
    std::fstream in(model_file, std::ios::in | std::ios::binary);
    if (!in) {
        ERROR("file %s does not exist", model_file.c_str());
    } else {
        net_proto.ParseFromIstream(&in);
    }

    this->ClearNodes();
    for (int i = 0; i < net_proto.nodes_size(); i++) {
        const NodeProto &node_proto = net_proto.nodes(i);
        //std::cout << node_proto.name() << " " << node_proto.node_type() << std::endl;
        Node *node = nullptr;
        switch (node_proto.node_type()) {
            case NodeProto::FULLY_CONNECT:
                node = new FullyConnect();
                break;
            case NodeProto::QUANTIZE_FULLY_CONNECT:
                node = new QuantizeFullyConnect();
                break;
            case NodeProto::RELU:
                node = new ReLU();
                break;
            case NodeProto::SIGMOID:
                node = new Sigmoid();
                break;
            case NodeProto::TANH:
                node = new Tanh();
                break;
            case NodeProto::SOFTMAX:
                node = new Softmax();
                break;
            default:
                ERROR("unknown node type %d", node_proto.node_type());
        }
        node->FromProto(node_proto);
        nodes_.push_back(node);
    }
}

