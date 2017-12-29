/* Created on 2017-12-03
 * Author: Binbin Zhang
 */

#include <math.h>

#include <fstream>
#include <algorithm>

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

void ReLU::Forward(const Matrix<float> &in, Matrix<float> *out) {
    CHECK(out != nullptr);
    out->Resize(in.NumRows(), in.NumCols());
    for (int i = 0; i < in.NumRows(); i++) {
        for (int j = 0; j < in.NumCols(); j++) {
            (*out)(i, j) = std::max(in(i, j), 0.0f);
        }
    }
}

void Sigmoid::Forward(const Matrix<float> &in, Matrix<float> *out) {
    CHECK(out != nullptr);
    out->Resize(in.NumRows(), in.NumCols());
    for (int i = 0; i < in.NumRows(); i++) {
        for (int j = 0; j < in.NumCols(); j++) {
            (*out)(i, j) = 1.0 / (1 + exp(-in(i, j)));
        }
    }
}

void Tanh::Forward(const Matrix<float> &in, Matrix<float> *out) {
    CHECK(out != nullptr);
    out->Resize(in.NumRows(), in.NumCols());
    for (int i = 0; i < in.NumRows(); i++) {
        for (int j = 0; j < in.NumCols(); j++) {
            (*out)(i, j) = tanh(in(i, j));
        }
    }
}

void Softmax::Forward(const Matrix<float> &in, Matrix<float> *out) {
    CHECK(out != nullptr);
    out->Resize(in.NumRows(), in.NumCols());
    for (int i = 0; i < in.NumRows(); i++) {
        float max = in(i, 0), sum = 0.0; 
        for (int j = 1; j < in.NumCols(); j++) {
            max = std::max(in(i, j), max);
        }
        for (int j = 0; j < in.NumCols(); j++) {
            sum += (*out)(i, j) = exp(in(i, j) - max);
        }
        for (int j = 0; j < in.NumCols(); j++) {
            (*out)(i, j) /= sum;
        }
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

void FullyConnect::ToProtoFunc(NodeProto *proto) const {
    FullyConnectParameter *param = proto->mutable_fully_connect_param();
    weight_.ToProto(param->mutable_weight());
    if (has_bias_) {
        bias_.ToProto(param->mutable_bias());
    }
}

Node* FullyConnect::Quantize() const {
    QuantizeFullyConnect *node = new QuantizeFullyConnect();
    Matrix<uint8_t> quantize_weight(weight_.NumRows(), weight_.NumCols());
    float scale = 0;
    uint8_t zero_point= 0;
    QuantizeData(weight_.Data(), weight_.Size(), &scale, &zero_point, 
                 quantize_weight.Data());
    node->SetWeight(quantize_weight);
    node->SetWeightScale(scale);
    node->SetWeightZeroPoint(zero_point);
    node->SetHasBias(has_bias_);
    if (has_bias_) {
        node->SetBias(bias_);
    }
    return node;
}

void FullyConnect::Forward(const Matrix<float> &in, Matrix<float> *out) {
    CHECK(out != nullptr);
    out->Resize(in.NumRows(), weight_.NumRows());
    out->Mul(in, weight_, true);
    if (has_bias_) {
        out->AddVec(bias_);
    }
}

void QuantizeFullyConnect::FromProtoFunc(const NodeProto &proto) {
    CHECK(proto.has_quantize_fully_connect_param());
    const QuantizeFullyConnectParameter &param = 
        proto.quantize_fully_connect_param();
    has_bias_ = false;
    weight_.FromProto(param.weight().tensor());
    w_scale_ = param.weight().scale();
    w_zero_point_ = static_cast<uint8_t>(param.weight().zero_point());
    if (param.has_bias()) { 
        bias_.FromProto(param.bias());
        has_bias_ = true;
    }
}

void QuantizeFullyConnect::ToProtoFunc(NodeProto *proto) const {
    QuantizeFullyConnectParameter *param = 
        proto->mutable_quantize_fully_connect_param();
    weight_.ToProto(param->mutable_weight()->mutable_tensor());
    param->mutable_weight()->set_scale(w_scale_);
    param->mutable_weight()->set_zero_point(w_zero_point_);
    if (has_bias_) {
        bias_.ToProto(param->mutable_bias());
    }
}

void QuantizeFullyConnect::Forward(const Matrix<float> &in, 
        Matrix<float> *out) {
    CHECK(out != nullptr);
    out->Resize(in.NumRows(), weight_.NumRows());
    // quantize in
    float in_scale;
    uint8_t in_zero_point;
    quantize_in_.Resize(in.NumRows(), in.NumCols());
    QuantizeData(in.Data(), in.NumRows() * in.NumCols(), &in_scale, 
        &in_zero_point, quantize_in_.Data());
    //// uint8 gemm
    quantize_out_.Resize(out->NumRows(), out->NumCols());
    IntegerGemm<true>(quantize_in_, weight_, static_cast<int>(in_zero_point), 
        static_cast<int>(w_zero_point_), &quantize_out_);
    //// dequantize
    float out_scale = in_scale * w_scale_;
    DequantizeData(quantize_out_.Data(), out->NumRows() * out->NumCols(), 
        out_scale, 0, out->Data());
    //// add bias
    if (has_bias_) {
        out->AddVec(bias_);
    }
}

void XNet::Info() {
    for (int i = 0; i < nodes_.size(); i++) 
        nodes_[i]->Info();
}

void XNet::ClearNodes() {
    for (int i = 0; i < nodes_.size(); i++) 
        delete nodes_[i];
}

void XNet::FromProto(std::string proto_file) {
    NetProto net_proto;
    std::fstream in(proto_file, std::ios::in | std::ios::binary);
    if (!in) {
        ERROR("file %s does not exist", proto_file.c_str());
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

void XNet::ToProto(std::string proto_file) const {
    NetProto net_proto;
    for (int i = 0; i < nodes_.size(); i++) {
        nodes_[i]->ToProto(net_proto.add_nodes());  
    }
    std::fstream output(proto_file, std::ios::out | std::ios::binary);
    if (!net_proto.SerializeToOstream(&output)) {
        ERROR("failed to write %s", proto_file.c_str());
    }
}

void XNet::Quantize(XNet *quantize_net) const {
    quantize_net->ClearNodes(); 
    for (int i = 0; i < nodes_.size(); i++) {
        quantize_net->AddNode(nodes_[i]->Quantize());
    }
}

void XNet::Forward(const Matrix<float> &in, Matrix<float> *out) {
    CHECK(out != nullptr);
    CHECK(nodes_.size() > 0);
    int num_layers = nodes_.size();
    if (forward_buf_.size() != num_layers) {
        for (int i = 0; i < num_layers - 1; i++) {
            forward_buf_.push_back(new Matrix<float>()); 
        }
    }
    if (nodes_.size() == 1) {
        nodes_[0]->Forward(in, out);
    }
    else {
        nodes_[0]->Forward(in, forward_buf_[0]);
        for (int i = 1; i < nodes_.size() - 1; i++) {
            nodes_[i]->Forward(*(forward_buf_[i-1]), forward_buf_[i]);
        }
        nodes_[num_layers-1]->Forward(*(forward_buf_[num_layers-2]), out);
    }
}

