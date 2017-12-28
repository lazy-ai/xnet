/* Created on 2017-12-03
 * Author: Binbin Zhang
 */

#ifndef XNET_H_
#define XNET_H_

#include <string>

#include "utils.h"
#include "net.pb.h"
#include "tensor.h"


class Node {
public:
    Node(NodeProto_NodeType type=NodeProto::UNKNOWN): type_(type) {}
    void FromProto(const NodeProto &proto) {
        CHECK(type_ == proto.node_type());
        FromProtoFunc(proto);
    }
    virtual void FromProtoFunc(const NodeProto &proto) {}
    //void Forward(const Matrix<float> &in, Matrix<float> *out);
    //int32_t InDim() const { return in_dim_; }
    //int32_t OutDim() const { return out_dim_; }
    //virtual NodeType Type() const { return proto; };
    virtual void Info() const {
        std::cout << NodeTypeToString(type_) << "\n";
    }
    static std::string NodeTypeToString(NodeProto_NodeType type);
protected:
    //virtual void ForwardFunc(const Matrix<float> &in, Matrix<float> *out) = 0;
    int32_t in_dim_,out_dim_;
    NodeProto_NodeType type_;
};

class ReLU: public Node {
public:
    ReLU(): Node(NodeProto::RELU) {}
};

class Sigmoid : public Node {
public:
    Sigmoid(): Node(NodeProto::SIGMOID) {}
};

class Tanh : public Node {
public:
    Tanh(): Node(NodeProto::TANH) {}
};

class Softmax: public Node {
public:
    Softmax(): Node(NodeProto::SOFTMAX) {}
};

class FullyConnect: public Node {
public:
    FullyConnect(): Node(NodeProto::FULLY_CONNECT), has_bias_(false) {}
    virtual void FromProtoFunc(const NodeProto &proto);
private:
    Matrix<float> weight_;
    Vector<float> bias_;
    bool has_bias_;
};

class QuantizeFullyConnect: public Node {
public:
    QuantizeFullyConnect(): Node(NodeProto::QUANTIZE_FULLY_CONNECT) {}
    virtual void FromProtoFunc(const NodeProto &proto);
private:
    Matrix<uint8_t> weight_;
    Vector<float> bias_;
    float w_scale_;
    uint8_t w_zero_point_;
    bool has_bias_;
};


// Current only support layer by layer structure
// Will add graph support if it is requried

class XNet {
public:
    void Read(std::string model_file);
    ~XNet() {
        ClearNodes();
    }
    void Info(); 
private:
    void ClearNodes();

    std::vector<Node *> nodes_;
};

#endif

