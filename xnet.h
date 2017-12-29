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
    void ToProto(NodeProto *proto) const {
        proto->set_node_type(type_);
        ToProtoFunc(proto);
    }
    virtual void Forward(const Matrix<float> &in, Matrix<float> *out) = 0;
    virtual void Info() const {
        std::cout << NodeTypeToString(type_) << "\n";
    }
    static std::string NodeTypeToString(NodeProto_NodeType type);
    virtual Node* Copy() const = 0;
    virtual Node* Quantize() const {
        return this->Copy();
    }
    NodeProto_NodeType Type() const { return type_; };
protected:
    virtual void FromProtoFunc(const NodeProto &proto) {}
    virtual void ToProtoFunc(NodeProto *proto) const {}
    NodeProto_NodeType type_;
};

class ReLU: public Node {
public:
    ReLU(): Node(NodeProto::RELU) {}
    Node * Copy() const { return new ReLU(*this); }
    void Forward(const Matrix<float> &in, Matrix<float> *out);
};

class Sigmoid : public Node {
public:
    Sigmoid(): Node(NodeProto::SIGMOID) {}
    Node * Copy() const { return new Sigmoid(*this); }
    void Forward(const Matrix<float> &in, Matrix<float> *out);
};

class Tanh : public Node {
public:
    Tanh(): Node(NodeProto::TANH) {}
    Node * Copy() const { return new Tanh(*this); }
    void Forward(const Matrix<float> &in, Matrix<float> *out);
};

class Softmax: public Node {
public:
    Softmax(): Node(NodeProto::SOFTMAX) {}
    Node * Copy() const { return new Softmax(*this); }
    void Forward(const Matrix<float> &in, Matrix<float> *out);
};

class FullyConnect: public Node {
public:
    FullyConnect(): Node(NodeProto::FULLY_CONNECT), has_bias_(false) {}
    Node * Copy() const { return new FullyConnect(*this); }
    virtual void FromProtoFunc(const NodeProto &proto);
    void ToProtoFunc(NodeProto *proto) const; 
    virtual Node* Quantize() const; 
    void Forward(const Matrix<float> &in, Matrix<float> *out);
private:
    Matrix<float> weight_;
    Vector<float> bias_;
    bool has_bias_;
};

class QuantizeFullyConnect: public Node {
public:
    QuantizeFullyConnect(): Node(NodeProto::QUANTIZE_FULLY_CONNECT) {}
    Node * Copy() const { return new QuantizeFullyConnect(*this); }
    virtual void FromProtoFunc(const NodeProto &proto);
    void ToProtoFunc(NodeProto *proto) const; 
    void SetWeight(const Matrix<uint8_t> &weight) { weight_.CopyFrom(weight); }
    void SetBias(const Vector<float> &bias) { bias_.CopyFrom(bias_); }
    void SetWeightScale(float scale) { w_scale_ = scale; };
    void SetWeightZeroPoint(uint8_t zero_point) { w_zero_point_ = zero_point; }
    void SetHasBias(bool has_bias) { has_bias_ = has_bias; }
    void Forward(const Matrix<float> &in, Matrix<float> *out);
private:
    Matrix<uint8_t> weight_;
    Vector<float> bias_;
    float w_scale_;
    uint8_t w_zero_point_;
    bool has_bias_;
    Matrix<int32_t> quantize_out_;
    Matrix<uint8_t> quantize_in_;
};


// Current only support layer by layer structure
// Will add graph support if it is requried

class XNet {
public:
    XNet() {}
    XNet(std::string proto_file) {
        FromProto(proto_file);
    }
    ~XNet() {
        ClearNodes();
    }
    void FromProto(std::string proto_file);
    void ToProto(std::string proto_file) const;
    void Info(); 
    void Quantize(XNet *net) const;
    void ClearNodes();
    void AddNode(Node *node) {
        nodes_.push_back(node); 
    }
    void Forward(const Matrix<float> &in, Matrix<float> *out); 
private:
    std::vector<Node *> nodes_;
    std::vector<Matrix<float> *> forward_buf_;
};

#endif

