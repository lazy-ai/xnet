/* Created on 2017-12-03
 * Author: Binbin Zhang
 */

#ifndef XNET_H_
#define XNET_H_

#include <string>

#include "utils.h"
#include "net.pb.h"


class Node {
public:
    Node(const NodeProto &proto): proto_(proto) {}
    //void Forward(const Matrix<float> &in, Matrix<float> *out);
    //int32_t InDim() const { return in_dim_; }
    //int32_t OutDim() const { return out_dim_; }
    //virtual NodeType Type() const { return proto; };
    //void Info() const {
    //    std::cout << NodeTypeToString(type_) << " in_dim " << in_dim_ 
    //              << " out_dim " << out_dim_ << "\n";
    //}
protected:
    //virtual void ForwardFunc(const Matrix<float> &in, Matrix<float> *out) = 0;
    int32_t in_dim_,out_dim_;
    const NodeProto &proto_;
};


class XNet {
public:
    void Read(std::string model_file);
};

#endif

