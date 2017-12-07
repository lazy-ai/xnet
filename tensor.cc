/* Created on 2017-12-05
 * Author: Binbin Zhang
 */



#include "tensor.h"


template <class DType>
void Tensor<DType>::Resize(std::vector<int> shape) {
    if (data_ != NULL) delete [] data_;
    shape_ = shape;
    int size = this->Size();
    if (size == 0) return;
    data_ = new DType[size]();
}

template <class DType>
void Tensor<DType>::Resize(int dim) {
    std::vector<int> shape = {dim};
    Resize(shape);
}

template <class DType>
void Tensor<DType>::Resize(int rows, int cols) {
    std::vector<int> shape = {rows, cols};
    Resize(shape);
}

template <class DType>
void Tensor<DType>::FromProto(const TensorProto &proto) {
    shape_.resize(proto.shape_size());
    for (int i = 0; i < proto.shape_size(); i++) {
        shape_[i] = proto.shape(i);
    }
    Resize(shape_);
    if (proto.data_type() == TensorProto::FLOAT) {
        for (int i = 0; i < proto.float_data_size(); i++) {
            data_[i] = proto.float_data(i);
        }
    }
    else if (proto.data_type() == TensorProto::DOUBLE) {
        for (int i = 0; i < proto.double_data_size(); i++) {
            data_[i] = proto.double_data(i);
        }
    }
    else if (proto.data_type() == TensorProto::INT32 || 
             proto.data_type() == TensorProto::INT16 ||
             proto.data_type() == TensorProto::INT8) {
         for (int i = 0; i < proto.int32_data_size(); i++) {
            data_[i] = static_cast<DType>(proto.int32_data(i));
         }
    }
    else {
        ERROR("Not implement");
    }
}

template <>
void Tensor<float>::ToProto(TensorProto *proto) const {
    proto->clear_shape();
    for (int i = 0; i < shape_.size(); i++) {
        proto->add_shape(shape_[i]);
    }
    proto->set_data_type(TensorProto::FLOAT);
    proto->clear_float_data();
    for (int i = 0; i < Size(); i++) {
        proto->add_float_data(data_[i]);   
    }
}

template <>
void Tensor<int>::ToProto(TensorProto *proto) const {
    proto->clear_shape();
    for (int i = 0; i < shape_.size(); i++) {
        proto->add_shape(shape_[i]);
    }
    proto->set_data_type(TensorProto::INT32);
    proto->clear_int32_data();
    for (int i = 0; i < Size(); i++) {
        proto->add_int32_data(data_[i]);   
    }
}

template <>
void Tensor<uint8_t>::ToProto(TensorProto *proto) const {
    proto->clear_shape();
    for (int i = 0; i < shape_.size(); i++) {
        proto->add_shape(shape_[i]);
    }
    proto->set_data_type(TensorProto::INT8);
    proto->clear_int32_data();
    for (int i = 0; i < Size(); i++) {
        proto->add_int32_data(data_[i]);   
    }
}

template class Tensor<float>;
template class Tensor<int>;
template class Tensor<uint8_t>;

