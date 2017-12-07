/* Created on 2017-12-05
 * Author: Binbin Zhang
 */
#include "tensor.h"

template <class DType, int32_t Dim>
void Tensor<DType, Dim>::Resize(const std::vector<int32_t> &shape) {
    CHECK(shape_.size() == Dim);
    CHECK(shape.size() == Dim);
    int32_t size = GetShapeSize(shape);
    if (size == 0 || size == this->Size()) return;
    if (holder_ && data_ != nullptr) delete [] data_;
    shape_ = shape;
    data_ = new DType[size]();
    holder_ = true;
}

template <class DType, int32_t Dim>
int32_t Tensor<DType, Dim>::GetShapeSize(const std::vector<int32_t> &shape) const {
    if (shape.size() == 0) return 0;
    int32_t size = 1;
    for (int32_t i = 0; i < shape.size(); i++) size *= shape[i];
    return size;
}

template <class DType, int32_t Dim>
void Tensor<DType, Dim>::FromProto(const TensorProto &proto) {
    CHECK(shape_.size() == Dim);
    CHECK(proto.shape_size() == Dim);
    std::vector<int32_t> shape(Dim, 0);
    for (int i = 0; i < proto.shape_size(); i++) {
        shape[i] = proto.shape(i);
    }
    Resize(shape);
    // Check type
    CHECK(ParseType<DType>::Type() == proto.data_type());
    if (proto.data_type() == TensorProto::FLOAT) {
        for (int i = 0; i < proto.float_data_size(); i++) {
            data_[i] = proto.float_data(i);
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

template <class DType, int32_t Dim>
void Tensor<DType, Dim>::ToProto(TensorProto *proto) const {
    proto->clear_shape();
    for (int i = 0; i < shape_.size(); i++) {
        proto->add_shape(shape_[i]);
    }
    TensorProto_DataType data_type = ParseType<DType>::Type();
    proto->set_data_type(data_type);
    if (data_type == TensorProto::FLOAT) {
        proto->clear_float_data();
        for (int i = 0; i < Size(); i++) {
            proto->add_float_data(data_[i]);
        }
    }
    else if (data_type == TensorProto::INT32 || 
             data_type == TensorProto::INT16 ||
             data_type == TensorProto::INT8) {
        proto->clear_int32_data();
        for (int i = 0; i < Size(); i++) {
            proto->add_int32_data(static_cast<int32_t>(data_[i]));
        }
    }
    else {
        ERROR("Not implement");
    }
}


