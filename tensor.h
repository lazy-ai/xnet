/* Created on 2017-12-05
 * Author: Binbin Zhang
 */

#ifndef TENSOR_H_
#define TENSOR_H_

#include <string>
#include <vector>

#include "utils.h"
#include "net.pb.h"

template <class DType>
class ParseType {
    static TensorProto_DataType Type() { return TensorProto::UNDEFINED; }
};

#define PARSE_TYPE(type, data_type) \
template <> \
TensorProto_DataType ParseType<type>::Type() { return TensorProto::data_type; }

PARSE_TYPE(float, FLOAT)
PARSE_TYPE(int32_t, INT32)
PARSE_TYPE(uint8_t, INT8)

template <class DType, int32_t Dim>
class Tensor {
public:
    Tensor(): data_(nullptr), shape_(Dim, 0), holder_(true) {}
    ~Tensor() {
        if (data_ != nullptr) delete [] data_;
    }
    virtual void FromProto(const TensorProto &proto); 
    virtual void ToProto(TensorProto *proto) const;
    void Resize(const std::vector<int32_t> &shape);
    int32_t Size() const {
        GetShapeSize(shape_);
    }
    DType *Data() const { return data_; } 
protected:
    int32_t GetShapeSize(const std::vector<int32_t> &shape) const;
protected:
    DType *data_;
    std::vector<int32_t> shape_;
    bool holder_;
};

template <class DType>
class Matrix : public Tensor<DType, 2> {
public:
    Matrix(int32_t row = 0, int32_t col = 0) {
        Resize(row, col);
    }
    void Resize(int32_t row,int32_t col) {
        std::vector<int32_t> shape = { row, col };
        Resize(shape);
    }
    int32_t NumRows() const { return this->shape_[0]; }
    int32_t NumCols() const { return this->shape_[1]; }
    const DType operator () (int r, int c) const {
        CHECK(r < NumRows());
        CHECK(c < NumCols());
        return *(this->data_ + r * NumCols() + c);
    }
    DType& operator () (int r, int c) {
        CHECK(r < NumRows());
        CHECK(c < NumCols());
        return *(this->data_ + r * NumCols() + c);
    }
};

template <class DType>
class Vector: public Tensor<DType, 1> {
public:
    Vector(int32_t dim = 0) {
        Resize(dim);
    }
    void Resize(int32_t dim) {
        std::vector<int32_t> shape = { dim };
        Resize(shape);
    }
    const DType operator () (int n) const {
        CHECK(n < this->Size());
        return *(this->data_ + n);
    }
    DType& operator () (int n) {
        CHECK(n < this->Size());
        return *(this->data_ + n);
    }
};


#endif

