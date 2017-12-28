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
public:
    static TensorProto_DataType Type() { return TensorProto::UNDEFINED; }
};

#define PARSE_TYPE(type, data_type) \
template <> \
TensorProto_DataType ParseType<type>::Type() { return TensorProto::data_type; }

template <class DType, int32_t Dim>
class Tensor {
public:
    Tensor(DType *data=nullptr): data_(data), shape_(Dim, 0), holder_(false) {}
    ~Tensor() {
        if (holder_ && data_ != nullptr) delete [] data_;
    }
    virtual void FromProto(const TensorProto &proto); 
    virtual void ToProto(TensorProto *proto) const;
    void Resize(const std::vector<int32_t> &shape);
    int32_t Size() const {
        GetShapeSize(shape_);
    }
    DType *Data() const { return data_; } 
    std::vector<int32_t> Shape() const { return shape_; }
    void CopyFrom(const Tensor<DType, Dim> &tensor); 
protected:
    int32_t GetShapeSize(const std::vector<int32_t> &shape) const;
protected:
    DType *data_;
    std::vector<int32_t> shape_;
    bool holder_;
};

template <typename DType>
class Vector;

template <class DType>
class Matrix : public Tensor<DType, 2> {
public:
    Matrix(int32_t row = 0, int32_t col = 0) {
        Resize(row, col);
    }
    Matrix(DType *data, int32_t row, int32_t col): Tensor<DType, 2>(data) {
        CHECK(this->shape_.size() == 2);
        this->shape_[0] = row;
        this->shape_[1] = col;
    }
    void Resize(int32_t row,int32_t col) {
        std::vector<int32_t> shape = { row, col };
        Tensor<DType, 2>::Resize(shape);
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
    Vector<DType> Row(int row) const;
    Matrix<DType> RowRange(int start, int length) const;

    void Mul(const Matrix<DType> &mat1, const Matrix<DType> &mat2, 
             bool transpose = false, float alpha = 0.0);
    void Transpose(const Matrix<DType> &mat);
    void AddVec(const Vector<DType> &vec);
};

template <class DType>
class Vector: public Tensor<DType, 1> {
public:
    Vector(int32_t dim = 0) {
        Resize(dim);
    }
    Vector(DType *data, int dim): Tensor<DType, 1>(data) {
        CHECK(this->shape_.size() == 1);
        this->shape_[0] = dim;
    }
    void Resize(int32_t dim) {
        std::vector<int32_t> shape = { dim };
        Tensor<DType, 1>::Resize(shape);
    }
    const DType operator () (int n) const {
        CHECK(n < this->shape_[0]);
        return *(this->data_ + n);
    }
    DType& operator () (int n) {
        CHECK(n < this->shape_[0]);
        return *(this->data_ + n);
    }

    void Add(const Vector<DType> &vec, float alpha = 1.0);
    void Scale(float alpha);
};

// Quantization Functions
void FindMinMax(float *data, int n, float *min, float *max);
void ChooseQuantizationParams(float min, float max, float *scale, 
                              uint8_t *zero_point);
void QuantizeData(float *src, int n, float scale, uint8_t zero_point, 
                  uint8_t *dest);

#endif

