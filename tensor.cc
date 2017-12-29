/* Created on 2017-12-05
 * Author: Binbin Zhang
 */
#include "tensor.h"

#ifdef USE_BLAS
#include <cblas.h>
#endif

PARSE_TYPE(float, FLOAT)
PARSE_TYPE(int32_t, INT32)
PARSE_TYPE(uint8_t, INT8)

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
int32_t Tensor<DType, Dim>::GetShapeSize(
        const std::vector<int32_t> &shape) const {
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

template <class DType, int32_t Dim>
void Tensor<DType, Dim>::CopyFrom(const Tensor<DType, Dim> &tensor) {
    Resize(tensor.Shape());
    memcpy(data_, tensor.Data(), Size() * sizeof(DType));
}

template <typename DType>
Matrix<DType> Matrix<DType>::RowRange(int start, int length) const {
    return Matrix<DType>(this->data_ + start * NumCols(), length, NumCols());
}

template <typename DType>
Vector<DType> Matrix<DType>::Row(int row) const {
    return Vector<DType>(this->data_ + row * NumCols(), NumCols());
}

template <typename DType>
void Matrix<DType>::Mul(const Matrix<DType> &mat1, const Matrix<DType> &mat2, 
        bool transpose, float alpha) {
    if (!transpose) {
        CHECK(mat1.NumCols() == mat2.NumRows());
        CHECK(NumRows() == mat1.NumRows());
        CHECK(NumCols() == mat2.NumCols());
        //this->Resize(mat1.NumRows(), mat2.NumCols());
        for (int i = 0; i < mat1.NumRows(); i++) {
            for (int j = 0; j < mat2.NumCols(); j++) {
                (*this)(i, j) *= alpha; 
                for (int k = 0; k < mat1.NumCols(); k++) {
                    (*this)(i, j) += mat1(i, k) * mat2(k, j); 
                }
            }
        }
    }
    else {
        CHECK(mat1.NumCols() == mat2.NumCols());
        CHECK(NumRows() == mat1.NumRows());
        CHECK(NumCols() == mat2.NumRows());
        this->Resize(mat1.NumRows(), mat2.NumRows());
        for (int i = 0; i < mat1.NumRows(); i++) {
            for (int j = 0; j < mat2.NumRows(); j++) {
                (*this)(i, j) *= alpha; 
                for (int k = 0; k < mat1.NumCols(); k++) {
                    (*this)(i, j) += mat1(i, k) * mat2(j, k); 
                }
            }
        }
    }
}

// cblas_sger
template<typename DType>
void Matrix<DType>::AddVec(const Vector<DType> &vec) {
    CHECK(NumCols() == vec.Size());
    for (int i = 0; i < NumRows(); i++) {
        for (int j = 0; j < NumCols(); j++) {
            (*this)(i, j) += vec(j); 
        }
    }
}

#ifdef USE_BLAS
template <>
void Matrix<float>::Mul(const Matrix<float> &mat1, const Matrix<float> &mat2, 
        bool transpose, float alpha) {
    CHECK((!transpose && mat1.NumCols() == mat2.NumRows() && 
            NumRows() == mat1.NumRows() && NumCols() == mat2.NumCols()) ||
            (transpose && mat1.NumCols() == mat2.NumCols() && 
            NumRows() == mat1.NumRows() && NumCols() == mat2.NumRows()));
    cblas_sgemm(CblasRowMajor, CblasNoTrans, !transpose ? CblasNoTrans : CblasTrans,
                NumRows(), NumCols(), mat1.NumCols(), 1.0, 
                mat1.Data(), mat1.NumCols(), mat2.Data(), mat2.NumCols(),
                alpha, data_, NumCols());
}
#endif

template <typename DType>
void Matrix<DType>::Transpose(const Matrix<DType> &mat) {
    this->Resize(mat.NumCols(), mat.NumRows());
    for (int i = 0; i < mat.NumRows(); i++) {
        for (int j = 0; j < mat.NumCols(); j++) {
            (*this)(j, i) = mat(i, j);
        }
    }
}


template <typename DType>
void Vector<DType>::Add(const Vector<DType> &vec, float alpha) {
    for (int i = 0; i < this->Size(); i++) {
        (*this)(i) += alpha * vec(i);
    }
}

template <typename DType>
void Vector<DType>::Scale(float alpha) {
    for (int i = 0; i < this->Size(); i++) {
        (*this)(i) *= alpha;
    }
}

static void FindMinMax(float *data, int n, float *min, float *max) {
    *min = *max = data[0];
    for (int i = 1; i < n; i++) {
        if (data[i] > *max) *max = data[i];
        if (data[i] < *min) *min = data[i];
    }
}

static void ChooseQuantizationParams(float min, float max, 
        float *scale, uint8_t *zero_point) {
    min = std::min(min, 0.f);
    max = std::max(max, 0.f);
    // the min and max quantized values, as floating-point values
    const float qmin = 0;
    const float qmax = 255;
    // First determine the scale.
    const double scale_double = (max - min) / (qmax - qmin);
    const double initial_zero_point = qmin - min / scale_double;
    std::uint8_t nudged_zero_point = 0;
    if (initial_zero_point < qmin) {
        nudged_zero_point = qmin;
    } else if (initial_zero_point > qmax) {
        nudged_zero_point = qmax;
    } else {
        nudged_zero_point =
            static_cast<std::uint8_t>(round(initial_zero_point));
    }
    *zero_point = nudged_zero_point;
    *scale = scale_double;
}

void QuantizeData(float *src, int n, float *scale, 
        uint8_t *zero_point, uint8_t *dest) {
    float min, max;
    FindMinMax(src, n, &min, &max);
    ChooseQuantizationParams(min, max, scale, zero_point);
    for (int i = 0; i < n; i++) {
        float point = (*zero_point) + src[i] / (*scale);  
        float round_point = std::max(0.f, std::min(255.f, point));
        dest[i] = static_cast<uint8_t>(round(round_point));
    }
}

void DequantizeData(int32_t *src, int n, float scale,
        uint8_t zero_point, float *dest) {
    for (int i = 0; i < n; i++) {
        dest[i] = scale * (src[i] - zero_point);
    }
}

template class Matrix<uint8_t>;
template class Matrix<int>;
template class Matrix<float>;
template class Vector<uint8_t>;
template class Vector<int>;
template class Vector<float>;

