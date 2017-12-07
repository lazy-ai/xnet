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
class Tensor {
public:
    Tensor(): data_(NULL) {}
    ~Tensor() {
        if (data_ != NULL) delete [] data_;
    }
    virtual void FromProto(const TensorProto &proto); 
    virtual void ToProto(TensorProto *proto) const;
    void Resize(std::vector<int> shape);
    void Resize(int dim);
    void Resize(int rows, int cols);
    int Size() const {
        if (shape_.size() == 0) return 0;
        int size = 1;
        for (int i = 0; i < shape_.size(); i++) size *= shape_[i];
        return size;
    }
private:
private:
    DType *data_;
    std::vector<int> shape_;
};


#endif

