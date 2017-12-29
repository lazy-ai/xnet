// Created on 2017-06-21
// Author: Binbin Zhang
#include <fstream>

#include "xnet.h"
#include "../tools/parse-option.h"

int BigLittleSwap(int a) {
    return ((((uint32_t)(a) & 0xff000000) >> 24) | \
            (((uint32_t)(a) & 0x00ff0000) >> 8) | \
            (((uint32_t)(a) & 0x0000ff00) << 8) | \
            (((uint32_t)(a) & 0x000000ff) << 24));
}

void ReadMnistLabel(std::string filename, std::vector<int> *label) {
    std::ifstream is(filename, std::ios::binary);
    if (is.fail()) {
       ERROR("read file %s error, check!!!", filename.c_str()); 
    }
    int magic = 0, num_images = 0;
    is.read((char *)&magic, 4);
    is.read((char *)&num_images, 4);
    magic = BigLittleSwap(magic);
    num_images = BigLittleSwap(num_images);
    std::cout << magic << " " << num_images << "\n";
    label->resize(num_images);
    unsigned char digit = 0;
    for (int i = 0; i < num_images; i++) {
        is.read((char *)&digit, 1);
        (*label)[i] = static_cast<int>(digit);
        //std::cout << (*label)[i] << "\n";
    }
}

void ReadMnistImage(std::string filename, Matrix<float> *data) {
    std::ifstream is(filename, std::ios::binary);
    if (is.fail()) {
       ERROR("read file %s error, check!!!", filename.c_str()); 
    }
    int magic = 0, num_images = 0;
    is.read((char *)&magic, 4);
    is.read((char *)&num_images, 4);
    magic = BigLittleSwap(magic);
    num_images = BigLittleSwap(num_images);
    std::cout << magic << " " << num_images << "\n";
    int rows = 0, cols = 0;
    is.read((char *)&rows, 4);
    is.read((char *)&cols, 4);
    rows = BigLittleSwap(rows);
    cols = BigLittleSwap(cols);
    std::cout << rows << " " << cols << "\n";
    data->Resize(num_images, rows * cols);
    unsigned char digit = 0;
    for (int i = 0; i < num_images; i++) {
        for (int j = 0; j < rows * cols; j++) {
            is.read((char *)&digit, 1);
            (*data)(i, j) = static_cast<float>(digit) / 255;
            //std::cout << static_cast<int>(digit) << " ";
        }
        //std::cout << "\n";
    }
}

int main(int argc, char *argv[]) {
    const char *usage = "Simple test on mnist data\n";
    ParseOptions option(usage);
    int batch = 32;
    option.Register("batch", &batch, "batch size for net forward");
    option.Read(argc, argv);

    if (option.NumArgs() != 3) {
        option.PrintUsage();
        exit(1);
    }

    std::string net_file = option.GetArg(1),
                image_file = option.GetArg(2),
                label_file = option.GetArg(3);
    
    XNet net(net_file);
    net.Info();
    std::vector<int> label;
    Matrix<float> data;
    ReadMnistLabel(label_file, &label);
    ReadMnistImage(image_file, &data);
    assert(label.size() == data.NumRows());
    int num_images = label.size(), num_correct = 0;
    for (int i = 0; i < num_images; i += batch) {
        int real_batch = i + batch < num_images ? batch : num_images - i; 
        Matrix<float> in(real_batch, data.NumCols()), out;
        // copy
        for (int m = 0; m < in.NumRows(); m++) {
            for (int n = 0; n < in.NumCols(); n++) {
                in(m, n) = data(i+m, n);
            }
        }

        net.Forward(in, &out);

        for (int m = 0; m < out.NumRows(); m++) {
            float max = out(m, 0);
            int max_idx = 0;
            for (int n = 1; n < out.NumCols(); n++) {
                if (out(m, n) > max) {
                    max = out(m, n);
                    max_idx = n;
                }
            }
            if (max_idx == label[i+m]) num_correct++;
        }
    }
    printf("Accuracy %.6lf\n", static_cast<double>(num_correct) / num_images);
    return 0;
}


