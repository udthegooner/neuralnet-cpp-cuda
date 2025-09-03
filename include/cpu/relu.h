#ifndef RELU_H
#define RELU_H

#include "base.h"

class ReLU: public Base {
    public:
        ReLU(int numVals);
        float* forward(float *input, int numData);
        void backward(int numData);
};

#endif