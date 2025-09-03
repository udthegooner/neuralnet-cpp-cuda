#ifndef SOFTMAX_H
#define SOFTMAX_H
#include "base.h"

class Softmax: public Base {
    public:
        Softmax(int _numVals);
        float* forward(float *input, int numData);
        void backward(int numData);
};

#endif