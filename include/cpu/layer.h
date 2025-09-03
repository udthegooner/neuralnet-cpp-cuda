#ifndef LAYER_H
#define LAYER_H

#include "base.h"

class Layer: public Base{
    public:
        float *weights, *bias, *oldWeights, lr, *newWeights;
        int numIn, weightSize;

        Layer(int nIn, int nOut, float _lr);

        float* forward(float *input, int numData);
        void backward(int numData);
        void update(int numData);

};

#endif