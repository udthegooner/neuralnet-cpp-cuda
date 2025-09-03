#include "relu.h"

ReLU::ReLU(int n){
    numOut = n;
}

float* ReLU::forward(float *in, int numData){
    input = in;
    output = new float[numData*numOut];

    for (int i=0; i<numData; i++){
        for (int j=0; j<numOut; j++){
            int index = i*numOut+j;
            output[index] = (in[index] > 0) ? in[index] : 0.0f;
        }
    }
    return output;
}

void ReLU::backward(int numData){
    for (int i=0; i<numData; i++){
        for (int j=0; j<numOut; j++){
            int index = i*numOut + j;
            input[index] = (input[index] > 0) * output[index];
        }
    }
    delete[] output;
}
