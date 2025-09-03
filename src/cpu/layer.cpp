#include "layer.h"
#include "utils.h"
#include <iostream>
#include <algorithm>
#include <vector>

Layer::Layer(int nIn, int nOut, float _lr){
    numIn = nIn; //number of input nodes
    numOut = nOut; //number of output nodes
    lr = _lr;

    weightSize = nIn*nOut;
    
    weights = new float[weightSize]; // stored as numIn rows, numOut cols
    kaiming_init(weights, nIn, nOut); // initialize weights
    // ones(weights, nIn*nOut);

    bias = new float[nOut];
    zeroes(bias, nOut);
}

float* Layer::forward(float *_input, int numData){
    input = _input;
    output = new float[numData*numOut];
    zeroes(output, numData*numOut);
    
    //iterate over each data point in input
    for (int i=0; i<numData; i++){

        // iterate over each input node/feature of data point
        for (int j=0; j<numIn; j++){
            int inIndex = i*numIn + j;
            float inVal = input[inIndex];

            // iterate over each output node
            for (int k=0; k<numOut; k++){
                int outindex = i*numOut + k;
                int wtIndex = j*numOut + k;
                float prod = weights[wtIndex]*inVal;
                output[outindex] += prod;
            }
        }
        
        // add bias to output
        for (int p=0; p<numOut; p++){
            int outIndex = i*numOut + p;
            output[outIndex] += bias[p];
        }
    }
    return output;
}

void Layer::update(int numData){
    // calculate column sum of gradient
    std::vector<float> sumGrad(numOut, 0.0f); // size = numOut, initialized to 0
    for (int i=0; i<numData; i++)
        for (int j=0; j<numOut; j++)
            sumGrad[j] += output[i*numOut+j];
    
    // update bias
    for (int i=0; i<numOut; i++)
        bias[i] -= lr*sumGrad[i]/numData;
            
    // calculate product of gradient and input data
    float* prodGrad = new float[weightSize];
    std::fill(prodGrad, prodGrad + weightSize, 0.0f);
    for (int i=0; i<numData; i++){ //iterate over data

        for (int j=0; j<numIn; j++){ //iterate over input nodes
            float inputTerm = input[i*numIn + j];

            for (int k=0; k<numOut; k++){ //iterate over output/gradient nodes
                float gradTerm = output[i*numOut+k];

                int index = j*numOut + k;
                // if (i==0) prodGrad[index] = 0.0f;
                prodGrad[index] += inputTerm*gradTerm;
            }
        }
    }

    // update weights 
    newWeights = new float[weightSize];
    for (int i=0; i<numIn; i++){
        for (int j=0; j<numOut; j++){
            int index = i*numOut+j;
            newWeights[index] = weights[index] - lr*(prodGrad[index] / float(numData));
        }
    }
    delete[] prodGrad;
}

void Layer::backward(int numData){
    // calculate prod of gradient and (old) weights
    for (int i=0; i<numData; i++){ //iterate over data

        for (int j=0; j<numIn; j++){ //iterate over weights
            int index = i*numIn + j;

            for (int k=0; k<numOut; k++){ //iterate over output/gradient nodes
                float weightTerm = weights[j*numOut + k];
                float gradTerm = output[i*numOut + k];

                if (k==0) input[index] = 0.0f;
                input[index] += weightTerm*gradTerm;
            }
        }
    }

    delete[] weights;
    weights = newWeights;
    delete[] output;
}
