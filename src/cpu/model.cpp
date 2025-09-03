#include "model.h"
#include <iostream>
#include "utils.h"
using namespace std;

Model::Model(Base** _layers, int _numLayers){
    layers = _layers;
    numLayers = _numLayers;
}

float* Model::forward(float *input, int numData){
    float *currOut;
    int sizeOut, i;

    for (i=0; i<numLayers; i++){
        Base* layer = layers[i];
        sizeOut = layer->numOut;
        currOut = layer->forward(input, numData);

        input = currOut;
    }
    return currOut;
}

void Model::update(int numData){

    for (int i=numLayers-1; i>=0; i--){
        Base* layer = layers[i];
        layer->update(numData);
        layer->backward(numData);
    }
}