#ifndef BASE_H
#define BASE_H

class Base{
    public:
        float *input, *output;
        int numOut=-1; //number of output nodes

        virtual float* forward(float *input, int numData){ return nullptr; };
        virtual void backward(int numData){};
        virtual void update(int numData){};
};

#endif