#include "Layer.h"
#include <stdexcept>

//// Example implementation for DenseLayer forward (no activation, no batch)
void DenseLayer::forward(const Tensor3D<float>& input, Tensor3D<float>& output) {
    // input: [inFeatures, 1, 1]
    // weights: [outFeatures, inFeatures, 1]
    // bias: [outFeatures, 1, 1]
    // output: [outFeatures, 1, 1]
    for (size_t o = 0; o < numOutputNeurons; ++o) {
        float sum = 0.0f;
        for (size_t i = 0; i < numInputNeurons; ++i) {
            sum += input(i, 0, 0) * weights(o, i, 0);
        }
        output(o, 0, 0) = sum + bias(o, 0, 0);
    }
}

// Example implementation for ConvLayer forward (stride and padding = 0/1, no batch)
void ConvLayer::forward(const Tensor3D<float>& input, Tensor3D<float>& output) {
    size_t inRows = input.getRows();
    size_t inCols = input.getCols();
    size_t outRows = (inRows + 2 * padding - kernelHeight) / stride + 1;
    size_t outCols = (inCols + 2 * padding - kernelWidth) / stride + 1;

    for (size_t oc = 0; oc < outChannels; ++oc) {
        for (size_t orow = 0; orow < outRows; ++orow) {
            for (size_t ocol = 0; ocol < outCols; ++ocol) {

                float sum = 0.0f;

                for (size_t ic = 0; ic < inChannels; ++ic) {

                    for (size_t krow = 0; krow < kernelHeight; ++krow) {
                        for (size_t kcol = 0; kcol < kernelWidth; ++kcol) {

                            int irow = static_cast<int>(orow * stride + krow - padding);
                            int icol = static_cast<int>(ocol * stride + kcol - padding);

                            float inputVal = 0.0f;
                            
                            if (irow >= 0 && irow < static_cast<int>(inRows) &&
                                icol >= 0 && icol < static_cast<int>(inCols)) {
                                inputVal = input(ic, irow, icol);
                            }
                            sum += inputVal * weights(oc, ic, krow * kernelWidth + kcol);//ic is exactly equal to the kernel depth
                        }
                    }
                }
                output(oc, orow, ocol) = sum + bias(oc, 0, 0);
            }
        }
    }
}

 
