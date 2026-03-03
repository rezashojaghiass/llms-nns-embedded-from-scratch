#pragma once
#include "Tensor3D.h"
class Layer {
public:
	virtual ~Layer() {}
	virtual void forward() = 0;
	virtual void backward() = 0;

};


class DenseLayer : public Layer {
public:
	DenseLayer(size_t numInputNeurons, size_t numOutputNeurons) : numInputNeurons(numInputNeurons), numOutputNeurons(numOutputNeurons),
		weights(numOutputNeurons, numInputNeurons, 1), // Example shape
		bias(numOutputNeurons, 1, 1) {}

	~DenseLayer() override = default;
	void forward() override;
	void backward() override;
	// For DenseLayer
	void setWeights(const Tensor3D<float>& newWeights) { weights = newWeights; }
	void setBias(const Tensor3D<float>& newBias) { bias = newBias; }

private:
	Tensor3D<float> weights;
	Tensor3D<float> bias;
	size_t numInputNeurons;
	size_t numOutputNeurons;
};

class ConvLayer : public Layer {
public:
	~ConvLayer() override = default;

	ConvLayer(size_t inChannels, size_t outChannels, size_t kernelHeight, size_t kernelWidth,
		size_t stride = 1, size_t padding = 0)
		: inChannels(inChannels), outChannels(outChannels),
		kernelHeight(kernelHeight), kernelWidth(kernelWidth),
		stride(stride), padding(padding),
		weights(outChannels, inChannels, kernelHeight* kernelWidth),
		bias(outChannels, 1, 1) {
	}
	
	void forward() override;

	// For ConvLayer
	void setWeights(const Tensor3D<float>& newWeights) { weights = newWeights; }
	void setBias(const Tensor3D<float>& newBias) { bias = newBias; }

private:
	size_t inChannels;
	size_t outChannels;
	size_t kernelHeight;
	size_t kernelWidth;
	size_t stride;
	size_t padding;

	Tensor3D<float> weights;
	Tensor3D<float> bias;

};