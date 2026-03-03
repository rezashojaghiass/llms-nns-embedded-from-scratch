#pragma once

#include<vector>


template<typename T>
class Tensor4D {

public:
	~Tensor4D() = default;
	Tensor4D(const std::vector<size_t>& shape) : shape_(shape), data_(computeSize(shape), T{}) {}

	//getter for shape
	const std::vector<size_t>& getShape() const {//As we return by reference to a member variable, and we promise that the function is const in the same time, it is necessary to return by const.
		return shape_; }

	//access element at (i,j,k,l)
	const T& getElement(size_t i, size_t j, size_t k, size_t l) const {
		return data_[flattenIndex(i, j, k, l)];
	}

	T& operator()(size_t i, size_t j, size_t k, size_t l) {
		return data_[flattenIndex(i, j, k, l)];
	}
	const T& operator()(size_t i, size_t j, size_t k, size_t l) const {
		return data_[flattenIndex(i, j, k, l)];
	}



private:
	std::vector<size_t> shape_;
	std::vector<T> data_;

	static size_t computeSize(const std::vector<size_t>& shape_) {
		size_t size = 1;
		
		for (size_t dim : shape_) {
			size *= dim;
		}
		return size;
	}

	size_t flattenIndex(size_t i, size_t j, size_t k, size_t l) const {
		// Assumes shape_.size() == 4
		return  i * (shape_[1] * shape_[2] * shape_[3]) + j * (shape_[2] * shape_[3]) + k * shape_[3] + l;

	}



};