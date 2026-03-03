#pragma once

/**
Tensor4D<float> tensor3d_as_4d({1, d1, d2, d3});
tensor3d_as_4d.getElement(0, i, j, k) = value; // Use 0 for the first dimension
**/

template <typename T>
class Tensor3D {
private:
	size_t depth, rows, cols;
	T* data;

public:
	Tensor3D(size_t depth, size_t rows, size_t cols) : depth(depth), rows(rows), cols(cols) {
		data = new T[depth * rows * cols]();// () is for value initialization, which initializes built-in types to zero and calls the default constructor for class types.
	};
	~Tensor3D() {
		delete[] data;
		data = nullptr; // Set pointer to nullptr after deletion to avoid dangling pointer issues.
	}

	// Copy constructor:
	// Hint: The object to be copied should be passed as const Reference to avoid unnecessary copying and to ensure that the original object is not modified during the copy process.
	Tensor3D(const Tensor3D& other) : depth(other.depth), rows(other.rows), cols(other.cols) {
		data = new T[depth * rows * cols];
		std::copy(other.data, other.data + depth * rows * cols, data);
	}

	//Assignment(a = b;) requires both a and b to already exist.
	Tensor3D& operator=(const Tensor3D& other) {//The assignment operator should return a reference to the current object to allow for chaining assignments (e.g., a = b = c;).
		if (this != &other) {// Check for self-assignment to avoid unnecessary work and potential issues when an object is assigned to itself.
			delete[] data;//free existing resources to prevent memory leaks before allocating new memory for the copy.
			depth = other.depth;
			rows = other.rows;
			cols = other.cols;
			data = new T[depth * rows * cols];
			std::copy(other.data, other.data + depth * rows * cols, data);
		}
		return *this;// This is how we return a reference to the current object, allowing for chaining assignments and ensuring that the assignment operator behaves as expected in C++.
	}


	// Move constructor
	Tensor3D(Tensor3D&& other) noexcept//noexcept is used to indicate that the move constructor will not throw exceptions, which can help with optimization and allows certain operations to be performed more efficiently.
		: depth(other.depth), rows(other.rows), cols(other.cols), data(other.data) {//the ownership is transferred here, so we just copy the pointer and dimensions from the source object to the new object.
		other.data = nullptr;// After transferring ownership, we set the source object's data pointer to nullptr to prevent it from trying to free the memory when it is destroyed, which would lead to a double-free error. We also reset the dimensions of the source object to zero to indicate that it no longer owns any data.
		other.depth = other.rows = other.cols = 0;// This is a common practice in move semantics to ensure that the moved-from object is left in a valid but empty state, preventing any accidental use of the moved-from object after the move operation.
	}

	// Move assignment operator
	Tensor3D& operator=(Tensor3D&& other) noexcept {
		if (this != &other) {
			delete[] data;
			data = other.data;//The ownership of the data is transferred from the source object (other) to the current object (this) by copying the pointer and dimensions. This allows the current object to take ownership of the resources without needing to perform a deep copy, which can be more efficient.
			depth = other.depth;
			rows = other.rows;
			cols = other.cols;
			other.data = nullptr;
			other.depth = other.rows = other.cols = 0;
		}
		return *this;
	}


	T& operator()(size_t d, size_t r, size_t c) {
		return data[d * rows * cols + r * cols + c];
	}
	const T& operator()(size_t d, size_t r, size_t c) const {
		return data[d * rows * cols + r * cols + c];
	}

	template <typename T>
	Tensor3D<T> flatten() const {
		Tensor3D<T> flat(depth * rows * cols, 1, 1);
		for (size_t d = 0; d < depth; ++d)
			for (size_t r = 0; r < rows; ++r)
				for (size_t c = 0; c < cols; ++c)
					flat(d * rows * cols + r * cols + c, 0, 0) = (*this)(d, r, c);
		return flat;
	}

};