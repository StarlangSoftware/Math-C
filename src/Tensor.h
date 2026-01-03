#ifndef MATH_TENSOR_H
#define MATH_TENSOR_H

// Define the Tensor structure
typedef struct {
    double *data;       // Flattened data array
    int *shape;         // Shape of the tensor
    int *strides;       // Strides for each dimension
    int dimensions;     // Number of dimensions
    int total_elements;
} Tensor;
typedef Tensor *Tensor_ptr;

Tensor_ptr create_tensor(const double *data, const int *shape, int dimensions);

Tensor_ptr create_tensor2(const int *shape, int dimensions);

Tensor_ptr create_tensor3(double *data, const int *shape, int dimensions);

/**
 * Frees the memory allocated for a tensor.
 *
 * @param tensor Pointer to the tensor to be freed.
 */
void free_tensor(Tensor_ptr tensor);

Tensor_ptr concat_tensor(const Tensor* tensor1, const Tensor* tensor2, int dimension);

Tensor_ptr tensor_get(const Tensor* tensor, const int* dimensions, int size);

void unflatten_index(int* indices, int flat_index, const int *strides, int dimensions);

/**
 * Retrieves the value at the given indices.
 *
 * @param tensor Pointer to the tensor.
 * @param indices Array of indices specifying the position. Must have tensor->dimensions elements.
 * @return Value at the specified position. Exits on invalid indices.
 */
double get_tensor_value(const Tensor* tensor, const int *indices);

/**
 * Sets the value at the given indices.
 *
 * @param tensor Pointer to the tensor.
 * @param indices Array of indices specifying the position. Must have tensor->dimensions elements.
 * @param value Value to set at the specified position. Exits on invalid indices.
 */
void set_tensor_value(Tensor_ptr tensor, const int *indices, double value);

/**
 * Reshapes the tensor to the specified new shape.
 * Returns a new tensor with the specified shape. The total number of elements must be the same.
 *
 * @param tensor Pointer to the tensor.
 * @param new_shape Array representing the new shape.
 * @param new_dimensions Size of the new shape array.
 * @return Pointer to the reshaped tensor. Returns NULL on failure or if element count changes.
 */
Tensor_ptr reshape_tensor(const Tensor* tensor, const int *new_shape, int new_dimensions);

/**
 * Transposes the tensor according to the specified axes.
 * Returns a new tensor with transposed axes.
 *
 * @param tensor Pointer to the tensor.
 * @param axes Array representing the order of axes. Must be a permutation of 0 to dimensions-1.
 * If NULL, reverses the axes.
 * @return Pointer to the transposed tensor. Returns NULL on failure or invalid axes.
 */
Tensor_ptr transpose_tensor(const Tensor* tensor, const int *axes);

/**
 * Computes the broadcasted shape of two tensors.
 *
 * @param shape1 Array representing the first tensor shape.
 * @param dimensions1 Size of the first shape array.
 * @param shape2 Array representing the second tensor shape.
 * @param dimensions2 Size of the second shape array.
 * @return A new array for the broadcast shape. Returns NULL if shapes are not broadcastable.
 */
int *compute_broadcast_shape(const int *shape1, int dimensions1, const int *shape2, int dimensions2);

/**
 * Broadcasts the tensor to the specified target shape.
 * Returns a new tensor with the target shape.
 *
 * @param tensor Pointer to the tensor.
 * @param target_shape Array representing the target shape.
 * @param target_shape_size Size of the target shape array.
 * @return New tensor with the target shape. Returns NULL on failure or if broadcasting is not possible.
 */
Tensor_ptr broadcast_to(const Tensor* tensor, const int *target_shape, int target_shape_size);

/**
 * Adds two tensors element-wise with broadcasting.
 * Returns a new tensor with the result of the addition.
 *
 * @param tensor1 Pointer to the first tensor.
 * @param tensor2 Pointer to the second tensor.
 * @return New tensor with the result. Returns NULL on failure or if shapes are not broadcastable.
 */
Tensor_ptr add_tensors(const Tensor* tensor1, const Tensor* tensor2);

/**
 * Subtracts one tensor from another element-wise with broadcasting.
 * Returns a new tensor with the result of the subtraction.
 *
 * @param tensor1 Pointer to the first tensor.
 * @param tensor2 Pointer to the second tensor.
 * @return New tensor with the result. Returns NULL on failure or if shapes are not broadcastable.
 */
Tensor_ptr subtract_tensors(const Tensor* tensor1, const Tensor* tensor2);

/**
 * Multiplies two tensors element-wise with broadcasting.
 * Returns a new tensor with the result of the multiplication.
 *
 * @param tensor1 Pointer to the first tensor.
 * @param tensor2 Pointer to the second tensor.
 * @return New tensor with the result. Returns NULL on failure or if shapes are not broadcastable.
 */
Tensor_ptr hadamard_product(const Tensor* tensor1, const Tensor* tensor2);

/**
 * Computes the dot product of two tensors.
 * Returns a new tensor with the result of the dot product.
 *
 * @param tensor1 Pointer to the first tensor.
 * @param tensor2 Pointer to the second tensor.
 * @return New tensor with the result. Returns NULL on failure or if shapes are not aligned.
 */
Tensor_ptr multiply_tensors(const Tensor* tensor1, const Tensor* tensor2);

/**
 * Extracts a sub-tensor from the given start indices to the end indices (exclusive).
 * Returns a new Tensor containing the extracted sub-tensor.
 *
 * @param tensor Pointer to the tensor.
 * @param start_indices Array specifying the start indices for each dimension. Must have tensor->dimensions elements.
 * @param end_indices Array specifying the end indices (exclusive) for each dimension. Must have tensor->dimensions elements.
 * @return A new Tensor containing the extracted sub-tensor. Returns NULL on failure or invalid indices.
 */
Tensor_ptr partial_tensor(const Tensor* tensor, const int *start_indices, const int *end_indices);

/**
 * Prints the tensor data for debugging purposes.
 *
 * @param tensor Pointer to the tensor.
 */
void print_tensor(const Tensor* tensor);

void update_tensor_data(Tensor_ptr tensor, const double* data);

#endif // MATH_TENSOR_H
