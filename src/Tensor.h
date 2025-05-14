#ifndef MATH_TENSOR_H
#define MATH_TENSOR_H

#include <stddef.h> // For size_t
#include <stdbool.h> // For bool type
#include <stdio.h> // For NULL
#include "../../DataStructure-C/src/Memory/Memory.h" // Include your custom memory management header

// Define the Tensor structure
typedef struct {
    double *data;       // Flattened data array
    int *shape;         // Shape of the tensor
    int *strides;       // Strides for each dimension
    int dimensions;     // Number of dimensions
} Tensor;

/**
 * Initializes a tensor with given data and shape.
 *
 * @param data Flattened array representing the tensor data.
 * @param shape Array representing the shape of the tensor.
 * @param dimensions Size of the shape array.
 * @return Pointer to the created tensor. Returns NULL on failure.
 */
Tensor *create_tensor(const double *data, const int *shape, int dimensions);

/**
 * Frees the memory allocated for a tensor.
 *
 * @param tensor Pointer to the tensor to be freed.
 */
void free_tensor(Tensor *tensor);

/**
 * Retrieves the value at the given indices.
 *
 * @param tensor Pointer to the tensor.
 * @param indices Array of indices specifying the position. Must have tensor->dimensions elements.
 * @return Value at the specified position. Exits on invalid indices.
 */
double get_tensor_value(const Tensor *tensor, const int *indices);

/**
 * Sets the value at the given indices.
 *
 * @param tensor Pointer to the tensor.
 * @param indices Array of indices specifying the position. Must have tensor->dimensions elements.
 * @param value Value to set at the specified position. Exits on invalid indices.
 */
void set_tensor_value(Tensor *tensor, const int *indices, double value);

/**
 * Reshapes the tensor to the specified new shape.
 * Returns a new tensor with the specified shape. The total number of elements must be the same.
 *
 * @param tensor Pointer to the tensor.
 * @param new_shape Array representing the new shape.
 * @param new_dimensions Size of the new shape array.
 * @return Pointer to the reshaped tensor. Returns NULL on failure or if element count changes.
 */
Tensor *reshape_tensor(const Tensor *tensor, const int *new_shape, int new_dimensions);

/**
 * Transposes the tensor according to the specified axes.
 * Returns a new tensor with transposed axes.
 *
 * @param tensor Pointer to the tensor.
 * @param axes Array representing the order of axes. Must be a permutation of 0 to dimensions-1.
 * If NULL, reverses the axes.
 * @return Pointer to the transposed tensor. Returns NULL on failure or invalid axes.
 */
Tensor *transpose_tensor(const Tensor *tensor, const int *axes);

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
Tensor *broadcast_to(const Tensor *tensor, const int *target_shape, int target_shape_size);

/**
 * Adds two tensors element-wise with broadcasting.
 * Returns a new tensor with the result of the addition.
 *
 * @param tensor1 Pointer to the first tensor.
 * @param tensor2 Pointer to the second tensor.
 * @return New tensor with the result. Returns NULL on failure or if shapes are not broadcastable.
 */
Tensor *add_tensors(const Tensor *tensor1, const Tensor *tensor2);

/**
 * Subtracts one tensor from another element-wise with broadcasting.
 * Returns a new tensor with the result of the subtraction.
 *
 * @param tensor1 Pointer to the first tensor.
 * @param tensor2 Pointer to the second tensor.
 * @return New tensor with the result. Returns NULL on failure or if shapes are not broadcastable.
 */
Tensor *subtract_tensors(const Tensor *tensor1, const Tensor *tensor2);

/**
 * Multiplies two tensors element-wise with broadcasting.
 * Returns a new tensor with the result of the multiplication.
 *
 * @param tensor1 Pointer to the first tensor.
 * @param tensor2 Pointer to the second tensor.
 * @return New tensor with the result. Returns NULL on failure or if shapes are not broadcastable.
 */
Tensor *multiply_tensors(const Tensor *tensor1, const Tensor *tensor2);

/**
 * Computes the dot product of two tensors.
 * Returns a new tensor with the result of the dot product.
 *
 * @param tensor1 Pointer to the first tensor.
 * @param tensor2 Pointer to the second tensor.
 * @return New tensor with the result. Returns NULL on failure or if shapes are not aligned.
 */
Tensor *dot_product(const Tensor *tensor1, const Tensor *tensor2);

/**
 * Extracts a sub-tensor from the given start indices to the end indices (exclusive).
 * Returns a new Tensor containing the extracted sub-tensor.
 *
 * @param tensor Pointer to the tensor.
 * @param start_indices Array specifying the start indices for each dimension. Must have tensor->dimensions elements.
 * @param end_indices Array specifying the end indices (exclusive) for each dimension. Must have tensor->dimensions elements.
 * @return A new Tensor containing the extracted sub-tensor. Returns NULL on failure or invalid indices.
 */
Tensor *partial_tensor(const Tensor *tensor, const int *start_indices, const int *end_indices);

/**
 * Prints the tensor data for debugging purposes.
 *
 * @param tensor Pointer to the tensor.
 */
void print_tensor(const Tensor *tensor);

// Helper functions (might be declared here or kept static in the .c file)
// Declaring them here makes them public, but they are primarily internal helpers.
// Keeping them static is generally preferred unless they need to be called externally.
// For this example, we'll keep them static in the .c file as they are implementation details.
/*
int compute_total_elements(const int *shape, int dimensions);
int *compute_strides(const int *shape, int dimensions);
void validate_indices(const Tensor *tensor, const int *indices);
int *unflatten_index(int flat_index, const int *shape, const int *strides, int dimensions);
*/

#endif // MATH_TENSOR_H