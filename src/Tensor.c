#include "Tensor.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>


// Helper function to compute the total number of elements in a tensor
int compute_total_elements(const int *shape, int dimensions) {
    int total_elements = 1;
    for (int i = 0; i < dimensions; i++) {
        if (shape[i] <= 0) return 0;
        total_elements *= shape[i];
    }
    return total_elements;
}

// Helper function to compute the strides for each dimension based on the shape.
int *compute_strides(const int *shape, int dimensions) {
    int *strides = malloc(dimensions * sizeof(int)); 
    if (!strides) {
        perror("Failed to allocate memory for strides");
        exit(EXIT_FAILURE);
    }
    int stride = 1;
    for (int i = dimensions - 1; i >= 0; i--) {
        strides[i] = stride;
        stride *= shape[i];
    }
    return strides;
}

// Helper function to validate indices are within the valid range for each dimension.
void validate_indices(const Tensor *tensor, const int *indices) {
    if (!tensor || !indices) {
        fprintf(stderr, "Error: Tensor or indices are NULL.\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < tensor->dimensions; i++) {
        if (indices[i] < 0 || indices[i] >= tensor->shape[i]) {
            fprintf(stderr, "Error: Index %d out of bounds for dimension %d with size %d.\n", indices[i], i, tensor->shape[i]);
            exit(EXIT_FAILURE); 
        }
    }
}

// Helper function to convert a flat index to multi-dimensional indices based on strides.
int *unflatten_index(int flat_index, const int *shape, const int *strides, int dimensions) {
    int *indices = malloc(dimensions * sizeof(int)); // Use standard malloc
    if (!indices) {
        perror("Failed to allocate memory for unflattened indices");
        exit(EXIT_FAILURE);
    }
    int temp_flat_index = flat_index;
    for (int i = 0; i < dimensions; i++) {
        if (strides[i] == 0) { 
             indices[i] = 0;
        } else {
             indices[i] = temp_flat_index / strides[i];
             temp_flat_index %= strides[i];
        }
    }
    return indices;
}


/**
 * Initializes a tensor with given data and shape.
 *
 * @param data Flattened array representing the tensor data.
 * @param shape Array representing the shape of the tensor.
 * @param dimensions Size of the shape array.
 * @return Pointer to the created tensor. Returns NULL on failure.
 */
Tensor *create_tensor(const double *data, const int *shape, int dimensions) {
    if (!shape || dimensions <= 0) {
        fprintf(stderr, "Error: Invalid shape or dimensions.\n");
        return NULL;
    }

    int total_elements = compute_total_elements(shape, dimensions);
    if (total_elements == 0 && data != NULL) {
         fprintf(stderr, "Warning: Shape indicates zero elements, but data is not NULL.\n");
    }
    if (total_elements > 0 && data == NULL) {
         fprintf(stderr, "Error: Shape indicates elements, but data is NULL.\n");
         return NULL;
    }


    Tensor *tensor = malloc(sizeof(Tensor)); 
    if (!tensor) {
        perror("Failed to allocate memory for Tensor");
        return NULL;
    }

    tensor->dimensions = dimensions;
    tensor->shape = malloc(dimensions * sizeof(int)); 
    if (!tensor->shape) {
        perror("Failed to allocate memory for tensor shape");
        free(tensor);
        return NULL;
    }
    memcpy(tensor->shape, shape, dimensions * sizeof(int));

    tensor->strides = compute_strides(tensor->shape, dimensions);
    if (!tensor->strides) {
        free(tensor->shape);
        free(tensor);
        return NULL;
    }

    tensor->data = malloc(total_elements * sizeof(double)); 
    if (!tensor->data) {
        perror("Failed to allocate memory for tensor data");
        free(tensor->strides);
        free(tensor->shape);
        free(tensor);
        return NULL;
    }
    if (total_elements > 0 && data != NULL) {
        memcpy(tensor->data, data, total_elements * sizeof(double));
    }


    return tensor;
}

/**
 * Frees the memory allocated for a tensor.
 *
 * @param tensor Pointer to the tensor to be freed.
 */
void free_tensor(Tensor *tensor) {
    if (tensor) {
        free(tensor->data);
        free(tensor->shape);
        free(tensor->strides);
        free(tensor);
    }
}

/**
 * Retrieves the value at the given indices.
 *
 * @param tensor Pointer to the tensor.
 * @param indices Array of indices specifying the position.
 * @return Value at the specified position.
 */
double get_tensor_value(const Tensor *tensor, const int *indices) {
    validate_indices(tensor, indices); 

    int flat_index = 0;
    for (int i = 0; i < tensor->dimensions; i++) {
        flat_index += indices[i] * tensor->strides[i];
    }
    return tensor->data[flat_index];
}

/**
 * Sets the value at the given indices.
 *
 * @param tensor Pointer to the tensor.
 * @param indices Array of indices specifying the position.
 * @param value Value to set at the specified position.
 */
void set_tensor_value(Tensor *tensor, const int *indices, double value) {
    validate_indices(tensor, indices); 

    int flat_index = 0;
    for (int i = 0; i < tensor->dimensions; i++) {
        flat_index += indices[i] * tensor->strides[i];
    }
    tensor->data[flat_index] = value;
}

/**
 * Reshapes the tensor to the specified new shape.
 * Returns a new tensor with the specified shape.
 *
 * @param tensor Pointer to the tensor.
 * @param new_shape Array representing the new shape.
 * @param new_dimensions Size of the new shape array.
 * @return Pointer to the reshaped tensor. Returns NULL on failure.
 */
Tensor *reshape_tensor(const Tensor *tensor, const int *new_shape, int new_dimensions) {
    if (!tensor || !new_shape || new_dimensions <= 0) {
        fprintf(stderr, "Error: Invalid input for reshape_tensor.\n");
        return NULL;
    }

    int total_elements_new = compute_total_elements(new_shape, new_dimensions);
    int total_elements_current = compute_total_elements(tensor->shape, tensor->dimensions);

    if (total_elements_new != total_elements_current) {
        fprintf(stderr, "Error: Total number of elements must remain the same during reshape.\n");
        return NULL; 
    }

    // Create a new tensor with the same data but new shape and strides
    Tensor *reshaped = create_tensor(tensor->data, new_shape, new_dimensions);
    return reshaped;
}

/**
 * Transposes the tensor according to the specified axes.
 * Returns a new tensor with transposed axes.
 *
 * @param tensor Pointer to the tensor.
 * @param axes Array representing the order of axes. If NULL, reverses the axes.
 * @return Pointer to the transposed tensor. Returns NULL on failure.
 */
Tensor *transpose_tensor(const Tensor *tensor, const int *axes) {
    if (!tensor) {
        fprintf(stderr, "Error: Input tensor is NULL for transpose.\n");
        return NULL;
    }

    int dimensions = tensor->dimensions;
    int *actual_axes = NULL;
    int *new_shape = malloc(dimensions * sizeof(int)); 
    if (!new_shape) {
         perror("Failed to allocate memory for new_shape in transpose");
         return NULL;
    }

    if (axes == NULL) {
        // Reverse axes if not provided
        actual_axes = malloc(dimensions * sizeof(int)); 
        if (!actual_axes) {
             perror("Failed to allocate memory for actual_axes in transpose");
             free(new_shape);
             return NULL;
        }
        for (int i = 0; i < dimensions; i++) {
            actual_axes[i] = dimensions - 1 - i;
            new_shape[i] = tensor->shape[actual_axes[i]];
        }
    } else {
        // Use provided axes and validate
        actual_axes = malloc(dimensions * sizeof(int)); 
         if (!actual_axes) {
             perror("Failed to allocate memory for actual_axes in transpose");
             free(new_shape);
             return NULL;
        }
        memcpy(actual_axes, axes, dimensions * sizeof(int));

        // Basic validation: check if axes are a permutation of 0 to dimensions-1
        int *check = calloc(dimensions, sizeof(int)); 
        if (!check) {
             perror("Failed to allocate memory for check in transpose");
             free(actual_axes);
             free(new_shape);
             return NULL;
        }
        for (int i = 0; i < dimensions; i++) {
            if (actual_axes[i] < 0 || actual_axes[i] >= dimensions || check[actual_axes[i]] > 0) {
                fprintf(stderr, "Error: Invalid transpose axes.\n");
                free(actual_axes);
                free(new_shape);
                free(check);
                return NULL;
            }
            check[actual_axes[i]] = 1;
            new_shape[i] = tensor->shape[actual_axes[i]];
        }
        free(check);
    }

    int total_elements = compute_total_elements(tensor->shape, dimensions);
    double *new_data = malloc(total_elements * sizeof(double)); 
     if (!new_data) {
         perror("Failed to allocate memory for new_data in transpose");
         free(actual_axes);
         free(new_shape);
         return NULL;
     }

    int *new_strides = compute_strides(new_shape, dimensions);
     if (!new_strides) {
         free(actual_axes);
         free(new_shape);
         free(new_data);
         return NULL;
     }


    // Rearrange data
    int *original_indices = malloc(dimensions * sizeof(int)); 
    int *new_indices = malloc(dimensions * sizeof(int));
     if (!original_indices || !new_indices) {
         perror("Failed to allocate memory for indices in transpose");
         free(actual_axes);
         free(new_shape);
         free(new_data);
         free(new_strides);
         free(original_indices); 
         free(new_indices); 
         return NULL;
     }

    for (int i = 0; i < total_elements; i++) {
        // Get new indices from flat index
        int temp_flat_index = i;
        for (int j = 0; j < dimensions; j++) {
             if (new_strides[j] == 0) {
                 new_indices[j] = 0;
             } else {
                 new_indices[j] = temp_flat_index / new_strides[j];
                 temp_flat_index %= new_strides[j];
             }
        }


        // Map new indices back to original indices using the axes permutation
        for (int j = 0; j < dimensions; j++) {
             original_indices[actual_axes[j]] = new_indices[j];
        }


        // Calculate original flat index
        int original_flat_index = 0;
        for (int j = 0; j < dimensions; j++) {
            original_flat_index += original_indices[j] * tensor->strides[j];
        }

        new_data[i] = tensor->data[original_flat_index];
    }

    free(original_indices);
    free(new_indices);
    free(actual_axes); 

    // Create the new tensor
    Tensor *transposed = create_tensor(new_data, new_shape, dimensions);

    free(new_data); 
    free(new_shape);
    free(new_strides);

    return transposed;
}


// Function to compute the broadcasted shape of two tensors
// Returns a new array for the broadcast shape, or NULL on failure.
int *compute_broadcast_shape(const int *shape1, int dimensions1, const int *shape2, int dimensions2) {
    int max_dimensions = dimensions1 > dimensions2 ? dimensions1 : dimensions2;
    int *broadcast_shape = malloc(max_dimensions * sizeof(int)); 
    if (!broadcast_shape) {
        perror("Failed to allocate memory for broadcast_shape");
        return NULL;
    }

    for (int i = 0; i < max_dimensions; i++) {
        int dim1 = (i < dimensions1) ? shape1[dimensions1 - 1 - i] : 1;
        int dim2 = (i < dimensions2) ? shape2[dimensions2 - 1 - i] : 1;

        if (dim1 != dim2 && dim1 != 1 && dim2 != 1) {
            fprintf(stderr, "Error: Shapes are not broadcastable.\n");
            free(broadcast_shape);
            return NULL;
        }

        broadcast_shape[max_dimensions - 1 - i] = (dim1 > dim2) ? dim1 : dim2;
    }

    return broadcast_shape;
}

// Function to broadcast a tensor to a target shape
// Returns a new broadcasted tensor, or NULL on failure.
Tensor *broadcast_to(const Tensor *tensor, const int *target_shape, int target_shape_size) {
    if (!tensor || !target_shape || target_shape_size <= 0) {
        fprintf(stderr, "Error: Invalid input for broadcast_to.\n");
        return NULL;
    }

    
    int expanded_dimensions = target_shape_size;
    int *expanded_shape = malloc(expanded_dimensions * sizeof(int));
     if (!expanded_shape) {
         perror("Failed to allocate memory for expanded_shape in broadcast_to");
         return NULL;
     }
    int offset = expanded_dimensions - tensor->dimensions;
    for(int i = 0; i < expanded_dimensions; ++i) {
        if (i < offset) expanded_shape[i] = 1;
        else expanded_shape[i] = tensor->shape[i - offset];
    }

    int broadcast_possible = 1;
    for(int i = 0; i < expanded_dimensions; ++i) {
        if (expanded_shape[i] != target_shape[i] && expanded_shape[i] != 1) {
            broadcast_possible = 0;
            break;
        }
    }
    free(expanded_shape); 

    if (!broadcast_possible) {
        fprintf(stderr, "Error: Cannot broadcast shape to target shape.\n");
        return NULL;
    }


    Tensor *broadcasted_tensor = malloc(sizeof(Tensor));
    if (!broadcasted_tensor) {
        perror("Failed to allocate memory for broadcasted_tensor");
        return NULL;
    }

    broadcasted_tensor->dimensions = target_shape_size;
    broadcasted_tensor->shape = malloc(target_shape_size * sizeof(int)); 
    if (!broadcasted_tensor->shape) {
        perror("Failed to allocate memory for broadcasted_tensor shape");
        free(broadcasted_tensor);
        return NULL;
    }
    memcpy(broadcasted_tensor->shape, target_shape, target_shape_size * sizeof(int));

    broadcasted_tensor->strides = compute_strides(broadcasted_tensor->shape, target_shape_size);
     if (!broadcasted_tensor->strides) {
        free(broadcasted_tensor->shape);
        free(broadcasted_tensor);
        return NULL;
    }

    int total_elements = compute_total_elements(target_shape, target_shape_size);
    broadcasted_tensor->data = malloc(total_elements * sizeof(double)); 
     if (!broadcasted_tensor->data) {
        perror("Failed to allocate memory for broadcasted_tensor data");
        free(broadcasted_tensor->strides);
        free(broadcasted_tensor->shape);
        free(broadcasted_tensor);
        return NULL;
    }


    // Pre-allocate index array for efficiency
    int *indices = malloc(target_shape_size * sizeof(int));
     if (!indices) {
         perror("Failed to allocate memory for indices in broadcast_to");
         free(broadcasted_tensor->data);
         free(broadcasted_tensor->strides);
         free(broadcasted_tensor->shape);
         free(broadcasted_tensor);
         return NULL;
     }


    for (int i = 0; i < total_elements; i++) {
        // Convert flat index to multi-dimensional indices for the target shape
        int temp_flat_index = i;
        for (int j = 0; j < target_shape_size; j++) {
            if (broadcasted_tensor->strides[j] == 0) {
                indices[j] = 0;
            } else {
                indices[j] = temp_flat_index / broadcasted_tensor->strides[j];
                temp_flat_index %= broadcasted_tensor->strides[j];
            }
        }

        // Map target indices back to original tensor indices
        int original_index = 0;
        int original_dim_offset = target_shape_size - tensor->dimensions;
        for (int j = 0; j < tensor->dimensions; j++) {
             // Use index 0 if the original dimension size was 1 
             int current_index = (tensor->shape[j] == 1) ? 0 : indices[original_dim_offset + j];
             original_index += current_index * tensor->strides[j];
        }

        broadcasted_tensor->data[i] = tensor->data[original_index];
    }

    free(indices);

    return broadcasted_tensor;
}


// Function to add two tensors
// Returns a new tensor with the result, or NULL on failure.
Tensor *add_tensors(const Tensor *tensor1, const Tensor *tensor2) {
    if (!tensor1 || !tensor2) {
        fprintf(stderr, "Error: Input tensor is NULL for addition.\n");
        return NULL;
    }

    int *broadcast_shape = compute_broadcast_shape(tensor1->shape, tensor1->dimensions, tensor2->shape, tensor2->dimensions);
    if (!broadcast_shape) {
        return NULL; 
    }
    int broadcast_dimensions = tensor1->dimensions > tensor2->dimensions ? tensor1->dimensions : tensor2->dimensions; // Use max dimensions


    Tensor *broadcasted_tensor1 = broadcast_to(tensor1, broadcast_shape, broadcast_dimensions);
    if (!broadcasted_tensor1) {
        free(broadcast_shape);
        return NULL; 
    }

    Tensor *broadcasted_tensor2 = broadcast_to(tensor2, broadcast_shape, broadcast_dimensions);
     if (!broadcasted_tensor2) {
        free_tensor(broadcasted_tensor1);
        free(broadcast_shape);
        return NULL; 
    }


    Tensor *result = malloc(sizeof(Tensor)); 
     if (!result) {
        perror("Failed to allocate memory for result tensor in add_tensors");
        free_tensor(broadcasted_tensor1);
        free_tensor(broadcasted_tensor2);
        free(broadcast_shape);
        return NULL;
    }

    result->dimensions = broadcast_dimensions;
    result->shape = broadcast_shape; 
    result->strides = compute_strides(result->shape, result->dimensions);
     if (!result->strides) {
        free_tensor(broadcasted_tensor1);
        free_tensor(broadcasted_tensor2);
        free(result->shape);
        free(result);
        return NULL;
    }


    int total_elements = compute_total_elements(result->shape, result->dimensions);
    result->data = malloc(total_elements * sizeof(double)); 
     if (!result->data) {
        perror("Failed to allocate memory for result data in add_tensors");
        free_tensor(broadcasted_tensor1);
        free_tensor(broadcasted_tensor2);
        free(result->strides);
        free(result->shape);
        free(result);
        return NULL;
    }


    for (int i = 0; i < total_elements; i++) {
        result->data[i] = broadcasted_tensor1->data[i] + broadcasted_tensor2->data[i];
    }

    free_tensor(broadcasted_tensor1); 
    free_tensor(broadcasted_tensor2);

    return result;
}

// Function to subtract two tensors
// Returns a new tensor with the result, or NULL on failure.
Tensor *subtract_tensors(const Tensor *tensor1, const Tensor *tensor2) {
    if (!tensor1 || !tensor2) {
        fprintf(stderr, "Error: Input tensor is NULL for subtraction.\n");
        return NULL;
    }

    int *broadcast_shape = compute_broadcast_shape(tensor1->shape, tensor1->dimensions, tensor2->shape, tensor2->dimensions);
    if (!broadcast_shape) {
        return NULL;
    }
    int broadcast_dimensions = tensor1->dimensions > tensor2->dimensions ? tensor1->dimensions : tensor2->dimensions;


    Tensor *broadcasted_tensor1 = broadcast_to(tensor1, broadcast_shape, broadcast_dimensions);
    if (!broadcasted_tensor1) {
        free(broadcast_shape);
        return NULL;
    }

    Tensor *broadcasted_tensor2 = broadcast_to(tensor2, broadcast_shape, broadcast_dimensions);
     if (!broadcasted_tensor2) {
        free_tensor(broadcasted_tensor1);
        free(broadcast_shape);
        return NULL;
    }


    Tensor *result = malloc(sizeof(Tensor));
     if (!result) {
        perror("Failed to allocate memory for result tensor in subtract_tensors");
        free_tensor(broadcasted_tensor1);
        free_tensor(broadcasted_tensor2);
        free(broadcast_shape);
        return NULL;
    }

    result->dimensions = broadcast_dimensions;
    result->shape = broadcast_shape; 
    result->strides = compute_strides(result->shape, result->dimensions);
     if (!result->strides) {
        free_tensor(broadcasted_tensor1);
        free_tensor(broadcasted_tensor2);
        free(result->shape);
        free(result);
        return NULL;
    }


    int total_elements = compute_total_elements(result->shape, result->dimensions);
    result->data = malloc(total_elements * sizeof(double));
     if (!result->data) {
        perror("Failed to allocate memory for result data in subtract_tensors");
        free_tensor(broadcasted_tensor1);
        free_tensor(broadcasted_tensor2);
        free(result->strides);
        free(result->shape);
        free(result);
        return NULL;
    }


    for (int i = 0; i < total_elements; i++) {
        result->data[i] = broadcasted_tensor1->data[i] - broadcasted_tensor2->data[i];
    }

    free_tensor(broadcasted_tensor1);
    free_tensor(broadcasted_tensor2);

    return result;
}

// Function to multiply two tensors
// Returns a new tensor with the result, or NULL on failure.
Tensor *multiply_tensors(const Tensor *tensor1, const Tensor *tensor2) {
    if (!tensor1 || !tensor2) {
        fprintf(stderr, "Error: Input tensor is NULL for multiplication.\n");
        return NULL;
    }

    int *broadcast_shape = compute_broadcast_shape(tensor1->shape, tensor1->dimensions, tensor2->shape, tensor2->dimensions);
     if (!broadcast_shape) {
        return NULL;
    }
    int broadcast_dimensions = tensor1->dimensions > tensor2->dimensions ? tensor1->dimensions : tensor2->dimensions;


    Tensor *broadcasted_tensor1 = broadcast_to(tensor1, broadcast_shape, broadcast_dimensions);
    if (!broadcasted_tensor1) {
        free(broadcast_shape);
        return NULL;
    }

    Tensor *broadcasted_tensor2 = broadcast_to(tensor2, broadcast_shape, broadcast_dimensions);
     if (!broadcasted_tensor2) {
        free_tensor(broadcasted_tensor1);
        free(broadcast_shape);
        return NULL;
    }


    Tensor *result = malloc(sizeof(Tensor)); 
     if (!result) {
        perror("Failed to allocate memory for result tensor in multiply_tensors");
        free_tensor(broadcasted_tensor1);
        free_tensor(broadcasted_tensor2);
        free(broadcast_shape);
        return NULL;
    }

    result->dimensions = broadcast_dimensions;
    result->shape = broadcast_shape; 
    result->strides = compute_strides(result->shape, result->dimensions);
     if (!result->strides) {
        free_tensor(broadcasted_tensor1);
        free_tensor(broadcasted_tensor2);
        free(result->shape);
        free(result);
        return NULL;
    }


    int total_elements = compute_total_elements(result->shape, result->dimensions);
    result->data = malloc(total_elements * sizeof(double)); 
     if (!result->data) {
        perror("Failed to allocate memory for result data in multiply_tensors");
        free_tensor(broadcasted_tensor1);
        free_tensor(broadcasted_tensor2);
        free(result->strides);
        free(result->shape);
        free(result);
        return NULL;
    }


    for (int i = 0; i < total_elements; i++) {
        result->data[i] = broadcasted_tensor1->data[i] * broadcasted_tensor2->data[i];
    }

    free_tensor(broadcasted_tensor1);
    free_tensor(broadcasted_tensor2);

    return result;
}

// Function to compute the dot product of two tensors
// Returns a new tensor with the result, or NULL on failure.
Tensor *dot_product(const Tensor *tensor1, const Tensor *tensor2) {
    if (!tensor1 || !tensor2) {
        fprintf(stderr, "Error: Input tensor is NULL for dot product.\n");
        return NULL;
    }

    if (tensor1->dimensions == 0 || tensor2->dimensions == 0) {
         fprintf(stderr, "Error: Cannot perform dot product on scalar tensors.\n");
         return NULL;
    }


    if (tensor1->shape[tensor1->dimensions - 1] != tensor2->shape[tensor2->dimensions - 2]) {
        fprintf(stderr, "Error: Shapes are not aligned for dot product. Last dim of tensor1 (%d) != second to last dim of tensor2 (%d).\n",
                tensor1->shape[tensor1->dimensions - 1], tensor2->shape[tensor2->dimensions - 2]);
        return NULL; 
    }

    int result_dimensions = (tensor1->dimensions - 1) + (tensor2->dimensions - 1); // Result dimensions
    if (result_dimensions == 0) result_dimensions = 1; // Handle scalar result case

    int *result_shape = malloc(result_dimensions * sizeof(int));
    if (!result_shape) {
        perror("Failed to allocate memory for result_shape in dot_product");
        return NULL;
    }

    int shape_idx = 0;
    // Copy shape from tensor1 (excluding the last dimension)
    for (int i = 0; i < tensor1->dimensions - 1; i++) {
        result_shape[shape_idx++] = tensor1->shape[i];
    }
    // Copy shape from tensor2 (excluding the first dimension)
    for (int i = 1; i < tensor2->dimensions; i++) {
         result_shape[shape_idx++] = tensor2->shape[i];
    }


    Tensor *result = malloc(sizeof(Tensor)); 
     if (!result) {
        perror("Failed to allocate memory for result tensor in dot_product");
        free(result_shape);
        return NULL;
    }

    result->dimensions = result_dimensions;
    result->shape = result_shape; 
    result->strides = compute_strides(result->shape, result->dimensions);
     if (!result->strides) {
        free(result->shape); 
        free(result);
        return NULL;
    }


    int total_elements = compute_total_elements(result->shape, result->dimensions);
    result->data = malloc(total_elements * sizeof(double)); 
     if (!result->data) {
        perror("Failed to allocate memory for result data in dot_product");
        free(result->strides);
        free(result->shape);
        free(result);
        return NULL;
    }


    // Pre-allocate index arrays for efficiency
    int *result_indices = malloc(result_dimensions * sizeof(int));
    int *tensor1_indices = malloc(tensor1->dimensions * sizeof(int));
    int *tensor2_indices = malloc(tensor2->dimensions * sizeof(int));
     if (!result_indices || !tensor1_indices || !tensor2_indices) {
         perror("Failed to allocate memory for indices in dot_product");
         free(result_indices); 
         free(tensor1_indices);
         free(tensor2_indices);
         free(result->data);
         free(result->strides);
         free(result->shape);
         free(result);
         return NULL;
     }


    for (int i = 0; i < total_elements; i++) {
        // Convert flat index to multi-dimensional indices for the result shape
        int temp_flat_index = i;
        for (int j = 0; j < result_dimensions; j++) {
            if (result->strides[j] == 0) {
                 result_indices[j] = 0;
            } else {
                 result_indices[j] = temp_flat_index / result->strides[j];
                 temp_flat_index %= result->strides[j];
            }
        }

        double dot_sum = 0;
        int contracted_dim_size = tensor1->shape[tensor1->dimensions - 1];

        for (int k = 0; k < contracted_dim_size; k++) {
            // Determine indices for tensor1
            int tensor1_idx_offset = 0;
            for (int d = 0; d < tensor1->dimensions - 1; d++) {
                tensor1_indices[d] = result_indices[tensor1_idx_offset++];
            }
            tensor1_indices[tensor1->dimensions - 1] = k; // Contracted dimension

            // Determine indices for tensor2
            int tensor2_idx_offset = tensor1->dimensions - 1; // Offset in result_indices for tensor2's dimensions
             tensor2_indices[0] = k; 
            for (int d = 1; d < tensor2->dimensions; d++) {
                tensor2_indices[d] = result_indices[tensor2_idx_offset++];
            }


            dot_sum += get_tensor_value(tensor1, tensor1_indices) * get_tensor_value(tensor2, tensor2_indices);
        }

        result->data[i] = dot_sum;
    }

    free(result_indices);
    free(tensor1_indices);
    free(tensor2_indices);

    return result;
}

// Function to extract a sub-tensor
// Returns a new tensor containing the extracted sub-tensor, or NULL on failure.
Tensor *partial_tensor(const Tensor *tensor, const int *start_indices, const int *end_indices) {
    if (!tensor || !start_indices || !end_indices) {
        fprintf(stderr, "Error: Input tensor or indices are NULL for partial_tensor.\n");
        return NULL;
    }

    int *new_shape = malloc(tensor->dimensions * sizeof(int)); 
    if (!new_shape) {
        perror("Failed to allocate memory for new_shape in partial_tensor");
        return NULL;
    }

    // Compute the new shape and validate indices range
    for (int i = 0; i < tensor->dimensions; i++) {
        if (start_indices[i] < 0 || start_indices[i] >= tensor->shape[i] ||
            end_indices[i] < 0 || end_indices[i] > tensor->shape[i] || 
            start_indices[i] > end_indices[i]) {
            fprintf(stderr, "Error: Invalid start or end indices for dimension %d.\n", i);
            free(new_shape);
            return NULL;
        }
        new_shape[i] = end_indices[i] - start_indices[i];
    }

    Tensor *result = malloc(sizeof(Tensor)); 
     if (!result) {
        perror("Failed to allocate memory for result tensor in partial_tensor");
        free(new_shape);
        return NULL;
    }

    result->dimensions = tensor->dimensions;
    result->shape = new_shape; 
    result->strides = compute_strides(result->shape, result->dimensions);
     if (!result->strides) {
        free(result->shape); 
        free(result);
        return NULL;
    }


    int total_elements = compute_total_elements(result->shape, result->dimensions);
    result->data = malloc(total_elements * sizeof(double));
     if (!result->data) {
        perror("Failed to allocate memory for result data in partial_tensor");
        free(result->strides);
        free(result->shape);
        free(result);
        return NULL;
    }


    // Pre-allocate index arrays for efficiency
    int *result_indices = malloc(tensor->dimensions * sizeof(int));
    int *original_indices = malloc(tensor->dimensions * sizeof(int));
     if (!result_indices || !original_indices) {
         perror("Failed to allocate memory for indices in partial_tensor");
         free(result_indices); 
         free(original_indices); 
         free(result->data);
         free(result->strides);
         free(result->shape);
         free(result);
         return NULL;
     }


    for (int i = 0; i < total_elements; i++) {
        // Convert flat index to multi-dimensional indices for the result shape
        int temp_flat_index = i;
         for (int j = 0; j < tensor->dimensions; j++) {
            if (result->strides[j] == 0) {
                 result_indices[j] = 0;
            } else {
                 result_indices[j] = temp_flat_index / result->strides[j];
                 temp_flat_index %= result->strides[j];
            }
        }


        // Calculate original indices by adding the start offset
        for (int j = 0; j < tensor->dimensions; j++) {
            original_indices[j] = start_indices[j] + result_indices[j];
        }

        result->data[i] = get_tensor_value(tensor, original_indices); // Use get_tensor_value for bounds check

    }

    free(result_indices); 
    free(original_indices);

    return result;
}


/**
 * Prints the tensor data for debugging purposes.
 *
 * @param tensor Pointer to the tensor.
 */
void print_tensor(const Tensor *tensor) {
    if (!tensor) {
        printf("Tensor is NULL\n");
        return;
    }
    printf("Tensor(shape=[");
    for (int i = 0; i < tensor->dimensions; i++) {
        printf("%d", tensor->shape[i]);
        if (i < tensor->dimensions - 1) {
            printf(", ");
        }
    }
    printf("]):\n");

    int total_elements = compute_total_elements(tensor->shape, tensor->dimensions);
    if (total_elements == 0) {
        printf("[]\n");
        return;
    }

    // Simple linear print for C
    printf("[");
    for (int i = 0; i < total_elements; i++) {
        printf("%f", tensor->data[i]);
        if (i < total_elements - 1) {
            printf(", ");
        }
    }
    printf("]\n");
}

