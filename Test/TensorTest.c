#include <stdio.h>
#include <stdlib.h> 
#include <math.h>
#include <stdbool.h>
#include <Memory/Memory.h>
#include "../src/Tensor.h"

int compute_total_elements_helper(const int *shape, int dimensions) {
    int total_elements = 1;
    for (int i = 0; i < dimensions; i++) {
        if (shape[i] <= 0) return 0; 
        total_elements *= shape[i];
    }
    return total_elements;
}


// Helper function to compare two tensors
bool are_tensors_equal(const Tensor* t1, const Tensor* t2, double tolerance) {
    if (!t1 || !t2) return t1 == t2; 
    if (t1->dimensions != t2->dimensions) return false;
    for (int i = 0; i < t1->dimensions; i++) {
        if (t1->shape[i] != t2->shape[i]) return false;
    }

    int total_elements1 = compute_total_elements_helper(t1->shape, t1->dimensions);
    int total_elements2 = compute_total_elements_helper(t2->shape, t2->dimensions);

    if (total_elements1 != total_elements2) return false;

    for (int i = 0; i < total_elements1; i++) {
        if (fabs(t1->data[i] - t2->data[i]) > tolerance) {
            return false;
        }
    }
    return true;
}

// Helper function to print test result
void print_test_result(const char *test_name, bool passed) {
    printf("[%s] %s\n", passed ? "PASSED" : "FAILED", test_name);
}

// Helper function to convert a flat index to multi-dimensional indices based on strides.
int *unflatten_index_helper(int flat_index, const int *shape, const int *strides, int dimensions) {
    int *indices = malloc_(dimensions * sizeof(int));
    if (!indices) {
        perror("Failed to allocate memory for unflattened indices in helper");
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


// Test function for create_tensor and free_tensor
void test_create_and_free() {
    const char *test_name = "test_create_and_free";
    bool passed = true;

    // Test Case 1: 1D Tensor
    double data1[] = {1.0, 2.0, 3.0};
    int shape1[] = {3};
    Tensor_ptr t1 = create_tensor(data1, shape1, 1);
    if (!t1 || t1->dimensions != 1 || t1->shape[0] != 3 || compute_total_elements_helper(shape1, 1) != 3) {
        passed = false;
    }
    free_tensor(t1);

    // Test Case 2: 2D Tensor
    double data2[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    int shape2[] = {2, 3};
    Tensor_ptr t2 = create_tensor(data2, shape2, 2);
     if (!t2 || t2->dimensions != 2 || t2->shape[0] != 2 || t2->shape[1] != 3 || compute_total_elements_helper(shape2, 2) != 6) {
        passed = false;
    }
    free_tensor(t2);

    // Test Case 3: Empty Tensor (shape with 0)
    int shape3[] = {0, 5};
    Tensor_ptr t3 = create_tensor(NULL, shape3, 2); 
    if (!t3 || t3->dimensions != 2 || t3->shape[0] != 0 || t3->shape[1] != 5 || compute_total_elements_helper(shape3, 2) != 0) {
         passed = false;
    }
    free_tensor(t3);

    // Test Case 4: Invalid shape (NULL)
    Tensor_ptr t4 = create_tensor(data1, NULL, 1);
    if (t4 != NULL) { // Should return NULL on invalid input
        passed = false;
        free_tensor(t4);
    }

    // Test Case 5: Invalid dimensions (0)
    Tensor_ptr t5 = create_tensor(data1, shape1, 0);
    if (t5 != NULL) { // Should return NULL on invalid input
        passed = false;
        free_tensor(t5);
    }

    print_test_result(test_name, passed);
}

// Test function for get_tensor_value and set_tensor_value
void test_get_set() {
    const char *test_name = "test_get_set";
    bool passed = true;

    // Test Get/Set on a 2x3 tensor
    double data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    int shape[] = {2, 3};
    Tensor_ptr t = create_tensor(data, shape, 2);
    if (!t) {
        print_test_result(test_name, false);
        return;
    }

    // Test get_tensor_value
    int indices1[] = {0, 1}; // Row 0, Col 1 should be 2.0
    if (fabs(get_tensor_value(t, indices1) - 2.0) > 1e-10) {
        passed = false;
    }

    int indices2[] = {1, 2}; // Row 1, Col 2 should be 6.0
    if (fabs(get_tensor_value(t, indices2) - 6.0) > 1e-10) {
        passed = false;
    }

    // Test set_tensor_value
    set_tensor_value(t, indices1, 10.0);
    if (fabs(get_tensor_value(t, indices1) - 10.0) > 1e-10) {
        passed = false;
    }

    set_tensor_value(t, indices2, 20.0);
    if (fabs(get_tensor_value(t, indices2) - 20.0) > 1e-10) {
        passed = false;
    }

    free_tensor(t);
    print_test_result(test_name, passed);
}

// Test function for reshape_tensor
void test_reshape() {
    const char *test_name = "test_reshape";
    bool passed = true;

    // Create a 2x3 tensor
    double data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    int shape[] = {2, 3};
    Tensor_ptr t = create_tensor(data, shape, 2);
    if (!t) {
        print_test_result(test_name, false);
        return;
    }

    // Reshape to 1D (flatten)
    int new_shape1[] = {6};
    Tensor_ptr reshaped1 = reshape_tensor(t, new_shape1, 1);
    if (!reshaped1 || reshaped1->dimensions != 1 || reshaped1->shape[0] != 6) {
        passed = false;
    } else {
        // Check that data is the same
        for (int i = 0; i < 6; i++) {
            if (fabs(reshaped1->data[i] - data[i]) > 1e-10) {
                passed = false;
                break;
            }
        }
    }
    free_tensor(reshaped1);

    // Reshape to 2D with different shape
    int new_shape2[] = {3, 2};
    Tensor_ptr reshaped2 = reshape_tensor(t, new_shape2, 2);
    if (!reshaped2 || reshaped2->dimensions != 2 || reshaped2->shape[0] != 3 || reshaped2->shape[1] != 2) {
        passed = false;
    } else {
        // Check specific values in the new shape
        int indices1[] = {0, 0}; 
        int indices2[] = {2, 1}; 
        if (fabs(get_tensor_value(reshaped2, indices1) - 1.0) > 1e-10 ||
            fabs(get_tensor_value(reshaped2, indices2) - 6.0) > 1e-10) {
            passed = false;
        }
    }
    free_tensor(reshaped2);

    // Test invalid reshape (different number of elements)
    int new_shape3[] = {5};
    Tensor_ptr reshaped3 = reshape_tensor(t, new_shape3, 1);
    if (reshaped3 != NULL) { 
        passed = false;
        free_tensor(reshaped3);
    }

    // Test NULL tensor
    int new_shape4[] = {6};
    Tensor_ptr reshaped4 = reshape_tensor(NULL, new_shape4, 1);
    if (reshaped4 != NULL) {
        passed = false;
        free_tensor(reshaped4);
    }

    free_tensor(t);
    print_test_result(test_name, passed);
}

// Test function for transpose_tensor
void test_transpose() {
    const char *test_name = "test_transpose";
    bool passed = true;

    // Test Case 1: 2D Tensor with NULL axes (default reverse)
    double data1[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    int shape1[] = {2, 3}; // 2 rows, 3 columns
    Tensor_ptr t1 = create_tensor(data1, shape1, 2);
    Tensor_ptr transposed1 = transpose_tensor(t1, NULL); // Reverse axes (0, 1) -> (1, 0)
    if (!transposed1 || transposed1->dimensions != 2 || transposed1->shape[0] != 3 || transposed1->shape[1] != 2) {
        passed = false;
    } else {
        
        int indices1[] = {0, 0}; // Should be data1[0] = 1.0
        int indices2[] = {0, 1}; // Should be data1[3] = 4.0
        if (fabs(get_tensor_value(transposed1, indices1) - 1.0) > 1e-10 ||
            fabs(get_tensor_value(transposed1, indices2) - 4.0) > 1e-10) {
            passed = false;
        }
    }
    free_tensor(transposed1);
    free_tensor(t1);

    double data2[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    int shape2[] = {2, 2, 2}; // (depth, row, col)
    Tensor_ptr t2 = create_tensor(data2, shape2, 3);
    int axes2[] = {2, 0, 1}; // Permute axes: (depth, row, col) -> (col, depth, row)
    Tensor_ptr transposed2 = transpose_tensor(t2, axes2); // New shape {2, 2, 2}
    if (!transposed2) {
        passed = false;
    } else {
        // We're permuting axes, but the shape is still {2,2,2}
        // Check dimensions and shape
        if (transposed2->dimensions != 3 || transposed2->shape[0] != 2 || 
            transposed2->shape[1] != 2 || transposed2->shape[2] != 2) {
            passed = false;
        }
    }
    free_tensor(transposed2);
    free_tensor(t2);

    // Test Case 3: Invalid axes (out of bounds)
    Tensor_ptr t3 = create_tensor(data1, shape1, 2);
    int invalid_axes[] = {0, 2}; // 2 is out of bounds for a 2D tensor
    Tensor_ptr transposed3 = transpose_tensor(t3, invalid_axes);
    if (transposed3 != NULL) { // Should be NULL as axes are invalid
        passed = false;
        free_tensor(transposed3);
    }
    free_tensor(t3);

    print_test_result(test_name, passed);
}

// Test function for add_tensors
void test_add() {
    const char *test_name = "test_add";
    bool passed = true;

    // Test Case 1: Same shape tensors
    double data1_1[] = {1.0, 2.0, 3.0, 4.0};
    double data1_2[] = {5.0, 6.0, 7.0, 8.0};
    int shape1[] = {2, 2};
    Tensor_ptr t1_1 = create_tensor(data1_1, shape1, 2);
    Tensor_ptr t1_2 = create_tensor(data1_2, shape1, 2);
    Tensor_ptr result1 = add_tensors(t1_1, t1_2);
    
    if (!result1 || result1->dimensions != 2 || result1->shape[0] != 2 || result1->shape[1] != 2) {
        passed = false;
    } else {
        // Check result values
        int indices1[] = {0, 0};  
        int indices2[] = {1, 1}; 
        if (fabs(get_tensor_value(result1, indices1) - 6.0) > 1e-10 ||
            fabs(get_tensor_value(result1, indices2) - 12.0) > 1e-10) {
            passed = false;
        }
    }
    free_tensor(result1);
    free_tensor(t1_1);
    free_tensor(t1_2);

    // Test Case 2: Broadcasting scalar to tensor
    double data2_1[] = {1.0};
    double data2_2[] = {5.0, 6.0, 7.0, 8.0};
    int shape2_1[] = {1};
    int shape2_2[] = {2, 2};
    Tensor_ptr t2_1 = create_tensor(data2_1, shape2_1, 1);
    Tensor_ptr t2_2 = create_tensor(data2_2, shape2_2, 2);
    Tensor_ptr result2 = add_tensors(t2_1, t2_2);
    
    if (!result2 || result2->dimensions != 2 || result2->shape[0] != 2 || result2->shape[1] != 2) {
        passed = false;
    } else {
        // Check result values: scalar broadcasts
        int indices1[] = {0, 0}; 
        int indices2[] = {1, 1}; 
        if (fabs(get_tensor_value(result2, indices1) - 6.0) > 1e-10 ||
            fabs(get_tensor_value(result2, indices2) - 9.0) > 1e-10) {
            passed = false;
        }
    }
    free_tensor(result2);
    free_tensor(t2_1);
    free_tensor(t2_2);

    // Test Case 3: Non-broadcastable shapes
    double data3_1[] = {1.0, 2.0, 3.0};
    double data3_2[] = {5.0, 6.0, 7.0, 8.0};
    int shape3_1[] = {3};
    int shape3_2[] = {2, 2};
    Tensor_ptr t3_1 = create_tensor(data3_1, shape3_1, 1);
    Tensor_ptr t3_2 = create_tensor(data3_2, shape3_2, 2);
    Tensor_ptr result3 = add_tensors(t3_1, t3_2);
    
    if (result3 != NULL) { // Should be NULL as shapes aren't broadcastable
        passed = false;
        free_tensor(result3);
    }
    free_tensor(t3_1);
    free_tensor(t3_2);

    print_test_result(test_name, passed);
}

// Test function for subtract_tensors
void test_subtract() {
    const char *test_name = "test_subtract";
    bool passed = true;

    // Test Case 1: Same shape tensors
    double data1_1[] = {5.0, 6.0, 7.0, 8.0};
    double data1_2[] = {1.0, 2.0, 3.0, 4.0};
    int shape1[] = {2, 2};
    Tensor_ptr t1_1 = create_tensor(data1_1, shape1, 2);
    Tensor_ptr t1_2 = create_tensor(data1_2, shape1, 2);
    Tensor_ptr result1 = subtract_tensors(t1_1, t1_2);
    
    if (!result1 || result1->dimensions != 2 || result1->shape[0] != 2 || result1->shape[1] != 2) {
        passed = false;
    } else {
        // Check result values
        int indices1[] = {0, 0};  
        int indices2[] = {1, 1};  
        if (fabs(get_tensor_value(result1, indices1) - 4.0) > 1e-10 ||
            fabs(get_tensor_value(result1, indices2) - 4.0) > 1e-10) {
            passed = false;
        }
    }
    free_tensor(result1);
    free_tensor(t1_1);
    free_tensor(t1_2);

    // Additional test cases similar to add_tensors...
    // (For brevity, broadcasting and invalid cases tests are omitted)

    print_test_result(test_name, passed);
}

// Test function for multiply_tensors
void test_multiply() {
    const char *test_name = "test_multiply";
    bool passed = true;

    // Test Case 1: Same shape tensors
    double data1_1[] = {1.0, 2.0, 3.0, 4.0};
    double data1_2[] = {5.0, 6.0, 7.0, 8.0};
    int shape1[] = {2, 2};
    Tensor_ptr t1_1 = create_tensor(data1_1, shape1, 2);
    Tensor_ptr t1_2 = create_tensor(data1_2, shape1, 2);
    Tensor_ptr result1 = hadamard_product(t1_1, t1_2);
    
    if (!result1 || result1->dimensions != 2 || result1->shape[0] != 2 || result1->shape[1] != 2) {
        passed = false;
    } else {
        // Check result values (element-wise multiplication)
        int indices1[] = {0, 0};  
        int indices2[] = {1, 1};  
        if (fabs(get_tensor_value(result1, indices1) - 5.0) > 1e-10 ||
            fabs(get_tensor_value(result1, indices2) - 32.0) > 1e-10) {
            passed = false;
        }
    }
    free_tensor(result1);
    free_tensor(t1_1);
    free_tensor(t1_2);

    // Additional test cases similar to add_tensors...
    // (For brevity, broadcasting and invalid cases tests are omitted)

    print_test_result(test_name, passed);
}

// Test function for dot_product
void test_dot_product() {
    const char *test_name = "test_dot_product";
    bool passed = true;

    // Test Case 1: Matrix multiplication (2x3 . 3x2)
    double data1_1[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}; 
    double data1_2[] = {7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
    int shape1_1[] = {2, 3};
    int shape1_2[] = {3, 2};
    Tensor_ptr t1_1 = create_tensor(data1_1, shape1_1, 2);
    Tensor_ptr t1_2 = create_tensor(data1_2, shape1_2, 2);
    Tensor_ptr result1 = multiply_tensors(t1_1, t1_2);
    
    if (!result1 || result1->dimensions != 2 || result1->shape[0] != 2 || result1->shape[1] != 2) {
        passed = false;
    } else {
        int indices1[] = {0, 0};
        int indices2[] = {0, 1};
        int indices3[] = {1, 0};
        int indices4[] = {1, 1};
        if (fabs(get_tensor_value(result1, indices1) - 58.0) > 1e-10 ||
            fabs(get_tensor_value(result1, indices2) - 64.0) > 1e-10 ||
            fabs(get_tensor_value(result1, indices3) - 139.0) > 1e-10 ||
            fabs(get_tensor_value(result1, indices4) - 154.0) > 1e-10) {
            passed = false;
        }
    }
    free_tensor(result1);
    free_tensor(t1_1);
    free_tensor(t1_2);

    // Test Case 2: Incompatible shapes for dot product
    double data2_1[] = {1.0, 2.0, 3.0};
    double data2_2[] = {4.0, 5.0, 6.0, 7.0};
    int shape2_1[] = {3};
    int shape2_2[] = {2, 2};
    Tensor_ptr t2_1 = create_tensor(data2_1, shape2_1, 1);
    Tensor_ptr t2_2 = create_tensor(data2_2, shape2_2, 2);
    Tensor_ptr result2 = multiply_tensors(t2_1, t2_2);
    
    if (result2 != NULL) { // Should be NULL as shapes aren't compatible for dot product
        passed = false;
        free_tensor(result2);
    }
    free_tensor(t2_1);
    free_tensor(t2_2);

    print_test_result(test_name, passed);
}

// Test function for partial_tensor (slicing)
void test_partial() {
    const char *test_name = "test_partial";
    bool passed = true;

    // Test Case 1: Extract a submatrix from a 3x3 matrix
    double data1[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0}; // 3x3 matrix
    int shape1[] = {3, 3};
    Tensor_ptr t1 = create_tensor(data1, shape1, 2);
    
    int start_indices1[] = {0, 0};
    int end_indices1[] = {2, 2}; // Extract top-left 2x2 submatrix
    
    Tensor_ptr result1 = partial_tensor(t1, start_indices1, end_indices1);
    if (!result1 || result1->dimensions != 2 || result1->shape[0] != 2 || result1->shape[1] != 2) {
        passed = false;
    } else {
        // Check values: [[1, 2], [4, 5]]
        int indices1[] = {0, 0};
        int indices2[] = {1, 1};
        if (fabs(get_tensor_value(result1, indices1) - 1.0) > 1e-10 ||
            fabs(get_tensor_value(result1, indices2) - 5.0) > 1e-10) {
            passed = false;
        }
    }
    free_tensor(result1);
    
    // Test Case 2: Invalid indices
    int start_indices2[] = {1, 1};
    int end_indices2[] = {4, 2}; // 4 is out of bounds
    
    Tensor_ptr result2 = partial_tensor(t1, start_indices2, end_indices2);
    if (result2 != NULL) { // Should be NULL as indices are invalid
        passed = false;
        free_tensor(result2);
    }
    
    free_tensor(t1);
    
    print_test_result(test_name, passed);
}

// Main test function
int main() {
    start_medium_memory_check();
    printf("Starting Tensor C Tests (Custom Memory)...\\n");
    
    test_create_and_free();
    test_get_set();
    test_reshape();
    test_transpose();
    test_add();
    test_subtract();
    test_multiply();
    test_dot_product();
    test_partial();
    
    printf("Tensor C Tests (Custom Memory) Finished.\\n");
    end_memory_check();
}
