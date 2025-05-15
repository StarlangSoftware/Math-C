#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <Memory/Memory.h>

#include "../src/Tensor.h"

// Helper function to compute the total number of elements in a tensor
// Replicating from tensor.c as it's likely static there.
int compute_total_elements_helper(const int *shape, int dimensions) {
    int total_elements = 1;
    for (int i = 0; i < dimensions; i++) {
        if (shape[i] <= 0) return 0;
        total_elements *= shape[i];
    }
    return total_elements;
}


// Helper function to compare two tensors
bool are_tensors_equal(const Tensor *t1, const Tensor *t2, double tolerance) {
    if (!t1 || !t2) return t1 == t2; // Both NULL is true, one NULL is false
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
    int *indices = malloc_(dimensions * sizeof(int), "unflatten_index_helper"); // Use standard malloc
    if (!indices) {
        perror("Failed to allocate memory for unflattened indices in helper");
        exit(EXIT_FAILURE);
    }
    int temp_flat_index = flat_index;
    for (int i = 0; i < dimensions; i++) {
        if (strides[i] == 0) {
            // Avoid division by zero if a dimension is 0
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
    Tensor *t1 = create_tensor(data1, shape1, 1);
    if (!t1 || t1->dimensions != 1 || t1->shape[0] != 3 || compute_total_elements_helper(shape1, 1) != 3) {
        passed = false;
    }
    free_tensor(t1);

    // Test Case 2: 2D Tensor
    double data2[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    int shape2[] = {2, 3};
    Tensor *t2 = create_tensor(data2, shape2, 2);
    if (!t2 || t2->dimensions != 2 || t2->shape[0] != 2 || t2->shape[1] != 3 || compute_total_elements_helper(shape2, 2)
        != 6) {
        passed = false;
    }
    free_tensor(t2);

    // Test Case 3: Empty Tensor (shape with 0)
    int shape3[] = {0, 5};
    Tensor *t3 = create_tensor(NULL, shape3, 2); // Data should be NULL for 0 elements
    if (!t3 || t3->dimensions != 2 || t3->shape[0] != 0 || t3->shape[1] != 5 || compute_total_elements_helper(shape3, 2)
        != 0) {
        passed = false;
    }
    free_tensor(t3);

    // Test Case 4: Invalid shape (NULL)
    Tensor *t4 = create_tensor(data1, NULL, 1);
    if (t4 != NULL) {
        // Should return NULL on invalid input
        passed = false;
        free_tensor(t4);
    }

    // Test Case 5: Invalid dimensions (<= 0)
    Tensor *t5 = create_tensor(data1, shape1, 0);
    if (t5 != NULL) {
        // Should return NULL
        passed = false;
        free_tensor(t5);
    }


    print_test_result(test_name, passed);
}

// Test function for get_tensor_value and set_tensor_value
void test_get_set() {
    const char *test_name = "test_get_set";
    bool passed = true;

    double data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    int shape[] = {2, 3};
    Tensor *t = create_tensor(data, shape, 2);

    // Test Case 1: Get values
    int indices1[] = {0, 0};
    if (fabs(get_tensor_value(t, indices1) - 1.0) > 1e-9) passed = false;
    int indices2[] = {0, 2};
    if (fabs(get_tensor_value(t, indices2) - 3.0) > 1e-9) passed = false;
    int indices3[] = {1, 1};
    if (fabs(get_tensor_value(t, indices3) - 5.0) > 1e-9) passed = false;

    // Test Case 2: Set values
    int indices4[] = {0, 1};
    set_tensor_value(t, indices4, 99.0);
    if (fabs(get_tensor_value(t, indices4) - 99.0) > 1e-9) passed = false;
    int indices5[] = {1, 2};
    set_tensor_value(t, indices5, -5.5);
    if (fabs(get_tensor_value(t, indices5) - -5.5) > 1e-9) passed = false;

    free_tensor(t);

    print_test_result(test_name, passed);
}

// Test function for reshape_tensor
void test_reshape() {
    const char *test_name = "test_reshape";
    bool passed = true;

    double data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    int shape[] = {2, 3};
    Tensor *t = create_tensor(data, shape, 2);

    // Test Case 1: Reshape to different shape with same element count
    int new_shape1[] = {6};
    Tensor *reshaped1 = reshape_tensor(t, new_shape1, 1);
    if (!reshaped1 || reshaped1->dimensions != 1 || reshaped1->shape[0] != 6 ||
        compute_total_elements_helper(new_shape1, 1) != 6) {
        passed = false;
    } else {
        // Check data integrity (should be the same flattened data)
        for (int i = 0; i < 6; ++i) {
            int idx[] = {i};
            if (fabs(get_tensor_value(reshaped1, idx) - data[i]) > 1e-9) passed = false;
        }
    }
    free_tensor(reshaped1);

    // Test Case 2: Reshape to another valid shape
    int new_shape2[] = {3, 2};
    Tensor *reshaped2 = reshape_tensor(t, new_shape2, 2);
    if (!reshaped2 || reshaped2->dimensions != 2 || reshaped2->shape[0] != 3 || reshaped2->shape[1] != 2 ||
        compute_total_elements_helper(new_shape2, 2) != 6) {
        passed = false;
    } else {
        // Check data integrity
        double expected_data2[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}; // Flattened data remains the same
        for (int i = 0; i < 6; ++i) {
            int idx_flat[] = {i};
            // Using the helper function defined in the test file
            int *idx_multi = unflatten_index_helper(i, reshaped2->shape, reshaped2->strides, reshaped2->dimensions);
            if (fabs(get_tensor_value(reshaped2, idx_multi) - expected_data2[i]) > 1e-9) passed = false;
            free(idx_multi);
        }
    }
    free_tensor(reshaped2);


    // Test Case 3: Reshape with different element count (should return NULL)
    int new_shape3[] = {5};
    Tensor *reshaped3 = reshape_tensor(t, new_shape3, 1);
    if (reshaped3 != NULL) {
        passed = false;
        free_tensor(reshaped3);
    }

    // Test Case 4: Reshape to empty shape
    int new_shape4[] = {0};
    Tensor *reshaped4 = reshape_tensor(t, new_shape4, 1);
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

    // Test Case 1: 2D Transpose (reverse axes)
    double data1[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}; // Shape {2, 3}
    int shape1[] = {2, 3};
    Tensor *t1 = create_tensor(data1, shape1, 2);
    Tensor *transposed1 = transpose_tensor(t1, NULL); // Reverse axes (0, 1) -> (1, 0)

    if (!transposed1 || transposed1->dimensions != 2 || transposed1->shape[0] != 3 || transposed1->shape[1] != 2) {
        passed = false;
    } else {
        double expected_data1[] = {1.0, 4.0, 2.0, 5.0, 3.0, 6.0};
        int total_elements = compute_total_elements_helper(transposed1->shape, transposed1->dimensions);
        for (int i = 0; i < total_elements; i++) {
            if (fabs(transposed1->data[i] - expected_data1[i]) > 1e-9) passed = false;
        }
    }
    free_tensor(t1);
    free_tensor(transposed1);

    // Test Case 2: 3D Transpose with specified axes
    double data2[] = {1, 2, 3, 4, 5, 6, 7, 8}; // Shape {2, 2, 2}
    int shape2[] = {2, 2, 2};
    Tensor *t2 = create_tensor(data2, shape2, 3);
    int axes2[] = {2, 0, 1}; // Transpose axes (0, 1, 2) -> (2, 0, 1)
    Tensor *transposed2 = transpose_tensor(t2, axes2); // New shape {2, 2, 2}

    if (!transposed2 || transposed2->dimensions != 3 || transposed2->shape[0] != 2 || transposed2->shape[1] != 2 ||
        transposed2->shape[2] != 2) {
        passed = false;
    } else {
        // Original indices (i, j, k) map to new indices (k, i, j)
        double expected_data2[] = {1, 3, 5, 7, 2, 4, 6, 8};
        int total_elements = compute_total_elements_helper(transposed2->shape, transposed2->dimensions);
        for (int i = 0; i < total_elements; i++) {
            if (fabs(transposed2->data[i] - expected_data2[i]) > 1e-9) passed = false;
        }
    }
    free_tensor(t2);
    free_tensor(transposed2);

    // Test Case 3: Transpose with invalid axes (should return NULL)
    int invalid_axes[] = {0, 2};
    Tensor *t3 = create_tensor(data1, shape1, 2);
    Tensor *transposed3 = transpose_tensor(t3, invalid_axes);
    if (transposed3 != NULL) {
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

    // Test Case 1: Element-wise addition (same shape)
    double data1_1[] = {1.0, 2.0, 3.0, 4.0};
    int shape1[] = {2, 2};
    Tensor *t1_1 = create_tensor(data1_1, shape1, 2);
    double data1_2[] = {5.0, 6.0, 7.0, 8.0};
    Tensor *t1_2 = create_tensor(data1_2, shape1, 2);
    Tensor *result1 = add_tensors(t1_1, t1_2);

    double expected_data1[] = {6.0, 8.0, 10.0, 12.0};
    int expected_shape1[] = {2, 2};
    Tensor *expected1 = create_tensor(expected_data1, expected_shape1, 2);

    if (!are_tensors_equal(result1, expected1, 1e-9)) passed = false;

    free_tensor(t1_1);
    free_tensor(t1_2);
    free_tensor(result1);
    free_tensor(expected1);

    // Test Case 2: Broadcasting a scalar
    double data2_1[] = {1.0, 2.0, 3.0, 4.0};
    int shape2_1[] = {2, 2};
    Tensor *t2_1 = create_tensor(data2_1, shape2_1, 2);
    double data2_2[] = {10.0};
    int shape2_2[] = {1};
    Tensor *t2_2 = create_tensor(data2_2, shape2_2, 1);
    Tensor *result2 = add_tensors(t2_1, t2_2);

    double expected_data2[] = {11.0, 12.0, 13.0, 14.0};
    int expected_shape2[] = {2, 2};
    Tensor *expected2 = create_tensor(expected_data2, expected_shape2, 2);

    if (!are_tensors_equal(result2, expected2, 1e-9)) passed = false;

    free_tensor(t2_1);
    free_tensor(t2_2);
    free_tensor(result2);
    free_tensor(expected2);

    // Test Case 3: Broadcasting a row vector
    double data3_1[] = {1.0, 2.0, 3.0, 4.0};
    int shape3_1[] = {2, 2};
    Tensor *t3_1 = create_tensor(data3_1, shape3_1, 2);
    double data3_2[] = {10.0, 20.0};
    int shape3_2[] = {1, 2};
    Tensor *t3_2 = create_tensor(data3_2, shape3_2, 2);
    Tensor *result3 = add_tensors(t3_1, t3_2);

    double expected_data3[] = {11.0, 22.0, 13.0, 24.0};
    int expected_shape3[] = {2, 2};
    Tensor *expected3 = create_tensor(expected_data3, expected_shape3, 2);

    if (!are_tensors_equal(result3, expected3, 1e-9)) passed = false;

    free_tensor(t3_1);
    free_tensor(t3_2);
    free_tensor(result3);
    free_tensor(expected3);

    // Test Case 4: Broadcasting a column vector
    double data4_1[] = {1.0, 2.0, 3.0, 4.0};
    int shape4_1[] = {2, 2};
    Tensor *t4_1 = create_tensor(data4_1, shape4_1, 2);
    double data4_2[] = {10.0, 20.0};
    int shape4_2[] = {2, 1};
    Tensor *t4_2 = create_tensor(data4_2, shape4_2, 2);
    Tensor *result4 = add_tensors(t4_1, t4_2);

    double expected_data4[] = {11.0, 12.0, 23.0, 24.0};
    int expected_shape4[] = {2, 2};
    Tensor *expected4 = create_tensor(expected_data4, expected_shape4, 2);

    if (!are_tensors_equal(result4, expected4, 1e-9)) passed = false;

    free_tensor(t4_1);
    free_tensor(t4_2);
    free_tensor(result4);
    free_tensor(expected4);


    // Test Case 5: Non-broadcastable shapes (should return NULL)
    double data5_1[] = {1.0, 2.0};
    int shape5_1[] = {2};
    Tensor *t5_1 = create_tensor(data5_1, shape5_1, 1);
    double data5_2[] = {10.0, 20.0, 30.0};
    int shape5_2[] = {3};
    Tensor *t5_2 = create_tensor(data5_2, shape5_2, 1);
    Tensor *result5 = add_tensors(t5_1, t5_2);
    if (result5 != NULL) {
        passed = false;
        free_tensor(result5);
    }
    free_tensor(t5_1);
    free_tensor(t5_2);


    print_test_result(test_name, passed);
}

// Test function for subtract_tensors
void test_subtract() {
    const char *test_name = "test_subtract";
    bool passed = true;

    // Test Case 1: Element-wise subtraction (same shape)
    double data1_1[] = {10.0, 20.0, 30.0, 40.0};
    int shape1[] = {2, 2};
    Tensor *t1_1 = create_tensor(data1_1, shape1, 2);
    double data1_2[] = {1.0, 2.0, 3.0, 4.0};
    Tensor *t1_2 = create_tensor(data1_2, shape1, 2);
    Tensor *result1 = subtract_tensors(t1_1, t1_2);

    double expected_data1[] = {9.0, 18.0, 27.0, 36.0};
    int expected_shape1[] = {2, 2};
    Tensor *expected1 = create_tensor(expected_data1, expected_shape1, 2);

    if (!are_tensors_equal(result1, expected1, 1e-9)) passed = false;

    free_tensor(t1_1);
    free_tensor(t1_2);
    free_tensor(result1);
    free_tensor(expected1);

    // Test Case 2: Broadcasting a scalar
    double data2_1[] = {10.0, 20.0, 30.0, 40.0};
    int shape2_1[] = {2, 2};
    Tensor *t2_1 = create_tensor(data2_1, shape2_1, 2);
    double data2_2[] = {5.0};
    int shape2_2[] = {1}; // Scalar
    Tensor *t2_2 = create_tensor(data2_2, shape2_2, 1);
    Tensor *result2 = subtract_tensors(t2_1, t2_2);

    double expected_data2[] = {5.0, 15.0, 25.0, 35.0};
    int expected_shape2[] = {2, 2};
    Tensor *expected2 = create_tensor(expected_data2, expected_shape2, 2);

    if (!are_tensors_equal(result2, expected2, 1e-9)) passed = false;

    free_tensor(t2_1);
    free_tensor(t2_2);
    free_tensor(result2);
    free_tensor(expected2);

    // Test Case 3: Broadcasting a row vector
    double data3_1[] = {10.0, 20.0, 30.0, 40.0};
    int shape3_1[] = {2, 2};
    Tensor *t3_1 = create_tensor(data3_1, shape3_1, 2);
    double data3_2[] = {1.0, 2.0};
    int shape3_2[] = {1, 2};
    Tensor *t3_2 = create_tensor(data3_2, shape3_2, 2);
    Tensor *result3 = subtract_tensors(t3_1, t3_2);

    double expected_data3[] = {9.0, 18.0, 29.0, 38.0};
    int expected_shape3[] = {2, 2};
    Tensor *expected3 = create_tensor(expected_data3, expected_shape3, 2);

    if (!are_tensors_equal(result3, expected3, 1e-9)) passed = false;

    free_tensor(t3_1);
    free_tensor(t3_2);
    free_tensor(result3);
    free_tensor(expected3);

    // Test Case 4: Broadcasting a column vector
    double data4_1[] = {10.0, 20.0, 30.0, 40.0};
    int shape4_1[] = {2, 2};
    Tensor *t4_1 = create_tensor(data4_1, shape4_1, 2);
    double data4_2[] = {1.0, 2.0};
    int shape4_2[] = {2, 1};
    Tensor *t4_2 = create_tensor(data4_2, shape4_2, 2);
    Tensor *result4 = subtract_tensors(t4_1, t4_2);

    double expected_data4[] = {9.0, 19.0, 28.0, 38.0};
    int expected_shape4[] = {2, 2};
    Tensor *expected4 = create_tensor(expected_data4, expected_shape4, 2);

    if (!are_tensors_equal(result4, expected4, 1e-9)) passed = false;

    free_tensor(t4_1);
    free_tensor(t4_2);
    free_tensor(result4);
    free_tensor(expected4);


    print_test_result(test_name, passed);
}

// Test function for multiply_tensors
void test_multiply() {
    const char *test_name = "test_multiply";
    bool passed = true;

    // Test Case 1: Element-wise multiplication (same shape)
    double data1_1[] = {1.0, 2.0, 3.0, 4.0};
    int shape1[] = {2, 2};
    Tensor *t1_1 = create_tensor(data1_1, shape1, 2);
    double data1_2[] = {5.0, 6.0, 7.0, 8.0};
    Tensor *t1_2 = create_tensor(data1_2, shape1, 2);
    Tensor *result1 = multiply_tensors(t1_1, t1_2);

    double expected_data1[] = {5.0, 12.0, 21.0, 32.0};
    int expected_shape1[] = {2, 2};
    Tensor *expected1 = create_tensor(expected_data1, expected_shape1, 2);

    if (!are_tensors_equal(result1, expected1, 1e-9)) passed = false;

    free_tensor(t1_1);
    free_tensor(t1_2);
    free_tensor(result1);
    free_tensor(expected1);

    // Test Case 2: Broadcasting a scalar
    double data2_1[] = {1.0, 2.0, 3.0, 4.0};
    int shape2_1[] = {2, 2};
    Tensor *t2_1 = create_tensor(data2_1, shape2_1, 2);
    double data2_2[] = {10.0};
    int shape2_2[] = {1};
    Tensor *t2_2 = create_tensor(data2_2, shape2_2, 1);
    Tensor *result2 = multiply_tensors(t2_1, t2_2);

    double expected_data2[] = {10.0, 20.0, 30.0, 40.0};
    int expected_shape2[] = {2, 2};
    Tensor *expected2 = create_tensor(expected_data2, expected_shape2, 2);

    if (!are_tensors_equal(result2, expected2, 1e-9)) passed = false;

    free_tensor(t2_1);
    free_tensor(t2_2);
    free_tensor(result2);
    free_tensor(expected2);

    // Test Case 3: Broadcasting a row vector
    double data3_1[] = {1.0, 2.0, 3.0, 4.0};
    int shape3_1[] = {2, 2};
    Tensor *t3_1 = create_tensor(data3_1, shape3_1, 2);
    double data3_2[] = {10.0, 20.0};
    int shape3_2[] = {1, 2};
    Tensor *t3_2 = create_tensor(data3_2, shape3_2, 2);
    Tensor *result3 = multiply_tensors(t3_1, t3_2);

    double expected_data3[] = {10.0, 40.0, 30.0, 80.0};
    int expected_shape3[] = {2, 2};
    Tensor *expected3 = create_tensor(expected_data3, expected_shape3, 2);

    if (!are_tensors_equal(result3, expected3, 1e-9)) passed = false;

    free_tensor(t3_1);
    free_tensor(t3_2);
    free_tensor(result3);
    free_tensor(expected3);

    // Test Case 4: Broadcasting a column vector
    double data4_1[] = {1.0, 2.0, 3.0, 4.0};
    int shape4_1[] = {2, 2};
    Tensor *t4_1 = create_tensor(data4_1, shape4_1, 2);
    double data4_2[] = {10.0, 20.0};
    int shape4_2[] = {2, 1};
    Tensor *t4_2 = create_tensor(data4_2, shape4_2, 2);
    Tensor *result4 = multiply_tensors(t4_1, t4_2);

    double expected_data4[] = {10.0, 20.0, 60.0, 80.0};
    int expected_shape4[] = {2, 2};
    Tensor *expected4 = create_tensor(expected_data4, expected_shape4, 2);

    if (!are_tensors_equal(result4, expected4, 1e-9)) passed = false;

    free_tensor(t4_1);
    free_tensor(t4_2);
    free_tensor(result4);
    free_tensor(expected4);


    print_test_result(test_name, passed);
}


// Test function for dot_product
void test_dot_product() {
    const char *test_name = "test_dot_product";
    bool passed = true;

    // Test Case 1: Matrix multiplication (2x3 * 3x2)
    double data1_1[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    int shape1_1[] = {2, 3};
    Tensor *t1_1 = create_tensor(data1_1, shape1_1, 2);
    double data1_2[] = {7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
    int shape1_2[] = {3, 2};
    Tensor *t1_2 = create_tensor(data1_2, shape1_2, 2);
    Tensor *result1 = dot_product_tensor(t1_1, t1_2);

    // Expected result:
    // [[1*7 + 2*9 + 3*11, 1*8 + 2*10 + 3*12],
    //  [4*7 + 5*9 + 6*11, 4*8 + 5*10 + 6*12]]
    // [[7 + 18 + 33, 8 + 20 + 36],
    //  [28 + 45 + 66, 32 + 50 + 72]]
    // [[58, 64],
    //  [139, 154]]

    double expected_data1[] = {58.0, 64.0, 139.0, 154.0};
    int expected_shape1[] = {2, 2};
    Tensor *expected1 = create_tensor(expected_data1, expected_shape1, 2);

    if (!are_tensors_equal(result1, expected1, 1e-9)) passed = false;

    free_tensor(t1_1);
    free_tensor(t1_2);
    free_tensor(result1);
    free_tensor(expected1);

    // Test Case 2: Vector dot product (1x3 * 3x1)
    double data2_1[] = {1.0, 2.0, 3.0};
    int shape2_1[] = {1, 3};
    Tensor *t2_1 = create_tensor(data2_1, shape2_1, 2);
    double data2_2[] = {4.0, 5.0, 6.0};
    int shape2_2[] = {3, 1};
    Tensor *t2_2 = create_tensor(data2_2, shape2_2, 2);
    Tensor *result2 = dot_product_tensor(t2_1, t2_2);

    // Expected result: [[1*4 + 2*5 + 3*6]] = [[4 + 10 + 18]] = [[32]]
    double expected_data2[] = {32.0};
    int expected_shape2[] = {1, 1};
    Tensor *expected2 = create_tensor(expected_data2, expected_shape2, 2);

    if (!are_tensors_equal(result2, expected2, 1e-9)) passed = false;

    free_tensor(t2_1);
    free_tensor(t2_2);
    free_tensor(result2);
    free_tensor(expected2);

    // Test Case 3: Matrix-vector multiplication (2x3 * 3)
    double data3_1[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}; // Shape {2, 3}
    int shape3_1[] = {2, 3};
    Tensor *t3_1 = create_tensor(data3_1, shape3_1, 2);
    double data3_2_adj[] = {7.0, 8.0, 9.0}; // Data for {3, 1}
    int shape3_2_adj[] = {3, 1};
    Tensor *t3_2_adj = create_tensor(data3_2_adj, shape3_2_adj, 2); // Shape {3, 1}
    Tensor *result3 = dot_product_tensor(t3_1, t3_2_adj); // Expected shape {2, 1}

    // Expected result:
    // [[1*7 + 2*8 + 3*9],
    //  [4*7 + 5*8 + 6*9]]
    // [[7 + 16 + 27],
    //  [28 + 40 + 54]]
    // [[50],
    //  [122]]
    double expected_data3[] = {50.0, 122.0};
    int expected_shape3[] = {2, 1};
    Tensor *expected3 = create_tensor(expected_data3, expected_shape3, 2);

    if (!are_tensors_equal(result3, expected3, 1e-9)) passed = false;

    free_tensor(t3_1);
    free_tensor(t3_2_adj);
    free_tensor(result3);
    free_tensor(expected3);


    // Test Case 4: Non-aligned shapes (should return NULL)
    double data4_1_bad[] = {1, 2, 3, 4, 5, 6};
    int shape4_1_bad[] = {2, 3};
    Tensor *t4_1_bad = create_tensor(data4_1_bad, shape4_1_bad, 2);
    double data4_2_bad[] = {1, 2, 3, 4, 5, 6, 7, 8};
    int shape4_2_bad[] = {4, 2};
    Tensor *t4_2_bad = create_tensor(data4_2_bad, shape4_2_bad, 2);

    Tensor *result4 = dot_product_tensor(t4_1_bad, t4_2_bad);
    if (result4 != NULL) {
        passed = false;
        free_tensor(result4);
    }
    free_tensor(t4_1_bad);
    free_tensor(t4_2_bad);


    print_test_result(test_name, passed);
}

// Test function for partial_tensor
void test_partial() {
    const char *test_name = "test_partial";
    bool passed = true;

    // Test Case 1: Extract a sub-matrix from a 3x3 matrix
    double data1[] = {
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0
    }; // Shape {3, 3}
    int shape1[] = {3, 3};
    Tensor *t1 = create_tensor(data1, shape1, 2);
    int start_indices1[] = {1, 1};
    int end_indices1[] = {3, 3};
    Tensor *partial1 = partial_tensor(t1, start_indices1, end_indices1);

    double expected_data1[] = {5.0, 6.0, 8.0, 9.0};
    int expected_shape1[] = {2, 2};
    Tensor *expected1 = create_tensor(expected_data1, expected_shape1, 2);

    if (!are_tensors_equal(partial1, expected1, 1e-9)) passed = false;

    free_tensor(t1);
    free_tensor(partial1);
    free_tensor(expected1);

    // Test Case 2: Extract a row vector from a matrix
    double data2[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    int shape2[] = {2, 3};
    Tensor *t2 = create_tensor(data2, shape2, 2);
    int start_indices2[] = {0, 0};
    int end_indices2[] = {1, 3};
    Tensor *partial2 = partial_tensor(t2, start_indices2, end_indices2);

    double expected_data2[] = {1.0, 2.0, 3.0};
    int expected_shape2[] = {1, 3};
    Tensor *expected2 = create_tensor(expected_data2, expected_shape2, 2);

    if (!are_tensors_equal(partial2, expected2, 1e-9)) passed = false;

    free_tensor(t2);
    free_tensor(partial2);
    free_tensor(expected2);

    // Test Case 3: Extract a single element (scalar)
    double data3[] = {1.0, 2.0, 3.0, 4.0}; // Shape {2, 2}
    int shape3[] = {2, 2};
    Tensor *t3 = create_tensor(data3, shape3, 2);
    int start_indices3[] = {1, 0};
    int end_indices3[] = {2, 1};
    Tensor *partial3 = partial_tensor(t3, start_indices3, end_indices3);

    double expected_data3[] = {3.0};
    int expected_shape3[] = {1, 1};
    Tensor *expected3 = create_tensor(expected_data3, expected_shape3, 2);

    if (!are_tensors_equal(partial3, expected3, 1e-9)) passed = false;

    free_tensor(t3);
    free_tensor(partial3);
    free_tensor(expected3);

    // Test Case 4: Invalid indices (start > end) (should return NULL)
    double data4[] = {1.0, 2.0};
    int shape4[] = {2};
    Tensor *t4 = create_tensor(data4, shape4, 1);
    int start_indices4[] = {1};
    int end_indices4[] = {0};
    Tensor *partial4 = partial_tensor(t4, start_indices4, end_indices4);
    if (partial4 != NULL) {
        passed = false;
        free_tensor(partial4);
    }
    free_tensor(t4);

    // Test Case 5: Indices out of bounds (should return NULL)
    double data5[] = {1.0, 2.0};
    int shape5[] = {2};
    Tensor *t5 = create_tensor(data5, shape5, 1);
    int start_indices5[] = {0};
    int end_indices5[] = {3}; // Out of bounds
    Tensor *partial5 = partial_tensor(t5, start_indices5, end_indices5);
    if (partial5 != NULL) {
        passed = false;
        free_tensor(partial5);
    }
    free_tensor(t5);


    print_test_result(test_name, passed);
}


// Main function to run all tests
int main() {
    printf("Starting Tensor C Tests (Standard Memory)...\n");

    test_create_and_free();
    test_get_set();
    test_reshape();
    test_transpose();
    test_add();
    test_subtract();
    test_multiply();
    test_dot_product();
    test_partial();

    printf("Tensor C Tests (Standard Memory) Finished.\n");

    return 0;
}
