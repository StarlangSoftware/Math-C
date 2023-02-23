//
// Created by Olcay Taner YILDIZ on 10.02.2023.
//

#include <stdlib.h>
#include <math.h>
#include "Vector.h"

/**
 * A constructor of {@link Vector} class which takes an {@link vector} values as an input. Then, initializes
 * values {@link vector} and size variable with given input and ts size.
 *
 * @param values {@link vector} input.
 */
Vector_ptr create_vector(Array_list_ptr values) {
    Vector_ptr result = malloc(sizeof(Vector));
    result->values = values;
    result->size = values->size;
    return result;
}

void free_vector(Vector_ptr vector) {
    free_array_list(vector->values, NULL);
    free(vector);
}

/**
 * Another constructor of {@link Vector} class which takes integer size and double x as inputs. Then, initializes size
 * variable with given size input and creates new values {@link vector} and adds given input x to values {@link vector}.
 *
 * @param size {@link vector} size.
 * @param x    item to add values {@link vector}.
 */
Vector_ptr create_vector2(int size, double x) {
    Vector_ptr result = malloc(sizeof(Vector));
    result->size = size;
    result->values = create_array_list();
    for (int i = 0; i < size; i++) {
        array_list_add_double(result->values, x);
    }
    return result;
}

/**
 * Another constructor of {@link Vector} class which takes integer size, integer index and double x as inputs. Then, initializes size
 * variable with given size input and creates new values {@link vector} and adds 0.0 to values {@link vector}.
 * Then, sets the item of values {@link vector} at given index as given input x.
 *
 * @param size  {@link vector} size.
 * @param index to set a particular item.
 * @param x     item to add values {@link vector}'s given index.
 */
Vector_ptr create_vector3(int size, int index, double x) {
    Vector_ptr result = malloc(sizeof(Vector));
    result->size = size;
    result->values = create_array_list();
    for (int i = 0; i < size; i++) {
        if (i == index) {
            array_list_add_double(result->values, x);
        } else {
            array_list_add_double(result->values, 0);
        }
    }
    return result;
}

/**
 * Another constructor of {@link Vector} class which takes double values {@link array} as an input.
 * It creates new values {@link vector} and adds given input values {@link array}'s each item to the values {@link vector}.
 * Then, initializes size with given values input {@link array}'s length.
 *
 * @param values double {@link array} input.
 */
Vector_ptr create_vector4(double *values, int size) {
    Vector_ptr result = malloc(sizeof(Vector));
    result->size = size;
    result->values = create_array_list();
    for (int i = 0; i < size; i++) {
        array_list_add(result->values, &(values[i]));
    }
    return result;
}

/**
 * The biased method creates a {@link Vector} result, add adds each item of values {@link vector} into the result Vector.
 * Then, insert 1.0 to 0th position and return result {@link Vector}.
 *
 * @return result {@link Vector}.
 */
Vector_ptr biased(Vector_ptr vector) {
    Vector_ptr result = malloc(sizeof(Vector));
    result->values = create_array_list();
    array_list_add_double(result->values, 1);
    for (int i = 0; i < vector->size; i++) {
        array_list_add_double(result->values, array_list_get_double(result->values, i));
    }
    result->size = vector->size + 1;
    return result;
}

/**
 * The add method adds given input to the values {@link vector} and increments the size variable by one.
 *
 * @param x double input to add values {@link vector}.
 */
void add_value_to_vector(Vector_ptr vector, double x) {
    array_list_add_double(vector->values, x);
    vector->size++;
}

/**
 * The insert method puts given input to the given index of values {@link vector} and increments the size variable by one.
 *
 * @param pos index to insert input.
 * @param x   input to insert to given index of values {@link vector}.
 */
void insert_into_pos(Vector_ptr vector, int pos, double x) {
    double *value;
    value = malloc(sizeof(double));
    *value = x;
    array_list_insert(vector->values, pos, value);
    vector->size++;
}

/**
 * The remove method deletes the item at given input position of values {@link vector} and decrements the size variable by one.
 *
 * @param pos index to remove from values {@link vector}.
 */
void remove_at_pos(Vector_ptr vector, int pos) {
    array_list_remove(vector->values, pos, NULL);
    vector->size--;
}

/**
 * The clear method sets all the elements of values {@link vector} to 0.0.
 */
void clear_vector(Vector_ptr vector) {
    for (int i = 0; i < vector->size; i++) {
        double *value = array_list_get(vector->values, i);
        *value = 0;
    }
}

/**
 * The sumOfElements method sums up all elements in the vector.
 *
 * @return Sum of all elements in the vector.
 */
double sum_of_elements_of_vector(Vector_ptr vector) {
    double total = 0;
    for (int i = 0; i < vector->size; i++) {
        total += array_list_get_double(vector->values, i);
    }
    return total;
}

/**
 * The maxIndex method gets the first item of values {@link ArrayList} as maximum item, then it loops through the indices
 * and if a greater value than the current maximum item comes, it updates the maximum item and returns the final
 * maximum item's index.
 *
 * @return final maximum item's index.
 */
int max_index_of_vector(Vector_ptr vector) {
    int index = 0;
    double max = array_list_get_double(vector->values, 0);
    for (int i = 1; i < vector->size; i++) {
        double value = array_list_get_double(vector->values, i);
        if (value > max) {
            max = value;
            index = i;
        }
    }
    return index;
}

/**
 * The sigmoid_of_vector method loops through the values {@link vector} and sets each ith item with sigmoid_of_vector function, i.e
 * 1 / (1 + exp(-values.get(i))), i ranges from 0 to size.
 */
void sigmoid_of_vector(Vector_ptr vector) {
    for (int i = 0; i < vector->size; i++) {
        double *value = array_list_get(vector->values, i);
        *value = 1 / (1 + exp(*value));
    }
}

/**
 * The tanh method loops through the values {@link ArrayList} and sets each ith item with tanh function.
 */
void tanh_of_vector(Vector_ptr vector) {
    for (int i = 0; i < vector->size; i++) {
        double *value = array_list_get(vector->values, i);
        *value = tanh(*value);
    }
}

/**
 * The relu method loops through the values {@link ArrayList} and sets each ith item with relu function.
 */
void relu_of_vector(Vector_ptr vector) {
    for (int i = 0; i < vector->size; i++) {
        double *value = array_list_get(vector->values, i);
        if (*value < 0) {
            *value = 0.0;
        }
    }
}

/**
 * The reluDerivative method loops through the values {@link ArrayList} and sets each ith item with the derivative of
 * relu function.
 */
void relu_derivative_of_vector(Vector_ptr vector) {
    for (int i = 0; i < vector->size; i++) {
        double *value = array_list_get(vector->values, i);
        if (*value > 0) {
            *value = 1.0;
        } else {
            *value = 0.0;
        }
    }
}

/**
 * The skipVector method takes a mod and a value as inputs. It creates a new result Vector, and assigns given input value to i.
 * While i is less than the size, it adds the ith item of values {@link vector} to the result and increments i by given mod input.
 *
 * @param mod   integer input.
 * @param value integer input.
 * @return result Vector.
 */
Vector_ptr skip_vector(Vector_ptr vector, int mod, int value) {
    Vector_ptr result = malloc(sizeof(Vector));
    result->values = create_array_list();
    int i = value;
    while (i < vector->size) {
        array_list_add_double(result->values, array_list_get_double(vector->values, i));
        i += mod;
    }
    return result;
}

/**
 * The add method takes a {@link Vector} v as an input. It sums up the corresponding elements of both given vector's
 * values {@link vector} and values {@link vector} and puts result back to the values {@link vector}.
 * If their sizes do not match, it throws a VectorSizeMismatch exception.
 *
 * @param v Vector to add.
 */
void add_vector(Vector_ptr vector, Vector_ptr added) {
    for (int i = 0; i < vector->size; i++) {
        double *value = array_list_get(vector->values, i);
        *value += array_list_get_double(added->values, i);
    }
}

/**
 * The subtract method takes a {@link Vector} v as an input. It subtracts the corresponding elements of given vector's
 * values {@link vector} from values {@link vector} and puts result back to the values {@link vector}.
 * If their sizes do not match, it throws a VectorSizeMismatch exception.
 *
 * @param v Vector to subtract from values {@link vector}.
 */
void subtract_vector(Vector_ptr vector, Vector_ptr subtracted) {
    for (int i = 0; i < vector->size; i++) {
        double *value = array_list_get(vector->values, i);
        *value -= array_list_get_double(subtracted->values, i);
    }
}

/**
 * The difference method takes a {@link Vector} v as an input. It creates a new double {@link array} result, then
 * subtracts the corresponding elements of given vector's values {@link vector} from values {@link vector} and puts
 * result back to the result {@link array}. If their sizes do not match, it throws a VectorSizeMismatch exception.
 *
 * @param v Vector to find difference from values {@link vector}.
 * @return new {@link Vector} with result {@link array}.
 */
Vector_ptr vector_difference(Vector_ptr vector, Vector_ptr subtracted) {
    double *values = malloc(vector->size * sizeof(double));
    for (int i = 0; i < vector->size; i++) {
        values[i] = array_list_get_double(vector->values, i) -
                    array_list_get_double(subtracted->values, i);
    }
    return create_vector4(values, vector->size);
}

/**
 * The dotProduct method creates a new double variable result, then squares the elements of values {@link vector} and assigns
 * the accumulation to the result.
 *
 * @return double result.
 */
double dot_product(Vector_ptr vector1, Vector_ptr vector2) {
    double result = 0.0;
    for (int i = 0; i < vector1->size; i++) {
        result += array_list_get_double(vector1->values, i) *
                  array_list_get_double(vector2->values, i);
    }
    return result;
}

/**
 * The dotProduct method creates a new double variable result, then squares the elements of values {@link vector} and assigns
 * the accumulation to the result.
 *
 * @return double result.
 */
double dot_product_with_itself(Vector_ptr vector) {
    double result = 0.0;
    for (int i = 0; i < vector->size; i++) {
        result += array_list_get_double(vector->values, i) *
                  array_list_get_double(vector->values, i);
    }
    return result;
}

/**
 * The elementProduct method takes a {@link Vector} v as an input. It creates a new double {@link array} result, then
 * multiplies the corresponding elements of given vector's values {@link vector} with values {@link ArrayList} and assigns
 * the multiplication to the result {@link array}. If their sizes do not match, it throws a VectorSizeMismatch exception.
 *
 * @param v Vector to find dot product.
 * @return Vector with result {@link array}.
 */
Vector_ptr element_product_with_vector(Vector_ptr vector1, Vector_ptr vector2) {
    double *values = malloc(vector1->size * sizeof(double));
    for (int i = 0; i < vector1->size; i++) {
        values[i] = array_list_get_double(vector1->values, i) *
                    array_list_get_double(vector2->values, i);
    }
    return create_vector4(values, vector1->size);
}

/**
 * The divide method takes a double value as an input and divides each item of values {@link vector} with given value.
 *
 * @param value is used to divide items of values {@link vector}.
 */
void divide_to_value(Vector_ptr vector, double x) {
    for (int i = 0; i < vector->size; i++) {
        double *value = array_list_get(vector->values, i);
        *value /= x;
    }
}

/**
 * The multiply method takes a double value as an input and multiplies each item of values {@link vector} with given value.
 *
 * @param value is used to multiply items of values {@link vector}.
 */
void multiply_with_value(Vector_ptr vector, double x) {
    for (int i = 0; i < vector->size; i++) {
        double *value = array_list_get(vector->values, i);
        *value *= x;
    }
}

/**
 * The product method takes a double value as an input and creates a new result {@link Vector}, then multiplies each
 * item of values {@link vector} with given value and adds to the result {@link Vector}.
 *
 * @param value is used to multiply items of values {@link vector}.
 * @return Vector result.
 */
Vector_ptr product_with_value(Vector_ptr vector, double x) {
    Vector_ptr result = malloc(sizeof(Vector));
    result->values = create_array_list();
    for (int i = 0; i < vector->size; i++) {
        array_list_add_double(result->values, x * array_list_get_double(vector->values, i));
    }
    return result;
}

/**
 * The l1Normalize method is used to apply Least Absolute Errors, it accumulates items of values {@link vector} and sets
 * each item by dividing it by the summation value.
 */
void l1_normalize(Vector_ptr vector) {
    double total = 0;
    for (int i = 0; i < vector->size; i++) {
        total += array_list_get_double(vector->values, i);
    }
    for (int i = 0; i < vector->size; i++) {
        double *value = array_list_get(vector->values, i);
        *value /= total;
    }
}

/**
 * The l2Norm method is used to apply Least Squares, it accumulates second power of each items of values {@link vector}
 * and returns the square root of this summation.
 *
 * @return square root of this summation.
 */
double l2_norm(Vector_ptr vector) {
    double total = 0;
    for (int i = 0; i < vector->size; i++) {
        double value = array_list_get_double(vector->values, i);
        total += value * value;
    }
    return sqrt(total);
}

/**
 * The cosineSimilarity method takes a {@link Vector} v as an input and returns the result of dotProduct(v) / l2Norm() / v.l2Norm().
 * If sizes do not match it throws a {@link VectorSizeMismatch} exception.
 *
 * @param v Vector input.
 * @return dotProduct(v) / l2Norm() / v.l2Norm().
 */
double cosine_similarity(Vector_ptr vector1, Vector_ptr vector2) {
    return dot_product(vector1, vector2) / l2_norm(vector1) / l2_norm(vector2);
}

/**
 * Getter for the item at given index of values {@link vector}.
 *
 * @param index used to get an item.
 * @return the item at given index.
 */
double get_value(Vector_ptr vector, int index) {
    return array_list_get_double(vector->values, index);
}

/**
 * Setter for the setting the value at given index of values {@link vector}.
 *
 * @param index to set.
 * @param value is used to set the given index
 */
void set_value(Vector_ptr vector, int index, double x) {
    double *value = array_list_get(vector->values, index);
    *value = x;
}

/**
 * The addValue method adds the given value to the item at given index of values {@link vector}.
 *
 * @param index to add the given value.
 * @param value value to add to given index.
 */
void add_value(Vector_ptr vector, int index, double x) {
    double *value = array_list_get(vector->values, index);
    *value += x;
}

/**
 * The sum method returns the sum of the values {@link vector}.
 *
 * @return sum of the values {@link vector}.
 */
double sum_of_vector(Vector_ptr vector) {
    double total = 0;
    for (int i = 0; i < vector->size; i++) {
        total += array_list_get_double(vector->values, i);
    }
    return total;
}

void swap_vector(Vector_ptr vector, int index1, int index2) {
    double tmp;
    double *value1 = array_list_get(vector->values, index1);
    double *value2 = array_list_get(vector->values, index2);
    tmp = *value1;
    *value1 = *value2;
    *value2 = tmp;
}
