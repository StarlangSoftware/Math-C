//
// Created by Olcay Taner YILDIZ on 10.02.2023.
//

#ifndef MATH_VECTOR_H
#define MATH_VECTOR_H

#include <ArrayList.h>
#include <stdio.h>

struct vector {
    int size;
    Array_list_ptr values;
};

typedef struct vector Vector;
typedef Vector *Vector_ptr;

Vector_ptr create_vector(Array_list_ptr values);

Vector_ptr create_vector2(int size, double x);

Vector_ptr create_vector3(int size, int index, double x);

Vector_ptr create_vector4(double *values, int size);

Vector_ptr create_vector5(FILE* input_file);

void free_vector(Vector_ptr vector);

Vector_ptr biased(const Vector* vector);

void add_value_to_vector(Vector_ptr vector, double x);

void insert_into_pos(Vector_ptr vector, int pos, double x);

void remove_at_pos(Vector_ptr vector, int pos);

void clear_vector(Vector_ptr vector);

double sum_of_elements_of_vector(const Vector* vector);

int max_index_of_vector(const Vector* vector);

void sigmoid_of_vector(Vector_ptr vector);

void tanh_of_vector(Vector_ptr vector);

void relu_of_vector(Vector_ptr vector);

void relu_derivative_of_vector(Vector_ptr vector);

Vector_ptr skip_vector(const Vector* vector, int mod, int value);

void add_vector(Vector_ptr vector, const Vector* added);

void subtract_vector(Vector_ptr vector, const Vector* subtracted);

Vector_ptr vector_difference(const Vector* vector, const Vector* subtracted);

double dot_product(const Vector* vector1, const Vector* vector2);

double dot_product_with_itself(const Vector* vector);

Vector_ptr element_product_with_vector(const Vector* vector1, const Vector* vector2);

void divide_to_value(Vector_ptr vector, double x);

void multiply_with_value(Vector_ptr vector, double x);

Vector_ptr product_with_value(const Vector* vector, double x);

void l1_normalize(Vector_ptr vector);

double l2_norm(const Vector* vector);

double cosine_similarity(const Vector* vector1, const Vector* vector2);

double get_value(const Vector* vector, int index);

void set_value(Vector_ptr vector, int index, double x);

void add_value(Vector_ptr vector, int index, double x);

double sum_of_vector(const Vector* vector);

void swap_vector(Vector_ptr vector, int index1, int index2);

#endif //MATH_VECTOR_H
