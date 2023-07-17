//
// Created by Olcay Taner YILDIZ on 13.02.2023.
//

#include "Vector.h"

#ifndef MATH_MATRIX_H
#define MATH_MATRIX_H

struct matrix {
    int row;
    int col;
    double **values;
};

typedef struct matrix Matrix;
typedef Matrix *Matrix_ptr;

Matrix_ptr create_matrix(int row, int col);

Matrix_ptr create_matrix2(int row, int col, double min, double max, int seed);

Matrix_ptr create_matrix3(int size);

Matrix_ptr create_matrix4(const Vector* vector1, const Vector* vector2);

Matrix_ptr create_matrix5(FILE* input_file);

Matrix_ptr clone(const Matrix* matrix);

void allocate_matrix(Matrix_ptr matrix, int row, int col);

void free_matrix(Matrix_ptr matrix);

void add_value_to_matrix(Matrix_ptr matrix, int rowNo, int colNo, double value);

void increment(Matrix_ptr matrix, int rowNo, int colNo);

Vector_ptr get_row(const Matrix* matrix, int row);

Array_list_ptr get_column(const Matrix* matrix, int column);

void column_wise_normalize(Matrix_ptr matrix);

void multiply_with_constant(Matrix_ptr matrix, double constant);

void divide_by_constant(Matrix_ptr matrix, double constant);

void add_matrix(Matrix_ptr matrix1, const Matrix* matrix2);

void add_vector_to_matrix(Matrix_ptr matrix, int row, const Vector* v);

void subtract_matrix(Matrix_ptr matrix1, const Matrix* matrix2);

Vector_ptr multiply_with_vector_from_left(const Matrix* matrix, const Vector* vector);

Vector_ptr multiply_with_vector_from_right(const Matrix* matrix, const Vector* vector);

double column_sum(const Matrix* matrix, int columnNo);

Vector_ptr sum_of_rows(const Matrix* matrix);

double row_sum(const Matrix* matrix, int row);

Matrix_ptr multiply_with_matrix(const Matrix* matrix1, const Matrix* matrix2);

Matrix_ptr element_product_with_matrix(const Matrix* matrix1, const Matrix* matrix2);

double sum_of_elements_of_matrix(const Matrix* matrix);

double trace(const Matrix* matrix);

Matrix_ptr transpose(const Matrix* matrix);

Matrix_ptr partial(const Matrix* matrix, int rowstart, int rowend, int colstart, int colend);

bool is_symmetric(const Matrix* matrix);

double determinant(const Matrix* matrix);

void inverse(Matrix_ptr matrix);

Matrix_ptr cholesky_decomposition(const Matrix* matrix);

void rotate(Matrix_ptr matrix, double s, double tau, int i, int j, int k, int l);

Array_list_ptr characteristics(const Matrix* matrix);

Matrix_ptr sum_matrix(const Matrix* matrix1, const Matrix* matrix2);

Matrix_ptr difference_matrix(const Matrix* matrix1, const Matrix* matrix2);

#endif //MATH_MATRIX_H
