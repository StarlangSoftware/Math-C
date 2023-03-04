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

Matrix_ptr create_matrix2(int row, int col, double min, double max);

Matrix_ptr create_matrix3(int size);

Matrix_ptr create_matrix4(Vector_ptr vector1, Vector_ptr vector2);

Matrix_ptr clone(Matrix_ptr matrix);

void allocate_matrix(Matrix_ptr matrix, int row, int col);

void free_matrix(Matrix_ptr matrix);

void add_value_to_matrix(Matrix_ptr matrix, int rowNo, int colNo, double value);

void increment(Matrix_ptr matrix, int rowNo, int colNo);

Vector_ptr get_row(Matrix_ptr matrix, int row);

Array_list_ptr get_column(Matrix_ptr matrix, int column);

void column_wise_normalize(Matrix_ptr matrix);

void multiply_with_constant(Matrix_ptr matrix, double constant);

void divide_by_constant(Matrix_ptr matrix, double constant);

void add_matrix(Matrix_ptr matrix1, Matrix_ptr matrix2);

void add_vector_to_matrix(Matrix_ptr matrix, int row, Vector_ptr v);

void subtract_matrix(Matrix_ptr matrix1, Matrix_ptr matrix2);

Vector_ptr multiply_with_vector_from_left(Matrix_ptr matrix, Vector_ptr vector);

Vector_ptr multiply_with_vector_from_right(Matrix_ptr matrix, Vector_ptr vector);

double column_sum(Matrix_ptr matrix, int columnNo);

Vector_ptr sum_of_rows(Matrix_ptr matrix);

double row_sum(Matrix_ptr matrix, int row);

Matrix_ptr multiply_with_matrix(Matrix_ptr matrix1, Matrix_ptr matrix2);

Matrix_ptr element_product_with_matrix(Matrix_ptr matrix1, Matrix_ptr matrix2);

double sum_of_elements_of_matrix(Matrix_ptr matrix);

double trace(Matrix_ptr matrix);

Matrix_ptr transpose(Matrix_ptr matrix);

Matrix_ptr partial(Matrix_ptr matrix, int rowstart, int rowend, int colstart, int colend);

bool is_symmetric(Matrix_ptr matrix);

double determinant(Matrix_ptr matrix);

void inverse(Matrix_ptr matrix);

Matrix_ptr cholesky_decomposition(Matrix_ptr matrix);

void rotate(Matrix_ptr matrix, double s, double tau, int i, int j, int k, int l);

Array_list_ptr characteristics(Matrix_ptr matrix);

Matrix_ptr sum_matrix(Matrix_ptr matrix1, Matrix_ptr matrix2);

Matrix_ptr difference_matrix(Matrix_ptr matrix1, Matrix_ptr matrix2);

#endif //MATH_MATRIX_H
