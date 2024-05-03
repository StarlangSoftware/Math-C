//
// Created by Olcay Taner YILDIZ on 13.02.2023.
//

#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <Memory/Memory.h>
#include "Matrix.h"
#include "Eigenvector.h"

/**
 * Another constructor of Matrix class which takes row and column numbers as inputs and creates new values
 * array with given parameters.
 *
 * @param row is used to create matrix.
 * @param col is used to create matrix.
 */
Matrix_ptr create_matrix(int row, int col) {
    Matrix_ptr result = malloc_(sizeof(Matrix), "create_matrix");
    allocate_matrix(result, row, col);
    return result;
}

/**
 * Another constructor of Matrix class which takes row, column, minimum and maximum values as inputs.
 * First it creates new values array with given row and column numbers. Then fills in the
 * positions with random numbers using minimum and maximum inputs.
 *
 * @param row is used to create matrix.
 * @param col is used to create matrix.
 * @param min minimum value.
 * @param max maximum value.
 */
Matrix_ptr create_matrix2(int row, int col, double min, double max, int seed) {
    srandom(seed);
    Matrix_ptr result = malloc_(sizeof(Matrix), "create_matrix2");
    allocate_matrix(result, row, col);
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            result->values[i][j] = min + (double) random() / (double) (RAND_MAX / (max - min));
        }
    }
    return result;
}

/**
 * Another constructor of Matrix class which takes size as input and creates new values array
 * with using size input and assigns 1 to each element at the diagonal.
 *
 * @param size is used declaring the size of the array.
 */
Matrix_ptr create_matrix3(int size) {
    Matrix_ptr result = malloc_(sizeof(Matrix), "create_matrix3");
    allocate_matrix(result, size, size);
    for (int i = 0; i < size; i++) {
        result->values[i][i] = 1;
    }
    return result;
}

Matrix_ptr create_matrix4(const Vector* vector1, const Vector* vector2) {
    Matrix_ptr result = malloc_(sizeof(Matrix), "create_matrix4");
    allocate_matrix(result, vector1->size, vector2->size);
    for (int i = 0; i < result->row; i++) {
        for (int j = 0; j < result->col; j++) {
            result->values[i][j] = get_value(vector1, i) * get_value(vector2, j);
        }
    }
    return result;
}

Matrix_ptr clone(const Matrix* matrix) {
    Matrix_ptr result = malloc_(sizeof(Matrix), "clone");
    allocate_matrix(result, matrix->row, matrix->col);
    for (int i = 0; i < matrix->row; i++) {
        for (int j = 0; j < matrix->col; j++) {
            result->values[i][j] = matrix->values[i][j];
        }
    }
    return result;
}

void allocate_matrix(Matrix_ptr matrix, int row, int col) {
    matrix->row = row;
    matrix->col = col;
    matrix->values = allocate_2d(matrix->row, matrix->col);
}

void free_matrix(Matrix_ptr matrix) {
    free_2d(matrix->values, matrix->row);
    free_(matrix);
}

/**
 * The addValue method adds the given value to the item at given index of values {@link java.lang.reflect.Array}.
 *
 * @param rowNo integer input for row number.
 * @param colNo integer input for column number.
 * @param value is used to add to given item at given index.
 */
void add_value_to_matrix(Matrix_ptr matrix, int rowNo, int colNo, double value) {
    matrix->values[rowNo][colNo] += value;
}

/**
 * The increment method adds 1 to the item at given index of values array.
 *
 * @param rowNo integer input for row number.
 * @param colNo integer input for column number.
 */
void increment(Matrix_ptr matrix, int rowNo, int colNo) {
    matrix->values[rowNo][colNo]++;
}

/**
 * The getRow method returns the vector of values array at given row input.
 *
 * @param _row integer input for row number.
 * @return Vector of values array at given row input.
 */
Vector_ptr get_row(const Matrix* matrix, int row) {
    return create_vector4(matrix->values[row], matrix->col);
}

/**
 * The getColumn method creates an vector and adds items at given column number of values array
 * to the vector.
 *
 * @param column integer input for column number.
 * @return Vector of given column number.
 */
Array_list_ptr get_column(const Matrix* matrix, int column) {
    Array_list_ptr vector;
    vector = create_array_list();
    for (int i = 0; i < matrix->row; i++) {
        array_list_add_double(vector, matrix->values[i][column]);
    }
    return vector;
}

/**
 * The columnWiseNormalize method, first accumulates items column by column then divides items by the summation.
 */
void column_wise_normalize(Matrix_ptr matrix) {
    for (int i = 0; i < matrix->row; i++) {
        double sum = 0.0;
        for (int j = 0; j < matrix->col; j++) {
            sum += matrix->values[i][j];
        }
        for (int j = 0; j < matrix->col; j++) {
            matrix->values[i][j] /= sum;
        }
    }
}

/**
 * The multiplyWithConstant method takes a constant as an input and multiplies each item of values array
 * with given constant.
 *
 * @param constant value to multiply items of values array.
 */
void multiply_with_constant(Matrix_ptr matrix, double constant) {
    int i;
    for (i = 0; i < matrix->row; i++) {
        for (int j = 0; j < matrix->col; j++) {
            matrix->values[i][j] *= constant;
        }
    }
}

/**
 * The divideByConstant method takes a constant as an input and divides each item of values array
 * with given constant.
 *
 * @param constant value to divide items of values array.
 */
void divide_by_constant(Matrix_ptr matrix, double constant) {
    int i;
    for (i = 0; i < matrix->row; i++) {
        for (int j = 0; j < matrix->col; j++) {
            matrix->values[i][j] /= constant;
        }
    }
}

/**
 * The add method takes a Matrix as an input and accumulates values array with the
 * corresponding items of given Matrix. If the sizes of both Matrix and values array do not match,
 * it throws MatrixDimensionMismatch exception.
 *
 * @param m Matrix type input.
 */
void add_matrix(Matrix_ptr matrix1, const Matrix* matrix2) {
    for (int i = 0; i < matrix1->row; i++) {
        for (int j = 0; j < matrix1->col; j++) {
            matrix1->values[i][j] += matrix2->values[i][j];
        }
    }
}

/**
 * The add method which takes a row number and a Vector as inputs. It sums up the corresponding values at the given row of
 * values array and given Vector. If the sizes of both Matrix and values
 * array do not match, it throws MatrixColumnMismatch exception.
 *
 * @param rowNo integer input for row number.
 * @param v     Vector type input.
 */
void add_vector_to_matrix(Matrix_ptr matrix, int row, const Vector* v) {
    for (int i = 0; i < matrix->col; i++) {
        matrix->values[row][i] += get_value(v, i);
    }
}

/**
 * The subtract method takes a Matrix as an input and subtracts from values array the
 * corresponding items of given Matrix. If the sizes of both Matrix and values aArray do not match,
 * it throws MatrixDimensionMismatch exception.
 *
 * @param m Matrix type input.
 */
void subtract_matrix(Matrix_ptr matrix1, const Matrix* matrix2) {
    for (int i = 0; i < matrix1->row; i++) {
        for (int j = 0; j < matrix1->col; j++) {
            matrix1->values[i][j] -= matrix2->values[i][j];
        }
    }
}

/**
 * The multiplyWithVectorFromLeft method takes a Vector as an input and creates a result array.
 * Then, multiplies values of input Vector starting from the left side with the values array,
 * accumulates the multiplication, and assigns to the result array. If the sizes of both Vector
 * and row number do not match, it throws MatrixRowMismatch exception.
 *
 * @param v Vector type input.
 * @return Vector that holds the result.
 */
Vector_ptr multiply_with_vector_from_left(const Matrix* matrix, const Vector* vector) {
    double *values = malloc_(matrix->col * sizeof(double), "multiply_with_vector_from_left");
    for (int i = 0; i < matrix->col; i++) {
        values[i] = 0.0;
        for (int j = 0; j < matrix->row; j++) {
            values[i] += get_value(vector, j) * matrix->values[j][i];
        }
    }
    Vector_ptr result = create_vector4(values, matrix->col);
    free_(values);
    return result;
}

/**
 * The multiplyWithVectorFromRight method takes a Vector as an input and creates a result array.
 * Then, multiplies values of input Vector starting from the right side with the values array,
 * accumulates the multiplication, and assigns to the result array. If the sizes of both Vector
 * and row number do not match, it throws MatrixColumnMismatch exception.
 *
 * @param v Vector type input.
 * @return Vector that holds the result.
 */
Vector_ptr multiply_with_vector_from_right(const Matrix* matrix, const Vector* vector) {
    double *values = malloc_(matrix->row * sizeof(double), "multiply_with_vector_from_right");
    for (int i = 0; i < matrix->row; i++) {
        values[i] = 0;
        for (int j = 0; j < matrix->col; j++) {
            values[i] += matrix->values[i][j] * get_value(vector, j);
        }
    }
    Vector_ptr result = create_vector4(values, matrix->row);
    free_(values);
    return result;
}

/**
 * The columnSum method takes a column number as an input and accumulates items at given column number of values
 * array.
 *
 * @param columnNo Column number input.
 * @return summation of given column of values array.
 */
double column_sum(const Matrix* matrix, int columnNo) {
    double sum = 0;
    for (int i = 0; i < matrix->row; i++) {
        sum += matrix->values[i][columnNo];
    }
    return sum;
}

/**
 * The sumOfRows method creates a mew result Vector and adds the result of columnDum method's corresponding
 * index to the newly created result Vector.
 *
 * @return Vector that holds column sum.
 */
Vector_ptr sum_of_rows(const Matrix* matrix) {
    double *values = malloc_(matrix->col * sizeof(double), "sum_of_rows");
    for (int i = 0; i < matrix->col; i++) {
        values[i] = column_sum(matrix, i);
    }
    Vector_ptr result = create_vector4(values, matrix->col);
    free_(values);
    return result;
}

/**
 * The rowSum method takes a row number as an input and accumulates items at given row number of values
 * array.
 *
 * @param rowNo Row number input.
 * @return summation of given row of values array.
 */
double row_sum(const Matrix* matrix, int row) {
    double sum = 0;
    for (int i = 0; i < matrix->col; i++) {
        sum += matrix->values[matrix->row][i];
    }
    return sum;
}

/**
 * The multiply method takes a Matrix as an input. First it creates a result Matrix and puts the
 * accumulatated multiplication of values array and given Matrix into result
 * Matrix. If the size of Matrix's row size and values array's column size do not match,
 * it throws MatrixRowColumnMismatch exception.
 *
 * @param m Matrix type input.
 * @return result Matrix.
 */
Matrix_ptr multiply_with_matrix(const Matrix* matrix1, const Matrix* matrix2) {
    double sum;
    Matrix_ptr result = malloc_(sizeof(Matrix), "multiply_with_matrix");
    allocate_matrix(result, matrix1->row, matrix2->col);
    for (int i = 0; i < matrix1->row; i++) {
        for (int j = 0; j < matrix2->col; j++) {
            sum = 0.0;
            for (int k = 0; k < matrix1->col; k++) {
                sum += matrix1->values[i][k] * matrix2->values[k][j];
            }
            result->values[i][j] = sum;
        }
    }
    return result;
}

/**
 * The elementProduct method takes a Matrix as an input and performs element wise multiplication. Puts result
 * to the newly created Matrix. If the size of Matrix's row and column size does not match with the values
 * array's row and column size, it throws MatrixDimensionMismatch exception.
 *
 * @param m Matrix type input.
 * @return result Matrix.
 */
Matrix_ptr element_product_with_matrix(const Matrix* matrix1, const Matrix* matrix2) {
    Matrix_ptr result = malloc_(sizeof(Matrix), "element_product_with_matrix");
    allocate_matrix(result, matrix1->row, matrix1->col);
    for (int i = 0; i < matrix1->row; i++) {
        for (int j = 0; j < matrix2->col; j++) {
            result->values[i][j] = matrix1->values[i][j] * matrix2->values[i][j];
        }
    }
    return result;
}

/**
 * The sumOfElements method accumulates all the items in values array and
 * returns this summation.
 *
 * @return sum of the items of values array.
 */
double sum_of_elements_of_matrix(const Matrix* matrix) {
    double sum = 0.0;
    for (int i = 0; i < matrix->row; i++) {
        for (int j = 0; j < matrix->col; j++) {
            sum += matrix->values[i][j];
        }
    }
    return sum;
}

/**
 * The trace method accumulates items of values {@link java.lang.reflect.Array} at the diagonal.
 *
 * @return sum of items at diagonal.
 */
double trace(const Matrix* matrix) {
    double sum = 0.0;
    for (int i = 0; i < matrix->row; i++) {
        sum += matrix->values[i][i];
    }
    return sum;
}

/**
 * The transpose method creates a new Matrix, then takes the transpose of values array
 * and puts transposition to the Matrix.
 *
 * @return Matrix type output.
 */
Matrix_ptr transpose(const Matrix* matrix) {
    Matrix_ptr result = malloc_(sizeof(Matrix), "transpose");
    allocate_matrix(result, matrix->col, matrix->row);
    for (int i = 0; i < matrix->row; i++) {
        for (int j = 0; j < matrix->col; j++) {
            result->values[j][i] = matrix->values[i][j];
        }
    }
    return result;
}

/**
 * The partial method takes 4 integer inputs; rowstart, rowend, colstart, colend and creates a Matrix size of
 * rowend - rowstart + 1 x colend - colstart + 1. Then, puts corresponding items of values array
 * to the new result Matrix.
 *
 * @param rowstart integer input for defining starting index of row.
 * @param rowend   integer input for defining ending index of row.
 * @param colstart integer input for defining starting index of column.
 * @param colend   integer input for defining ending index of column.
 * @return result Matrix.
 */
Matrix_ptr partial(const Matrix* matrix, int rowstart, int rowend, int colstart, int colend) {
    Matrix_ptr result = malloc_(sizeof(Matrix), "partial");
    allocate_matrix(result, rowend - rowstart + 1, colend - colstart + 1);
    for (int i = rowstart; i <= rowend; i++)
        for (int j = colstart; j <= colend; j++)
            result->values[i - rowstart][j - colstart] = matrix->values[i][j];
    return result;
}

/**
 * The isSymmetric method compares each item of values array at positions (i, j) with (j, i)
 * and returns true if they are equal, false otherwise.
 *
 * @return true if items are equal, false otherwise.
 */
bool is_symmetric(const Matrix* matrix) {
    for (int i = 0; i < matrix->row - 1; i++) {
        for (int j = i + 1; j < matrix->row; j++) {
            if (matrix->values[i][j] != matrix->values[j][i]) {
                return false;
            }
        }
    }
    return true;
}

/**
 * The determinant method first creates a new array, and copies the items of  values
 * array into new array. Then, calculates the determinant of this
 * new array.
 *
 * @return determinant of values array.
 */
double determinant(const Matrix* matrix) {
    double det = 1, ratio;
    Matrix_ptr copy = clone(matrix);
    for (int i = 0; i < matrix->row; i++) {
        det *= copy->values[i][i];
        if (det == 0.0)
            break;
        for (int j = i + 1; j < matrix->row; j++) {
            ratio = copy->values[j][i] / copy->values[i][i];
            for (int k = i; k < matrix->col; k++)
                copy->values[j][k] -= copy->values[i][k] * ratio;
        }
    }
    free_matrix(copy);
    return det;
}

/**
 * The inverse method finds the inverse of values array.
 */
void inverse(Matrix_ptr matrix) {
    double big;
    double dum, pivinv;
    int i, icol, irow, k, l, ll;
    Matrix_ptr b = create_matrix3(matrix->row);
    int *indxc, *indxr, *ipiv;
    indxc = calloc_(matrix->row, sizeof(int), "inverse_1");
    indxr = calloc_(matrix->row, sizeof(int), "inverse_2");
    ipiv = calloc_(matrix->row, sizeof(int), "inverse_3");
    for (i = 1; i <= matrix->row; i++) {
        big = 0.0;
        irow = -1;
        icol = -1;
        for (int j = 1; j <= matrix->row; j++) {
            if (ipiv[j - 1] != 1) {
                for (k = 1; k <= matrix->row; k++) {
                    if (ipiv[k - 1] == 0) {
                        if (fabs(matrix->values[j - 1][k - 1]) >= big) {
                            big = fabs(matrix->values[j - 1][k - 1]);
                            irow = j;
                            icol = k;
                        }
                    }
                }
            }
        }
        ipiv[icol - 1] = ipiv[icol - 1] + 1;
        if (irow != icol) {
            double *dummy = malloc_(matrix->col * sizeof(double), "inverse");
            memcpy(dummy, matrix->values[irow - 1], matrix->col * sizeof(double));
            memcpy(matrix->values[irow - 1], matrix->values[icol - 1], matrix->col * sizeof(double));
            memcpy(matrix->values[icol - 1], dummy, matrix->col * sizeof(double));
            memcpy(dummy, b->values[irow - 1], matrix->col * sizeof(double));
            memcpy(b->values[irow - 1], b->values[icol - 1], matrix->col * sizeof(double));
            memcpy(b->values[icol - 1], dummy, matrix->col * sizeof(double));
            free_(dummy);
        }
        indxr[i - 1] = irow;
        indxc[i - 1] = icol;
        pivinv = (1.0) / (matrix->values[icol - 1][icol - 1]);
        matrix->values[icol - 1][icol - 1] = 1.0;
        for (int j = 0; j < matrix->col; j++) {
            matrix->values[icol - 1][j] *= pivinv;
            b->values[icol - 1][j] *= pivinv;
        }
        for (ll = 1; ll <= matrix->row; ll++)
            if (ll != icol) {
                dum = matrix->values[ll - 1][icol - 1];
                matrix->values[ll - 1][icol - 1] = 0.0;
                for (l = 1; l <= matrix->row; l++)
                    matrix->values[ll - 1][l - 1] -= matrix->values[icol - 1][l - 1] * dum;
                for (l = 1; l <= matrix->row; l++)
                    b->values[ll - 1][l - 1] -= b->values[icol - 1][l - 1] * dum;
            }
    }
    for (l = matrix->row; l >= 1; l--) {
        if (indxr[l - 1] != indxc[l - 1]) {
            for (k = 1; k <= matrix->row; k++) {
                double tmp = matrix->values[k - 1][indxr[l - 1] - 1];
                matrix->values[k - 1][indxr[l - 1] - 1] = matrix->values[k - 1][indxc[l - 1] - 1];
                matrix->values[k - 1][indxc[l - 1] - 1] = tmp;
            }
        }
    }
    free_matrix(b);
    free_(indxc);
    free_(indxr);
    free_(ipiv);
}

/**
 * The choleskyDecomposition method creates a new Matrix and puts the Cholesky Decomposition of values Array
 * into this Matrix. Also, it throws MatrixNotSymmetric exception if it is not symmetric and
 * MatrixNotPositiveDefinite exception if the summation is negative.
 *
 * @return Matrix type output.
 */
Matrix_ptr cholesky_decomposition(const Matrix* matrix) {
    Matrix_ptr b;
    double sum;
    b = create_matrix(matrix->row, matrix->col);
    for (int i = 0; i < matrix->row; i++) {
        for (int j = i; j < matrix->row; j++) {
            sum = matrix->values[i][j];
            for (int k = i - 1; k >= 0; k--) {
                sum -= matrix->values[i][k] * matrix->values[j][k];
            }
            if (i == j) {
                b->values[i][i] = sqrt(sum);
            } else
                b->values[j][i] = sum / b->values[i][i];
        }
    }
    return b;
}

/**
 * The rotate method rotates values array according to given inputs.
 *
 * @param s   double input.
 * @param tau double input.
 * @param i   integer input.
 * @param j   integer input.
 * @param k   integer input.
 * @param l   integer input.
 */
void rotate(Matrix_ptr matrix, double s, double tau, int i, int j, int k, int l) {
    double g = matrix->values[i][j];
    double h = matrix->values[k][l];
    matrix->values[i][j] = g - s * (h + g * tau);
    matrix->values[k][l] = h + s * (g - h * tau);
}

/**
 * The characteristics method finds and returns a sorted vector of Eigenvectors. And it throws
 * MatrixNotSymmetric exception if it is not symmetric.
 *
 * @return a sorted vector of Eigenvectors.
 */
Array_list_ptr characteristics(const Matrix* matrix) {
    int j, iq, ip, i;
    double threshold, theta, tau, t, sm, s, h, g, c;
    Matrix_ptr matrix1 = clone(matrix);
    Matrix_ptr v;
    v = create_matrix3(matrix->row);
    double *d = malloc_(matrix->row * sizeof(double), "characteristics_1");
    double *b = malloc_(matrix->row * sizeof(double), "characteristics_2");
    double *z = malloc_(matrix->row * sizeof(double), "characteristics_3");
    double EPS = 0.000000000000000001;
    for (ip = 0; ip < matrix->row; ip++) {
        b[ip] = d[ip] = matrix1->values[ip][ip];
        z[ip] = 0.0;
    }
    for (i = 1; i <= 50; i++) {
        sm = 0.0;
        for (ip = 0; ip < matrix->row - 1; ip++)
            for (iq = ip + 1; iq < matrix->row; iq++)
                sm += fabs(matrix1->values[ip][iq]);
        if (sm == 0.0) {
            break;
        }
        if (i < 4)
            threshold = 0.2 * sm / pow(matrix->row, 2);
        else
            threshold = 0.0;
        for (ip = 0; ip < matrix->row - 1; ip++) {
            for (iq = ip + 1; iq < matrix->row; iq++) {
                g = 100.0 * fabs(matrix1->values[ip][iq]);
                if (i > 4 && g <= EPS * fabs(d[ip]) && g <= EPS * fabs(d[iq])) {
                    matrix1->values[ip][iq] = 0.0;
                } else {
                    if (fabs(matrix1->values[ip][iq]) > threshold) {
                        h = d[iq] - d[ip];
                        if (g <= EPS * fabs(h)) {
                            t = matrix1->values[ip][iq] / h;
                        } else {
                            theta = 0.5 * h / matrix1->values[ip][iq];
                            t = 1.0 / (fabs(theta) + sqrt(1.0 + pow(theta, 2)));
                            if (theta < 0.0) {
                                t = -t;
                            }
                        }
                        c = 1.0 / sqrt(1 + pow(t, 2));
                        s = t * c;
                        tau = s / (1.0 + c);
                        h = t * matrix1->values[ip][iq];
                        z[ip] -= h;
                        z[iq] += h;
                        d[ip] -= h;
                        d[iq] += h;
                        matrix1->values[ip][iq] = 0.0;
                        for (j = 0; j < ip; j++) {
                            rotate(matrix1, s, tau, j, ip, j, iq);
                        }
                        for (j = ip + 1; j < iq; j++) {
                            rotate(matrix1, s, tau, ip, j, j, iq);
                        }
                        for (j = iq + 1; j < matrix->row; j++) {
                            rotate(matrix1, s, tau, ip, j, iq, j);
                        }
                        for (j = 0; j < matrix->row; j++) {
                            rotate(v, s, tau, j, ip, j, iq);
                        }
                    }
                }
            }
        }
        for (ip = 0; ip < matrix->row; ip++) {
            b[ip] = b[ip] + z[ip];
            d[ip] = b[ip];
            z[ip] = 0.0;
        }
    }
    Array_list_ptr result;
    result = create_array_list();
    for (i = 0; i < matrix->row; i++) {
        if (d[i] > 0) {
            array_list_add(result, create_eigenvector(d[i], get_column(v, i)));
        }
    }
    qsort(result, result->size, sizeof(Eigenvector_ptr), (int (*)(const void *, const void *)) compare_eigenvector);
    free_(d);
    free_(b);
    free_(z);
    return result;
}

/**
 * The sum_matrix method takes two matrices as an input and sums values array of first with the
 * corresponding items of second Matrix and returns as a new Matrix.
 * @param matrix1 First matrix to be added.
 * @param matrix2 Second matrix to be added.
 * @return Sum of matrix1 and matrix2.
 */
Matrix_ptr sum_matrix(const Matrix* matrix1, const Matrix* matrix2) {
    Matrix_ptr result = create_matrix(matrix1->row, matrix1->col);
    for (int i = 0; i < matrix1->row; i++) {
        for (int j = 0; j < matrix1->col; j++) {
            result->values[i][j] = matrix1->values[i][j] + matrix2->values[i][j];
        }
    }
    return result;
}

/**
 * The difference_matrix method takes two matrices as an input and subtracts values array of second from the
 * corresponding items of first Matrix and returns as a new Matrix.
 * @param matrix1 First matrix from which second matrix is subtracted.
 * @param matrix2 Second matrix to be subtracted.
 * @return Difference of matrix1 and matrix2.
 */
Matrix_ptr difference_matrix(const Matrix* matrix1, const Matrix* matrix2) {
    Matrix_ptr result = create_matrix(matrix1->row, matrix1->col);
    for (int i = 0; i < matrix1->row; i++) {
        for (int j = 0; j < matrix1->col; j++) {
            result->values[i][j] = matrix1->values[i][j] - matrix2->values[i][j];
        }
    }
    return result;
}

/**
 * Reads a matrix from an input file
 * @param input_file Input file.
 * @return Matrix created from input file.
 */
Matrix_ptr create_matrix5(FILE *input_file) {
    int row, col;
    fscanf(input_file, "%d %d", &row, &col);
    Matrix_ptr result = create_matrix(row, col);
    for (int i = 0; i < row; i++){
        for (int j = 0; j < col; j++){
            fscanf(input_file, "%lf", &result->values[i][j]);
        }
    }
    return result;
}

