//
// Created by Olcay Taner YILDIZ on 13.02.2023.
//

#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "Matrix.h"
#include "Eigenvector.h"

/**
 * Another constructor of {@link Matrix} class which takes row and column numbers as inputs and creates new values
 * {@link array} with given parameters.
 *
 * @param row is used to create matrix.
 * @param col is used to create matrix.
 */
Matrix_ptr create_matrix(int row, int col) {
    Matrix_ptr result = malloc(sizeof(Matrix));
    allocate_matrix(result, row, col);
    return result;
}

/**
 * Another constructor of {@link Matrix} class which takes row, column, minimum and maximum values as inputs.
 * First it creates new values {@link array} with given row and column numbers. Then fills in the
 * positions with random numbers using minimum and maximum inputs.
 *
 * @param row is used to create matrix.
 * @param col is used to create matrix.
 * @param min minimum value.
 * @param max maximum value.
 */
Matrix_ptr create_matrix2(int row, int col, double min, double max) {
    Matrix_ptr result = malloc(sizeof(Matrix));
    allocate_matrix(result, row, col);
    for (int i = 0; i < row; i++){
        for (int j = 0; j < col; j++){
            result->values[i][j] = min + (double) random()/(double)(RAND_MAX/ (max - min));
        }
    }
    return result;
}

/**
 * Another constructor of {@link Matrix} class which takes size as input and creates new values {@link array}
 * with using size input and assigns 1 to each element at the diagonal.
 *
 * @param size is used declaring the size of the array.
 */
Matrix_ptr create_matrix3(int size) {
    Matrix_ptr result = malloc(sizeof(Matrix));
    allocate_matrix(result, size, size);
    for (int i = 0; i < size; i++){
        result->values[i][i] = 1;
    }
    return result;
}

Matrix_ptr create_matrix4(Vector_ptr vector1, Vector_ptr vector2) {
    Matrix_ptr result = malloc(sizeof(Matrix));
    allocate_matrix(result, vector1->size, vector2->size);
    for (int i = 0; i < result->row; i++){
        for (int j = 0; j < result->col; j++){
            result->values[i][j] = get_value(vector1, i) * get_value(vector2, j);
        }
    }
    return result;
}

Matrix_ptr clone(Matrix_ptr matrix) {
    Matrix_ptr result = malloc(sizeof(Matrix));
    allocate_matrix(result, matrix->row, matrix->col);
    for (int i = 0; i < matrix->row; i++){
        for (int j = 0; j < matrix->col; j++){
            result->values[i][j] = matrix->values[i][j];
        }
    }
    return result;
}

void allocate_matrix(Matrix_ptr matrix, int row, int col) {
    matrix->row = row;
    matrix->col = col;
    matrix->values = malloc(matrix->row * sizeof(double*));
    for (int i = 0; i < matrix->row; i++) {
        matrix->values[i] = calloc(matrix->col, sizeof(double));
    }
}

void free_matrix(Matrix_ptr matrix) {
    for (int i = 0; i < matrix->row; i++) {
        free(matrix->values[i]);
    }
    free(matrix->values);
    free(matrix);
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
 * The increment method adds 1 to the item at given index of values {@link array}.
 *
 * @param rowNo integer input for row number.
 * @param colNo integer input for column number.
 */
void increment(Matrix_ptr matrix, int rowNo, int colNo) {
    matrix->values[rowNo][colNo]++;
}

/**
 * The getRow method returns the vector of values {@link array} at given row input.
 *
 * @param _row integer input for row number.
 * @return Vector of values {@link array} at given row input.
 */
Vector_ptr get_row(Matrix_ptr matrix, int row) {
    return create_vector4(matrix->values[row], matrix->col);
}

/**
 * The getColumn method creates an {@link vector} and adds items at given column number of values {@link array}
 * to the {@link vector}.
 *
 * @param column integer input for column number.
 * @return Vector of given column number.
 */
Array_list_ptr get_column(Matrix_ptr matrix, int column) {
    Array_list_ptr vector;
    vector = create_array_list();
    for (int i = 0; i < matrix->row; i++){
        double* x;
        x = malloc(sizeof(double));
        *x = matrix->values[i][column];
        array_list_add(vector, x);
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
 * The multiplyWithConstant method takes a constant as an input and multiplies each item of values {@link array}
 * with given constant.
 *
 * @param constant value to multiply items of values {@link array}.
 */
void multiply_with_constant(Matrix_ptr matrix, double constant) {
    int i;
    for (i = 0; i < matrix->row; i++) {
        for (int j = 0; j < matrix->col; j++){
            matrix->values[i][j] *= constant;
        }
    }
}

/**
 * The divideByConstant method takes a constant as an input and divides each item of values {@link array}
 * with given constant.
 *
 * @param constant value to divide items of values {@link array}.
 */
void divide_by_constant(Matrix_ptr matrix, double constant) {
    int i;
    for (i = 0; i < matrix->row; i++) {
        for (int j = 0; j < matrix->col; j++){
            matrix->values[i][j] /= constant;
        }
    }
}

/**
 * The add method takes a {@link Matrix} as an input and accumulates values {@link array} with the
 * corresponding items of given Matrix. If the sizes of both Matrix and values {@link array} do not match,
 * it throws {@link MatrixDimensionMismatch} exception.
 *
 * @param m Matrix type input.
 */
void add_matrix(Matrix_ptr matrix1, Matrix_ptr matrix2) {
    for (int i = 0; i < matrix1->row; i++) {
        for (int j = 0; j < matrix1->col; j++){
            matrix1->values[i][j] += matrix2->values[i][j];
        }
    }
}

/**
 * The add method which takes a row number and a Vector as inputs. It sums up the corresponding values at the given row of
 * values {@link array} and given {@link Vector}. If the sizes of both Matrix and values
 * {@link array} do not match, it throws {@link MatrixColumnMismatch} exception.
 *
 * @param rowNo integer input for row number.
 * @param v     Vector type input.
 */
void add_vector_to_matrix(Matrix_ptr matrix, int row, Vector_ptr v) {
    for (int i = 0; i < matrix->col; i++){
        matrix->values[row][i] += get_value(v, i);
    }
}

/**
 * The subtract method takes a {@link Matrix} as an input and subtracts from values {@link array} the
 * corresponding items of given Matrix. If the sizes of both Matrix and values {@link aArray} do not match,
 * it throws {@link MatrixDimensionMismatch} exception.
 *
 * @param m Matrix type input.
 */
void subtract_matrix(Matrix_ptr matrix1, Matrix_ptr matrix2) {
    for (int i = 0; i < matrix1->row; i++) {
        for (int j = 0; j < matrix1->col; j++){
            matrix1->values[i][j] -= matrix2->values[i][j];
        }
    }
}

/**
 * The multiplyWithVectorFromLeft method takes a Vector as an input and creates a result {@link array}.
 * Then, multiplies values of input Vector starting from the left side with the values {@link array},
 * accumulates the multiplication, and assigns to the result {@link array}. If the sizes of both Vector
 * and row number do not match, it throws {@link MatrixRowMismatch} exception.
 *
 * @param v {@link Vector} type input.
 * @return Vector that holds the result.
 */
Vector_ptr multiply_with_vector_from_left(Matrix_ptr matrix, Vector_ptr vector) {
    double* result = malloc(matrix->col * sizeof(double));
    for (int i = 0; i < matrix->col; i++) {
        result[i] = 0.0;
        for (int j = 0; j < matrix->row; j++) {
            result[i] += get_value(vector, j) * matrix->values[j][i];
        }
    }
    return create_vector4(result, matrix->col);
}

/**
 * The multiplyWithVectorFromRight method takes a Vector as an input and creates a result {@link array}.
 * Then, multiplies values of input Vector starting from the right side with the values {@link array},
 * accumulates the multiplication, and assigns to the result {@link array}. If the sizes of both Vector
 * and row number do not match, it throws {@link MatrixColumnMismatch} exception.
 *
 * @param v {@link Vector} type input.
 * @return Vector that holds the result.
 */
Vector_ptr multiply_with_vector_from_right(Matrix_ptr matrix, Vector_ptr vector) {
    double* result = malloc(matrix->row * sizeof(double));
    for (int i = 0; i < matrix->row; i++) {
        result[i] = 0;
        for (int j = 0; j < matrix->col; j++){
            result[i] += matrix->values[i][j] * get_value(vector, j);
        }
    }
    return create_vector4(result, matrix->row);
}

/**
 * The columnSum method takes a column number as an input and accumulates items at given column number of values
 * {@link array}.
 *
 * @param columnNo Column number input.
 * @return summation of given column of values {@link array}.
 */
double column_sum(Matrix_ptr matrix, int columnNo) {
    double sum = 0;
    for (int i = 0; i < matrix->row; i++) {
        sum += matrix->values[i][columnNo];
    }
    return sum;
}

/**
 * The sumOfRows method creates a mew result {@link Vector} and adds the result of columnDum method's corresponding
 * index to the newly created result {@link Vector}.
 *
 * @return Vector that holds column sum.
 */
Vector_ptr sum_of_rows(Matrix_ptr matrix) {
    double* result = malloc(matrix->col * sizeof(double));
    for (int i = 0; i < matrix->col; i++) {
        result[i] = column_sum(matrix, i);
    }
    return create_vector4(result, matrix->col);
}

/**
 * The rowSum method takes a row number as an input and accumulates items at given row number of values
 * {@link array}.
 *
 * @param rowNo Row number input.
 * @return summation of given row of values {@link array}.
 */
double row_sum(Matrix_ptr matrix, int row) {
    double sum = 0;
    for (int i = 0; i < matrix->col; i++){
        sum += matrix->values[matrix->row][i];
    }
    return sum;
}

/**
 * The multiply method takes a {@link Matrix} as an input. First it creates a result {@link Matrix} and puts the
 * accumulatated multiplication of values {@link array} and given {@link Matrix} into result
 * {@link Matrix}. If the size of Matrix's row size and values {@link array}'s column size do not match,
 * it throws {@link MatrixRowColumnMismatch} exception.
 *
 * @param m Matrix type input.
 * @return result {@link Matrix}.
 */
Matrix_ptr multiply_with_matrix(Matrix_ptr matrix1, Matrix_ptr matrix2) {
    double sum;
    Matrix_ptr result = malloc(sizeof(Matrix));
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
 * The elementProduct method takes a {@link Matrix} as an input and performs element wise multiplication. Puts result
 * to the newly created Matrix. If the size of Matrix's row and column size does not match with the values
 * {@link array}'s row and column size, it throws {@link MatrixDimensionMismatch} exception.
 *
 * @param m Matrix type input.
 * @return result {@link Matrix}.
 */
Matrix_ptr element_product_with_matrix(Matrix_ptr matrix1, Matrix_ptr matrix2) {
    Matrix_ptr result = malloc(sizeof(Matrix));
    allocate_matrix(result, matrix1->row, matrix1->col);
    for (int i = 0; i < matrix1->row; i++) {
        for (int j = 0; j < matrix2->col; j++){
            result->values[i][j] = matrix1->values[i][j] * matrix2->values[i][j];
        }
    }
    return result;
}

/**
 * The sumOfElements method accumulates all the items in values {@link array} and
 * returns this summation.
 *
 * @return sum of the items of values {@link array}.
 */
double sum_of_elements_of_matrix(Matrix_ptr matrix) {
    double sum = 0.0;
    for (int i = 0; i < matrix->row; i++) {
        for (int j = 0; j < matrix->col; j++){
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
double trace(Matrix_ptr matrix) {
    double sum = 0.0;
    for (int i = 0; i < matrix->row; i++) {
        sum += matrix->values[i][i];
    }
    return sum;
}

/**
 * The transpose method creates a new {@link Matrix}, then takes the transpose of values {@link array}
 * and puts transposition to the {@link Matrix}.
 *
 * @return Matrix type output.
 */
Matrix_ptr transpose(Matrix_ptr matrix) {
    Matrix_ptr result = malloc(sizeof(Matrix));
    allocate_matrix(result, matrix->col, matrix->row);
    for (int i = 0; i < matrix->row; i++) {
        for (int j = 0; j < matrix->col; j++) {
            result->values[j][i] = matrix->values[i][j];
        }
    }
    return result;
}

/**
 * The partial method takes 4 integer inputs; rowstart, rowend, colstart, colend and creates a {@link Matrix} size of
 * rowend - rowstart + 1 x colend - colstart + 1. Then, puts corresponding items of values {@link array}
 * to the new result {@link Matrix}.
 *
 * @param rowstart integer input for defining starting index of row.
 * @param rowend   integer input for defining ending index of row.
 * @param colstart integer input for defining starting index of column.
 * @param colend   integer input for defining ending index of column.
 * @return result Matrix.
 */
Matrix_ptr partial(Matrix_ptr matrix, int rowstart, int rowend, int colstart, int colend) {
    Matrix_ptr result = malloc(sizeof(Matrix));
    allocate_matrix(matrix, rowend - rowstart + 1, colend - colstart + 1);
    for (int i = rowstart; i <= rowend; i++)
        for (int j = colstart; j <= colend; j++)
            result->values[i - rowstart][j - colstart] = matrix->values[i][j];
    return result;
}

/**
 * The isSymmetric method compares each item of values {@link array} at positions (i, j) with (j, i)
 * and returns true if they are equal, false otherwise.
 *
 * @return true if items are equal, false otherwise.
 */
int is_symmetric(Matrix_ptr matrix) {
    for (int i = 0; i < matrix->row - 1; i++) {
        for (int j = i + 1; j < matrix->row; j++) {
            if (matrix->values[i][j] != matrix->values[j][i]) {
                return 0;
            }
        }
    }
    return 1;
}

/**
 * The determinant method first creates a new {@link array}, and copies the items of  values
 * {@link array} into new {@link array}. Then, calculates the determinant of this
 * new {@link array}.
 *
 * @return determinant of values {@link array}.
 */
double determinant(Matrix_ptr matrix) {
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
 * The inverse method finds the inverse of values {@link array}.
 */
void inverse(Matrix_ptr matrix) {
    double big;
    double dum, pivinv;
    int i, icol, irow, k, l, ll;
    Matrix_ptr b = create_matrix3(matrix->row);
    int* indxc, *indxr, *ipiv;
    indxc = calloc(matrix->row, sizeof (int));
    indxr = calloc(matrix->row, sizeof (int));
    ipiv = calloc(matrix->row, sizeof (int));
    for (i = 1; i <= matrix->row; i++) {
        big = 0.0;
        irow = -1;
        icol = -1;
        for (int j = 1; j <= matrix->row; j++){
            if (ipiv[j - 1] != 1){
                for (k = 1; k <= matrix->row; k++){
                    if (ipiv[k - 1] == 0){
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
            double* dummy = malloc(matrix->col * sizeof(double));
            memcpy(dummy, matrix->values[irow - 1], matrix->col * sizeof(double));
            memcpy(matrix->values[irow - 1], matrix->values[icol - 1], matrix->col * sizeof(double));
            memcpy(matrix->values[icol - 1], dummy, matrix->col * sizeof(double));
            memcpy(dummy, b->values[irow - 1], matrix->col * sizeof(double));
            memcpy(b->values[irow - 1], b->values[icol - 1], matrix->col * sizeof(double));
            memcpy(b->values[icol - 1], dummy, matrix->col * sizeof(double));
            free(dummy);
        }
        indxr[i - 1] = irow;
        indxc[i - 1] = icol;
        pivinv = (1.0) / (matrix->values[icol - 1][icol - 1]);
        matrix->values[icol - 1][icol - 1] = 1.0;
        for (int j = 0; j < matrix->col; j++){
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
    for (l = matrix->row; l >= 1; l--){
        if (indxr[l - 1] != indxc[l - 1]){
            for (k = 1; k <= matrix->row; k++) {
                double tmp = matrix->values[k - 1][indxr[l - 1] - 1];
                matrix->values[k - 1][indxr[l - 1] - 1] = matrix->values[k - 1][indxc[l - 1] - 1];
                matrix->values[k - 1][indxc[l - 1] - 1] = tmp;
            }
        }
    }
    free(indxc);
    free(indxr);
    free(ipiv);
}

/**
 * The choleskyDecomposition method creates a new {@link Matrix} and puts the Cholesky Decomposition of values Array
 * into this {@link Matrix}. Also, it throws {@link MatrixNotSymmetric} exception if it is not symmetric and
 * {@link MatrixNotPositiveDefinite} exception if the summation is negative.
 *
 * @return Matrix type output.
 */
Matrix_ptr cholesky_decomposition(Matrix_ptr matrix) {
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
 * The rotate method rotates values {@link array} according to given inputs.
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
 * The characteristics method finds and returns a sorted {@link vector} of {@link Eigenvector}s. And it throws
 * {@link MatrixNotSymmetric} exception if it is not symmetric.
 *
 * @return a sorted {@link vector} of {@link Eigenvector}s.
 */
Array_list_ptr characteristics(Matrix_ptr matrix) {
    int j, iq, ip, i;
    double threshold, theta, tau, t, sm, s, h, g, c;
    Matrix_ptr matrix1 = clone(matrix);
    Matrix_ptr v;
    v = create_matrix3(matrix->row);
    double* d = malloc(matrix->row * sizeof(double));
    double* b = malloc(matrix->row * sizeof(double));
    double* z = malloc(matrix->row * sizeof(double));
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
    free(d);
    free(b);
    free(z);
    return result;
}

Matrix_ptr sum_matrix(Matrix_ptr matrix1, Matrix_ptr matrix2) {
    Matrix_ptr result = create_matrix(matrix1->row, matrix1->col);
    for (int i = 0; i < matrix1->row; i++) {
        for (int j = 0; j < matrix1->col; j++){
            result->values[i][j] = matrix1->values[i][j] + matrix2->values[i][j];
        }
    }
    return result;
}

Matrix_ptr difference_matrix(Matrix_ptr matrix1, Matrix_ptr matrix2) {
    Matrix_ptr result = create_matrix(matrix1->row, matrix1->col);
    for (int i = 0; i < matrix1->row; i++) {
        for (int j = 0; j < matrix1->col; j++){
            result->values[i][j] = matrix1->values[i][j] - matrix2->values[i][j];
        }
    }
    return result;
}

