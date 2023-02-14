//
// Created by Olcay Taner YILDIZ on 14.02.2023.
//

#include "../src/Matrix.h"
#include <stdio.h>

void testColumnWiseNormalize(){
    Matrix_ptr small = create_matrix(3, 3);
    for (int i = 0; i < 3; i++){
        for (int j = 0; j < 3; j++){
            small->values[i][j] = 1.0;
        }
    }
    Matrix_ptr large = create_matrix(1000, 1000);
    for (int i = 0; i < 1000; i++){
        for (int j = 0; j < 1000; j++){
            large->values[i][j] = 1.0;
        }
    }
    Matrix_ptr identity = create_matrix3(100);
    column_wise_normalize(small);
    if (3 != sum_of_elements_of_matrix(small)){
        printf("Error in testColumnWiseNormalize 1\n");
    }
    column_wise_normalize(large);
    if (1000 != sum_of_elements_of_matrix(large)){
        printf("Error in testColumnWiseNormalize 2\n");
    }
    column_wise_normalize(identity);
    if (100 != sum_of_elements_of_matrix(identity)){
        printf("Error in testColumnWiseNormalize 3\n");
    }
    free_matrix(small);
    free_matrix(large);
    free_matrix(identity);
}

void testMultiplyWithConstant(){
    Matrix_ptr small = create_matrix(3, 3);
    for (int i = 0; i < 3; i++){
        for (int j = 0; j < 3; j++){
            small->values[i][j] = 1.0;
        }
    }
    Matrix_ptr large = create_matrix(1000, 1000);
    for (int i = 0; i < 1000; i++){
        for (int j = 0; j < 1000; j++){
            large->values[i][j] = 1.0;
        }
    }
    Matrix_ptr random = create_matrix2(100, 100, 1, 10);
    double originalSum = sum_of_elements_of_matrix(random);
    multiply_with_constant(small, 4);
    if (36 != sum_of_elements_of_matrix(small)){
        printf("Error in testMultiplyWithConstant 1\n");
    }
    multiply_with_constant(large, 1.001);
    if (1001000 != sum_of_elements_of_matrix(large)){
        printf("Error in testMultiplyWithConstant 2\n");
    }
    multiply_with_constant(random, 3.6);
    if (originalSum * 3.6 != sum_of_elements_of_matrix(random)){
        printf("Error in testMultiplyWithConstant 3\n");
    }
    free_matrix(small);
    free_matrix(large);
    free_matrix(random);
}

void testDivideByConstant(){
    Matrix_ptr small = create_matrix(3, 3);
    for (int i = 0; i < 3; i++){
        for (int j = 0; j < 3; j++){
            small->values[i][j] = 1.0;
        }
    }
    Matrix_ptr large = create_matrix(1000, 1000);
    for (int i = 0; i < 1000; i++){
        for (int j = 0; j < 1000; j++){
            large->values[i][j] = 1.0;
        }
    }
    Matrix_ptr random = create_matrix2(100, 100, 1, 10);
    double originalSum = sum_of_elements_of_matrix(random);
    divide_by_constant(small, 4);
    if (2.25 != sum_of_elements_of_matrix(small)){
        printf("Error in testDivideByConstant 1\n");
    }
    divide_by_constant(large, 10);
    if (100000 != sum_of_elements_of_matrix(large)){
        printf("Error in testDivideByConstant 2\n");
    }
    divide_by_constant(random, 3.6);
    if (originalSum / 3.6 != sum_of_elements_of_matrix(random)){
        printf("Error in testDivideByConstant 3\n");
    }
    free_matrix(small);
    free_matrix(large);
    free_matrix(random);
}

int main(){
    testColumnWiseNormalize();
    testMultiplyWithConstant();
    testDivideByConstant();
}