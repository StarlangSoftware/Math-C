//
// Created by Olcay Taner YILDIZ on 14.02.2023.
//

#include "../src/Vector.h"
#include <stdio.h>
#include <math.h>

void testBiased(){
    double data1[] = {2, 3, 4, 5, 6};
    Vector_ptr smallVector1 = create_vector4(data1, 5);
    Vector_ptr biased_vector = biased(smallVector1);
    if (1 != get_value(biased_vector, 0)){
        printf("Error in testBiased 1\n");
    }
    if (smallVector1->size + 1 != biased_vector->size){
        printf("Error in testBiased 2\n");
    }
    free_vector(smallVector1);
    free_vector(biased_vector);
}

void testElementAdd(){
    double data1[] = {2, 3, 4, 5, 6};
    Vector_ptr smallVector1 = create_vector4(data1, 5);
    add_value_to_vector(smallVector1, 7);
    if (7 != get_value(smallVector1, 5)){
        printf("Error in testElementAdd 1\n");
    }
    if (6 != smallVector1->size){
        printf("Error in testElementAdd 2\n");
    }
    free_vector(smallVector1);
}

void testInsert(){
    double data1[] = {2, 3, 4, 5, 6};
    Vector_ptr smallVector1 = create_vector4(data1, 5);
    insert_into_pos(smallVector1, 3, 6);
    if (6 != get_value(smallVector1, 3)){
        printf("Error in testInsert 1\n");
    }
    if (6 != smallVector1->size){
        printf("Error in testInsert 2\n");
    }
    free_vector(smallVector1);
}

void testRemove(){
    double data1[] = {2, 3, 4, 5, 6};
    Vector_ptr smallVector1 = create_vector4(data1, 5);
    remove_at_pos(smallVector1, 2);
    if (5 != get_value(smallVector1, 2)){
        printf("Error in testRemove 1\n");
    }
    if (4 != smallVector1->size){
        printf("Error in testRemove 2\n");
    }
    free_vector(smallVector1);
}

void testSumOfElements(){
    double data1[] = {2, 3, 4, 5, 6};
    double data2[] = {8, 7, 6, 5, 4};
    Vector_ptr smallVector1 = create_vector4(data1, 5);
    Vector_ptr smallVector2 = create_vector4(data2, 5);
    if (20 != sum_of_elements_of_vector(smallVector1)){
        printf("Error in testSumOfElementsSmall 1\n");
    }
    if (30 != sum_of_elements_of_vector(smallVector2)){
        printf("Error in testSumOfElementsSmall 2\n");
    }
    free_vector(smallVector1);
    free_vector(smallVector2);
}

void testMaxIndex(){
    double data1[] = {2, 3, 4, 5, 6};
    double data2[] = {8, 7, 6, 5, 4};
    Vector_ptr smallVector1 = create_vector4(data1, 5);
    Vector_ptr smallVector2 = create_vector4(data2, 5);
    if (4 != max_index_of_vector(smallVector1)){
        printf("Error in testMaxIndex 1\n");
    }
    if (0 != max_index_of_vector(smallVector2)){
        printf("Error in testMaxIndex 2\n");
    }
    free_vector(smallVector1);
    free_vector(smallVector2);
}

void testSkipVector(){
    double data1[] = {2, 3, 4, 5, 6};
    Vector_ptr smallVector1 = create_vector4(data1, 5);
    Vector_ptr smallVector3 = skip_vector(smallVector1, 2, 0);
    if (2 != get_value(smallVector3, 0)){
        printf("Error in testSkipVector 1\n");
    }
    if (6 != get_value(smallVector3, 2)){
        printf("Error in testSkipVector 2\n");
    }
    free_vector(smallVector3);
    smallVector3 = skip_vector(smallVector1, 3, 1);
    if (3 != get_value(smallVector3, 0)){
        printf("Error in testSkipVector 3\n");
    }
    if (6 != get_value(smallVector3, 1)){
        printf("Error in testSkipVector 4\n");
    }
    free_vector(smallVector1);
    free_vector(smallVector3);
}

void testVectorAdd(){
    double data1[] = {2, 3, 4, 5, 6};
    double data2[] = {8, 7, 6, 5, 4};
    Vector_ptr smallVector1 = create_vector4(data1, 5);
    Vector_ptr smallVector2 = create_vector4(data2, 5);
    add_vector(smallVector1, smallVector2);
    if (50 != sum_of_elements_of_vector(smallVector1)){
        printf("Error in testVectorAdd\n");
    }
    free_vector(smallVector1);
    free_vector(smallVector2);
}

void testSubtract(){
    double data1[] = {2, 3, 4, 5, 6};
    double data2[] = {8, 7, 6, 5, 4};
    Vector_ptr smallVector1 = create_vector4(data1, 5);
    Vector_ptr smallVector2 = create_vector4(data2, 5);
    subtract_vector(smallVector1, smallVector2);
    if (-10 != sum_of_elements_of_vector(smallVector1)){
        printf("Error in testSubtract\n");
    }
    free_vector(smallVector1);
    free_vector(smallVector2);
}

void testDotProductWithVector(){
    double data1[] = {2, 3, 4, 5, 6};
    double data2[] = {8, 7, 6, 5, 4};
    Vector_ptr smallVector1 = create_vector4(data1, 5);
    Vector_ptr smallVector2 = create_vector4(data2, 5);
    double dotProduct = dot_product(smallVector1, smallVector2);
    if (110 != dotProduct){
        printf("Error in testDotProductWithVector\n");
    }
    free_vector(smallVector1);
    free_vector(smallVector2);
}

void testDotProductWithItself(){
    double data1[] = {2, 3, 4, 5, 6};
    Vector_ptr smallVector1 = create_vector4(data1, 5);
    double dotProduct = dot_product_with_itself(smallVector1);
    if (90 != dotProduct){
        printf("Error in testDotProductWithItself\n");
    }
    free_vector(smallVector1);
}

void testElementProduct(){
    double data1[] = {2, 3, 4, 5, 6};
    double data2[] = {8, 7, 6, 5, 4};
    Vector_ptr smallVector1 = create_vector4(data1, 5);
    Vector_ptr smallVector2 = create_vector4(data2, 5);
    Vector_ptr smallVector3 = element_product_with_vector(smallVector1, smallVector2);
    if (110 != sum_of_elements_of_vector(smallVector3)){
        printf("Error in testDotProductWithVector\n");
    }
    free_vector(smallVector1);
    free_vector(smallVector2);
    free_vector(smallVector3);
}

void testDivide(){
    double data1[] = {2, 3, 4, 5, 6};
    Vector_ptr smallVector1 = create_vector4(data1, 5);
    divide_to_value(smallVector1,10);
    if (2 != sum_of_elements_of_vector(smallVector1)){
        printf("Error in testDivide\n");
    }
    free_vector(smallVector1);
}

void testMultiply(){
    double data1[] = {2, 3, 4, 5, 6};
    Vector_ptr smallVector1 = create_vector4(data1, 5);
    multiply_with_value(smallVector1,10);
    if (200 != sum_of_elements_of_vector(smallVector1)){
        printf("Error in testDivide\n");
    }
    free_vector(smallVector1);
}

void testL1Normalize(){
    double data1[] = {2, 3, 4, 5, 6};
    Vector_ptr smallVector1 = create_vector4(data1, 5);
    l1_normalize(smallVector1);
    if (1 != sum_of_elements_of_vector(smallVector1)){
        printf("Error in testL1Normalize\n");
    }
    free_vector(smallVector1);
}

void testL2Norm(){
    double data1[] = {2, 3, 4, 5, 6};
    Vector_ptr smallVector1 = create_vector4(data1, 5);
    double norm = l2_norm(smallVector1);
    if (sqrt(90) != norm){
        printf("Error in testL2Norm\n");
    }
    free_vector(smallVector1);
}

int main(){
    testBiased();
    testElementAdd();
    testInsert();
    testRemove();
    testSumOfElements();
    testMaxIndex();
    testSkipVector();
    testVectorAdd();
    testSubtract();
    testDotProductWithVector();
    testDotProductWithItself();
    testElementProduct();
    testDivide();
    testMultiply();
    testL1Normalize();
    testL2Norm();
}