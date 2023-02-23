//
// Created by Olcay Taner YILDIZ on 14.02.2023.
//

#include "../src/DiscreteDistribution.h"
#include <stdio.h>
#include <string.h>

void testAddItem() {
    Discrete_distribution_ptr smallDistribution = create_discrete_distribution();
    add_item(smallDistribution, "item1");
    add_item(smallDistribution, "item2");
    add_item(smallDistribution, "item3");
    add_item(smallDistribution, "item1");
    add_item(smallDistribution, "item2");
    add_item(smallDistribution, "item1");
    if (3 != get_count(smallDistribution, "item1")) {
        printf("Error in testAddItem 1\n");
    }
    if (2 != get_count(smallDistribution, "item2")) {
        printf("Error in testAddItem 2\n");
    }
    if (1 != get_count(smallDistribution, "item3")) {
        printf("Error in testAddItem 3\n");
    }
    free_discrete_distribution(smallDistribution);
}

void testRemoveItem() {
    Discrete_distribution_ptr smallDistribution = create_discrete_distribution();
    add_item(smallDistribution, "item1");
    add_item(smallDistribution, "item2");
    add_item(smallDistribution, "item3");
    add_item(smallDistribution, "item1");
    add_item(smallDistribution, "item2");
    add_item(smallDistribution, "item1");
    remove_item(smallDistribution, "item1");
    remove_item(smallDistribution, "item2");
    remove_item(smallDistribution, "item3");
    if (2 != get_count(smallDistribution, "item1")) {
        printf("Error in testRemoveItem 1\n");
    }
    if (1 != get_count(smallDistribution, "item2")) {
        printf("Error in testRemoveItem 2\n");
    }
    free_discrete_distribution(smallDistribution);
}

void testAddDistribution() {
    Discrete_distribution_ptr smallDistribution = create_discrete_distribution();
    add_item(smallDistribution, "item1");
    add_item(smallDistribution, "item2");
    add_item(smallDistribution, "item3");
    add_item(smallDistribution, "item1");
    add_item(smallDistribution, "item2");
    add_item(smallDistribution, "item1");
    Discrete_distribution_ptr discreteDistribution = create_discrete_distribution();
    add_item(discreteDistribution, "item4");
    add_item(discreteDistribution, "item5");
    add_item(discreteDistribution, "item5");
    add_item(discreteDistribution, "item2");
    add_distribution(smallDistribution, discreteDistribution);
    if (3 != get_count(smallDistribution, "item1")) {
        printf("Error in testAddDistribution 1\n");
    }
    if (3 != get_count(smallDistribution, "item2")) {
        printf("Error in testAddDistribution 2\n");
    }
    if (1 != get_count(smallDistribution, "item3")) {
        printf("Error in testAddDistribution 3\n");
    }
    if (1 != get_count(smallDistribution, "item4")) {
        printf("Error in testAddDistribution 4\n");
    }
    if (2 != get_count(smallDistribution, "item5")) {
        printf("Error in testAddDistribution 5\n");
    }
    free_discrete_distribution(smallDistribution);
    free_discrete_distribution(discreteDistribution);
}

void testRemoveDistribution() {
    Discrete_distribution_ptr smallDistribution = create_discrete_distribution();
    add_item(smallDistribution, "item1");
    add_item(smallDistribution, "item2");
    add_item(smallDistribution, "item3");
    add_item(smallDistribution, "item1");
    add_item(smallDistribution, "item2");
    add_item(smallDistribution, "item1");
    Discrete_distribution_ptr discreteDistribution = create_discrete_distribution();
    add_item(discreteDistribution, "item1");
    add_item(discreteDistribution, "item1");
    add_item(discreteDistribution, "item2");
    remove_distribution(smallDistribution, discreteDistribution);
    if (1 != get_count(smallDistribution, "item1")) {
        printf("Error in testRemoveDistribution 1\n");
    }
    if (1 != get_count(smallDistribution, "item2")) {
        printf("Error in testRemoveDistribution 2\n");
    }
    if (1 != get_count(smallDistribution, "item3")) {
        printf("Error in testRemoveDistribution 3\n");
    }
    free_discrete_distribution(smallDistribution);
    free_discrete_distribution(discreteDistribution);
}

void testGetIndex() {
    Discrete_distribution_ptr smallDistribution = create_discrete_distribution();
    add_item(smallDistribution, "item1");
    add_item(smallDistribution, "item2");
    add_item(smallDistribution, "item3");
    add_item(smallDistribution, "item1");
    add_item(smallDistribution, "item2");
    add_item(smallDistribution, "item1");
    if (0 != get_index(smallDistribution, "item1")) {
        printf("Error in testGetIndex 1\n");
    }
    if (1 != get_index(smallDistribution, "item2")) {
        printf("Error in testGetIndex 2\n");
    }
    if (2 != get_index(smallDistribution, "item3")) {
        printf("Error in testGetIndex 3\n");
    }
    free_discrete_distribution(smallDistribution);
}

void testGetMaxItem() {
    Discrete_distribution_ptr smallDistribution = create_discrete_distribution();
    add_item(smallDistribution, "item1");
    add_item(smallDistribution, "item2");
    add_item(smallDistribution, "item3");
    add_item(smallDistribution, "item1");
    add_item(smallDistribution, "item2");
    add_item(smallDistribution, "item1");
    if (strcmp(get_max_item(smallDistribution), "item1") != 0) {
        printf("Error in testGetMaxItem\n");
    }
    free_discrete_distribution(smallDistribution);
}

void testGetProbability() {
    Discrete_distribution_ptr smallDistribution = create_discrete_distribution();
    add_item(smallDistribution, "item1");
    add_item(smallDistribution, "item2");
    add_item(smallDistribution, "item3");
    add_item(smallDistribution, "item1");
    add_item(smallDistribution, "item2");
    add_item(smallDistribution, "item1");
    if (get_probability(smallDistribution, "item1") != 1 / 2.0) {
        printf("Error in testGetProbability 1\n");
    }
    if (get_probability(smallDistribution, "item2") != 1 / 3.0) {
        printf("Error in testGetProbability 2\n");
    }
    if (get_probability(smallDistribution, "item3") != 1 / 6.0) {
        printf("Error in testGetProbability 3\n");
    }
    free_discrete_distribution(smallDistribution);
}

void testGetProbabilityLaplaceSmoothing() {
    Discrete_distribution_ptr smallDistribution = create_discrete_distribution();
    add_item(smallDistribution, "item1");
    add_item(smallDistribution, "item2");
    add_item(smallDistribution, "item3");
    add_item(smallDistribution, "item1");
    add_item(smallDistribution, "item2");
    add_item(smallDistribution, "item1");
    if (get_probability_laplace_smoothing(smallDistribution, "item1") != 0.4) {
        printf("Error in testGetProbabilityLaplaceSmoothing 1\n");
    }
    if (get_probability_laplace_smoothing(smallDistribution, "item2") != 0.3) {
        printf("Error in testGetProbabilityLaplaceSmoothing 2\n");
    }
    if (get_probability_laplace_smoothing(smallDistribution, "item3") != 0.2) {
        printf("Error in testGetProbabilityLaplaceSmoothing 3\n");
    }
    if (get_probability_laplace_smoothing(smallDistribution, "item4") != 0.1) {
        printf("Error in testGetProbabilityLaplaceSmoothing 3\n");
    }
    free_discrete_distribution(smallDistribution);
}

int main() {
    testAddItem();
    testRemoveItem();
    testAddDistribution();
    testRemoveDistribution();
    testGetIndex();
    testGetMaxItem();
    testGetProbability();
    testGetProbabilityLaplaceSmoothing();
}