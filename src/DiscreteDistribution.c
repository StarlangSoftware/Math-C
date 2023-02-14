//
// Created by Olcay Taner YILDIZ on 9.02.2023.
//

#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "DiscreteDistribution.h"

/**
 * A constructor of {@link DiscreteDistribution} class which calls its super class.
 */
Discrete_distribution_ptr create_discrete_distribution() {
    Discrete_distribution_ptr result = malloc(sizeof(Discrete_distribution));
    result->map = create_linked_hash_map((unsigned int (*)(void *, int)) hash_function_string,
                                         (int (*)(void *, void *)) compare_string);
    result->sum = 0;
    return result;
}

void free_discrete_distribution(Discrete_distribution_ptr discrete_distribution) {
    free_linked_hash_map(discrete_distribution->map, free);
}

/**
 * The addItem method takes a String item as an input and if this map contains a mapping for the item it puts the item
 * with given value + 1, else it puts item with value of 1.
 *
 * @param item string input.
 */
void add_item(Discrete_distribution_ptr discrete_distribution, char *item) {
    if (linked_hash_map_contains(discrete_distribution->map, item)){
        int* previous_value = linked_hash_map_get(discrete_distribution->map, item);
        (*previous_value)++;
        linked_hash_map_insert(discrete_distribution->map, item, previous_value);
    } else {
        int* value = malloc(sizeof(int));
        *value = 1;
        linked_hash_map_insert(discrete_distribution->map, item, value);
    }
    discrete_distribution->sum++;
}

/**
 * The removeItem method takes a String item as an input and if this map contains a mapping for the item it puts the item
 * with given value - 1, and if its value is 0, it removes the item.
 *
 * @param item String input.
 */
void remove_item(Discrete_distribution_ptr discrete_distribution, char *item) {
    if (linked_hash_map_contains(discrete_distribution->map, item)) {
        int *previous_value = linked_hash_map_get(discrete_distribution->map, item);
        (*previous_value)--;
        if (*previous_value != 0){
            linked_hash_map_insert(discrete_distribution->map, item, previous_value);
        } else {
            linked_hash_map_remove(discrete_distribution->map, item, NULL);
        }
        discrete_distribution->sum--;
    }
}

/**
 * The addDistribution method takes a {@link DiscreteDistribution} as an input and loops through the entries in this distribution
 * and if this map contains a mapping for the entry it puts the entry with its value + entry, else it puts entry with its value.
 * It also accumulates the values of entries and assigns to the sum variable.
 *
 * @param distribution {@link DiscreteDistribution} type input.
 */
void add_distribution(Discrete_distribution_ptr dst, Discrete_distribution_ptr added) {
    Array_list_ptr list = linked_hash_map_key_value_list(added->map);
    for (int i = 0; i < list->size; i++){
        Hash_node_ptr hash_node = array_list_get(list, i);
        if (linked_hash_map_contains(dst->map, hash_node->key)){
            int* previous_value = linked_hash_map_get(dst->map, hash_node->key);
            (*previous_value) += *(int*) hash_node->value;
            linked_hash_map_insert(dst->map, hash_node->key, previous_value);
        } else {
            int* value = malloc(sizeof(int));
            *value = *(int*) hash_node->value;
            linked_hash_map_insert(dst->map, hash_node->key, value);
        }
        dst->sum += *((int*) hash_node->value);
    }
}

/**
 * The removeDistribution method takes a {@link DiscreteDistribution} as an input and loops through the entries in this distribution
 * and if this map contains a mapping for the entry it puts the entry with its key - value, else it removes the entry.
 * It also decrements the value of entry from sum and assigns to the sum variable.
 *
 * @param distribution {@link DiscreteDistribution} type input.
 */
void remove_distribution(Discrete_distribution_ptr dst, Discrete_distribution_ptr removed) {
    Array_list_ptr list = linked_hash_map_key_value_list(removed->map);
    for (int i = 0; i < list->size; i++){
        Hash_node_ptr hash_node = array_list_get(list, i);
        if (linked_hash_map_contains(dst->map, hash_node->key)){
            int* previous_value = linked_hash_map_get(dst->map, hash_node->key);
            (*previous_value) -= *(int*) hash_node->value;
            if (*previous_value != 0){
                linked_hash_map_insert(dst->map, hash_node->key, previous_value);
            } else {
                linked_hash_map_remove(dst->map, hash_node->key, free);
            }
        }
        dst->sum -= *((int*) hash_node->value);
    }
}

/**
 * The getCount method takes an item as an input returns the value to which the specified item is mapped, or {@code null}
 * if this map contains no mapping for the key.
 *
 * @param item is used to search for value.
 * @return the value to which the specified item is mapped
 */
int get_count(Discrete_distribution_ptr discrete_distribution, char *item) {
    return *(int*) linked_hash_map_get(discrete_distribution->map, item);
}

/**
 * The getMaxItem method loops through the entries and gets the entry with maximum value.
 *
 * @return the entry with maximum value.
 */
char *get_max_item(Discrete_distribution_ptr discrete_distribution) {
    int max = -1;
    char* max_item = NULL;
    Node_ptr iterator = discrete_distribution->map->linked_list->head;
    while (iterator != NULL){
        Hash_node_ptr hash_node = iterator->data;
        if (*(int *)hash_node->value > max){
            max = *(int *)hash_node->value;
            max_item = (char*) hash_node->key;
        }
        iterator = iterator->next;
    }
    return max_item;
}

/**
 * Another getMaxItem method which takes an {@link vector} of Strings. It loops through the items in this {@link vector}
 * and gets the item with maximum value.
 *
 * @param includeTheseOnly {@link vector} of Strings.
 * @return the item with maximum value.
 */
char *get_max_item_include_only(Discrete_distribution_ptr discrete_distribution, Array_list_ptr include_these_only) {
    int max = -1;
    char* max_item = NULL;
    for (int i = 0; i < include_these_only->size; i++){
        char* item = array_list_get(include_these_only, i);
        int frequency = 0;
        if (linked_hash_map_contains(discrete_distribution->map, item)){
            frequency = *(int*) linked_hash_map_get(discrete_distribution->map, item);
        }
        if (frequency > max){
            max = frequency;
            max_item = item;
        }
    }
    return max_item;
}

/**
 * The getProbability method takes an item as an input returns the value to which the specified item is mapped over sum,
 * or 0.0 if this map contains no mapping for the key.
 *
 * @param item is used to search for probability.
 * @return the probability to which the specified item is mapped.
 */
double get_probability(Discrete_distribution_ptr discrete_distribution, char *item) {
    if (linked_hash_map_contains(discrete_distribution->map, item)){
        return *(int*)linked_hash_map_get(discrete_distribution->map, item) / discrete_distribution->sum;
    } else {
        return 0.0;
    }
}

/**
 * The getProbabilityLaplaceSmoothing method takes an item as an input returns the smoothed value to which the specified
 * item is mapped over sum, or 1.0 over sum if this map contains no mapping for the key.
 *
 * @param item is used to search for probability.
 * @return the smoothed probability to which the specified item is mapped.
 */
double get_probability_laplace_smoothing(Discrete_distribution_ptr discrete_distribution, char *item) {
    int size = discrete_distribution->map->hash_map->count;
    if (linked_hash_map_contains(discrete_distribution->map, item)){
        return (*(int*)linked_hash_map_get(discrete_distribution->map, item) + 1) / (discrete_distribution->sum + size + 1);
    } else {
        return 1.0 / (discrete_distribution->sum + size + 1);
    }
}

/**
 * The entropy method loops through the values and calculates the entropy of these values.
 *
 * @return entropy value.
 */
double entropy(Discrete_distribution_ptr discrete_distribution) {
    double total = 0.0, probability;
    Node_ptr iterator = discrete_distribution->map->linked_list->head;
    while (iterator != NULL){
        Hash_node_ptr hash_node = iterator->data;
        probability = *(int *)hash_node->value / discrete_distribution->sum;
        total += -probability * (log(probability) / log(2));
        iterator = iterator->next;
    }
    return total;
}

int get_index(Discrete_distribution_ptr discrete_distribution, char *item) {
    int i = 0;
    Node_ptr iterator = discrete_distribution->map->linked_list->head;
    while (iterator != NULL){
        Hash_node_ptr hash_node = iterator->data;
        if (strcmp(hash_node->key, item) == 0){
            return i;
        }
        i++;
        iterator = iterator->next;
    }
    return -1;
}

Array_list_ptr get_items(Discrete_distribution_ptr discrete_distribution) {
    Array_list_ptr result = create_array_list();
    Node_ptr iterator = discrete_distribution->map->linked_list->head;
    while (iterator != NULL){
        Hash_node_ptr hash_node = iterator->data;
        array_list_add(result, hash_node->key);
        iterator = iterator->next;
    }
    return result;
}

Hash_map_ptr get_probability_distribution(Discrete_distribution_ptr discrete_distribution) {
    Hash_map_ptr result = create_hash_map((unsigned int (*)(void *, int)) hash_function_string,
                                          (int (*)(void *, void *)) compare_string);
    Node_ptr iterator = discrete_distribution->map->linked_list->head;
    while (iterator != NULL){
        Hash_node_ptr hash_node = iterator->data;
        double* probability = malloc(sizeof(double));
        *probability = get_probability(discrete_distribution, hash_node->key);
        hash_map_insert(result, hash_node->key, probability);
        iterator = iterator->next;
    }
    return result;
}
