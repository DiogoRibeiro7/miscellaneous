# Python 3.7
# Version: 1.0
# Author: Diogo Ribeiro

# Description: Implementation of several sorting algorithms such as insertion sort,
# bubble sort, shaker sort, gnome sort, merge sort and selection sort.
####################################################################################

list1 = [6, 3, 2, 89, 10, 2.3, 5, 80.2]

def insertion_sort(list):
    for i in range(1, len(list)):
        element = list[i]
        j = i
        while j > 0 and list[j-1] > element:
            list[j] = list[j - 1]
            j = j-1
        list[j] = element
    return list


def bubble_sort(list):
    permutation = True
    passage = 0
    while permutation == True:
        permutation = False
        passage = passage + 1
        for element in range(0, len(list) - passage):
            if list[element] > list[element + 1]:
                permutation = True
                list[element], list[element + 1] = list[element + 1], list[element]
    return list


def shaker_sort(list):
    permutation, way, element = True, 1, 0
    start, end = 0, len(list) - 2
    while permutation == True:
        permutation = False
        while (element < end and way == 1) or \
        (element > start and way == -1) :
            # Test si echange
            if list[element] > list[element + 1]:
                permutation = True
                # On echange les deux elements
                list[element], list[element + 1] = \
                    list[element + 1], list[element]
            element = element + way
        # On change le sens du parcours
        if way==1:
            fin = end - 1
        else:
            start = start + 1
        way = -way
    return list

def gnome_sort(list):
    i_b, i_i, size = 1,2,len(list)
    while i_b < size:
        if list[i_b-1] <= list[i_b]:
            i_b, i_i = i_i, i_i+1
        else:
            list[i_b - 1], list[i_b] = list[i_b], list[i_b - 1]
            i_b -= 1
            if i_b == 0:
                i_b, i_i = i_i, i_i+1
    return list

def merge(left, right):
    result = []
    left_index, right_index = 0, 0
    while left_index < len(left) and right_index < len(right):
        if left[left_index] <= right[right_index]:
            result.append(left[left_index])
            left_index += 1
        else:
            result.append(right[right_index])
            right_index += 1
    if left:
        result.extend(left[left_index:])
    if right:
        result.extend(right[right_index:])
    return result

def merge_sort(x):
    if len(x) <= 1:
        return x
    middle = len(x) // 2
    left = x[:middle]
    right = x[middle:]
    left = merge_sort(left)
    right = merge_sort(right)
    return list(merge(left, right))

def selection_sort(list):
    nb = len(list)
    for element in range(0,nb):
        min = element
        for j in range(element+1,nb) :
            if list[j] < list[min] :
                min = j
        if min is not element :
            temp = list[element]
            list[element] = list[min]
            list[min] = temp
    return list


print(insertion_sort(list1))
print(bubble_sort(list1))
print(shaker_sort(list1))
print(gnome_sort(list1))
print(merge_sort(list1))
print(selection_sort(list1))