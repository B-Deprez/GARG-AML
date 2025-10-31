def measure_00_function(adj_full, size_0):
    piece_00 = adj_full[:size_0, :size_0]
    total_sum_00 = piece_00.sum()
    total_size_00 = piece_00.size
    reduced_size_00 = total_size_00 - (3 * size_0) + 2

    if reduced_size_00 > 0:
        rel_00 = total_sum_00/reduced_size_00
    else:
        rel_00 = 0

    return rel_00, reduced_size_00

def measure_01_function(adj_full, size_0, size_1):
    piece_01 = adj_full[:size_0, size_0:size_0 + size_1]
    total_sum_01 = piece_01.sum()
    total_size_01 = piece_01.size

    if total_size_01 > 0:
        rel_01 = total_sum_01/total_size_01
    else:
        rel_01 = 1 #Since block only contains sure connections => full sum

    return rel_01, total_size_01

def measure_02_function(adj_full, size_0, size_1, size_2):
    piece_02 = adj_full[:size_0, size_0 + size_1:]
    total_sum_02 = piece_02.sum()
    total_size_02 = piece_02.size
    reduced_size_02 = total_size_02 - size_2

    if reduced_size_02 > 0:
        rel_02 = total_sum_02/reduced_size_02
    else:
        rel_02 = 0
    
    return rel_02, reduced_size_02

def measure_10_function(adj_full, size_0, size_1):
    piece_10 = adj_full[size_0:size_0 + size_1, :size_0]
    total_sum_10 = piece_10.sum()
    total_size_10 = piece_10.size

    if total_size_10 > 0:
        rel_10 = total_sum_10/total_size_10
    else:
        rel_10 = 0
    
    return rel_10, total_size_10

def measure_11_function(adj_full, size_0, size_1):
    piece_11 = adj_full[size_0:size_0 + size_1, size_0:size_0 + size_1]
    total_sum_11 = piece_11.sum()
    total_size_11 = piece_11.size
    reduced_size_11 = total_size_11 - size_1

    if reduced_size_11 > 0:
        rel_11 = total_sum_11/reduced_size_11
    else:
        rel_11 = 0
    
    return rel_11, reduced_size_11

def measure_12_function(adj_full, size_0, size_1, size_2):
    piece_12 = adj_full[size_0:size_0 + size_1, size_0 + size_1:]
    total_sum_12 = piece_12.sum()
    total_size_12 = piece_12.size

    if total_size_12 > 0:
        rel_12 = total_sum_12/total_size_12
    elif size_2 > 0:
        rel_12 = 1 #Since block only contains sure connections => full sum
    else:
        rel_12 = 0 #No connections at all

    return rel_12, total_size_12

def measure_20_function(adj_full, size_0, size_1, size_2):
    piece_20 = adj_full[size_0+size_1:, :size_0]
    total_sum_20 = piece_20.sum()
    total_size_20 = piece_20.size
    reduced_size_20 = total_size_20 - size_2

    if reduced_size_20 > 0:
        rel_20 = total_sum_20/reduced_size_20
    else:
        rel_20 = 0
    
    return rel_20, reduced_size_20

def measure_21_function(adj_full, size_0, size_1):
    piece_21 = adj_full[size_0+size_1:, size_0:size_0 + size_1]
    total_sum_21 = piece_21.sum()
    total_size_21 = piece_21.size

    if total_size_21 > 0:
        rel_21 = total_sum_21/total_size_21
    else:
        rel_21 = 0
    
    return rel_21, total_size_21

def measure_22_function(adj_full, size_0, size_1, size_2):
    piece_22 = adj_full[size_0+size_1:, size_0+size_1:]
    total_sum_22 = piece_22.sum()
    total_size_22 = piece_22.size
    reduced_size_22 = total_size_22 - size_2

    if reduced_size_22 > 0:
        rel_22 = total_sum_22/reduced_size_22
    else:
        rel_22 = 0
    
    return rel_22, reduced_size_22