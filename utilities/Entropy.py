import math as m

def binary_entropy(np, nn):
    p = np / (np + nn)
    return (- p * m.log(p, 2) - (1 - p) * m.log((1 - p), 2))

def split_info_gain(np, nn, np1, nn1, np2, nn2):
    # Sanity check
    if (np1 + nn1 + np2 + nn2) != (np + nn):
        print("Invalid input")
        return
    
    p1 = (np1 + nn1) / (np + nn)
    p2 = (np2 + nn2) / (np + nn)
    
    h = binary_entropy(np, nn)
    h1 = binary_entropy(np1, nn1)
    h2 = binary_entropy(np2, nn2)
    
    return h - p1 * h1 - p2 * h2

