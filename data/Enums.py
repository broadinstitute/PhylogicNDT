# from enum import Enum, unique #requires >2.7.6 or 3.x

def Enum(**enums):
    return type('Enum', (), enums)


Cluster = Enum(**dict([('C1', 'green'), ('C2', 'blue'), ('C3', 'red')]))

# csize contains chromosome bp lengths
CSIZE = [249250621, 243199373, 198022430, 191154276, 180915260, 171115067, 159138663, 146364022, 141213431,
         135534747, 135006516, 133851895, 115169878, 107349540, 102531392, 90354753, 81195210, 78077248, 59128983,
         63025520, 48129895, 51304566, 156040895, 57227415]

chrom_dict = {'chr' + str(c): s for c, s in zip(list(range(1, 23)) + ['X', 'Y'], CSIZE)}
ChromSize = Enum(**chrom_dict)

# centromeres (define arm-level lengths)
CENT_LOOKUP = {1: 125000000, 2: 93300000, 3: 91000000, 4: 50400000, 5: 48400000,
               6: 61000000, 7: 59900000, 8: 45600000, 9: 49000000, 10: 40200000,
               11: 53700000, 12: 35800000, 13: 17900000, 14: 17600000, 15: 19000000,
               16: 36600000, 17: 24000000, 18: 17200000, 19: 26500000, 20: 27500000, 21: 13200000,
               22: 14700000, 23: 60600000, 24: 12500000}

MutStatus = Enum(OK="OK",
                 REMOVED="REMOVED",  # blacklisted
                 GRAYLIST="GRAYLIST")  # not used in clustering

MutType = Enum(INS="INS",
               DEL="DEL",
               SNV="SNV")
