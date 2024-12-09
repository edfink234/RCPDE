from os import system
import csv

vals = [-1.5, -1. , -0.5,  0. ,  0.5,  1. ,  1.5]
file_path = "../dataFiles/IC_expressions.txt"

def write_vals(my_string, my_float):
    # Writing a string and a float to a file
    with open("data.txt", "w") as f:
        my_float = float(my_float)
        f.write(f"{my_string},{my_float}\n")


for val in vals:
    system(f"./InvertedPendulum {val} 0")

with open(file_path, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        write_vals(row['expression'], row['x_0'])
        system("python InvertedPendulum.py")

#x_0,MSE,depth,expression_type,expression,orig_expression
#-1.5,0.000468703,7,postfix,(((tanh(2) / 2) / 0.860789) * ((2 - acos(acos(1.000000))) ** (sech(sin(((2 * t) / 2))) / t))),2 tanh 2 / 0.860789 / 2 1.000000 acos acos - 2 x0 * 2 / sin sech x0 / ^ *
#-1,0.00544415,7,postfix,(tanh(acos(arcsin(tanh(1.000000)))) / ((0.000000 + 1.000000) - ((4 * sech(tanh(t))) * (ln(10.000000) * (0.000000 + sech((0.000000 - t))))))),1.000000 tanh arcsin acos tanh 0.000000 1.000000 + 4 x0 tanh sech * 10.000000 ln 0.000000 0.000000 x0 - sech + * * - /
#-0.5,0.016826,7,postfix,(t / (t + ((t - 4) * (sech((sech(4) / t)) + asin(asin(cos(1))))))),x0 x0 x0 4 - 4 sech x0 / sech 1 cos asin asin + * + /
#0.5,0.00862833,7,postfix,(t / (arccos(tanh(cos(sqrt(4)))) * ((t + cos(arccos(1))) + (sin(asin(-(0))) * 0.000000)))),x0 4 sqrt cos tanh arccos x0 1 arccos cos + 0 ~ asin sin 0.000000 * + * /
#1,0.000978641,7,postfix,(tanh(tanh((1.000000 + (1.000000 ** (4 + 0))))) ** (log(2) + ((4 - 4) - (arcsin(sin(1.000000)) - exp(exp((10.000000 - t))))))),1.000000 1.000000 4 0 + ^ + tanh tanh 2 log 4 4 - 1.000000 sin arcsin 10.000000 x0 - exp exp - - + ^
#1.5,0.0503068,7,postfix,tanh(sech((tanh(t) + (4 / (t + ((1.007802 - 2) - sin(t))))))),x0 tanh 4 x0 1.007802 2 - x0 sin - + / + sech tanh
#0,0.211272,7,postfix,((tanh((ln((0.000000 - asin(0.000000))) * 1.000000)) / ln(10.000000)) * (sin((-(t) + acos(0.000000))) / (4 - ((t ** cos((1 + 1))) * (sech(0) + asin(sin(10.000000))))))),0.000000 0.000000 asin - ln 1.000000 * tanh 10.000000 ln / x0 ~ 0.000000 acos + sin 4 x0 1 1 + cos ^ 0 sech 10.000000 sin asin + * - / *
