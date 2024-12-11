from os import system
import csv

vals = [-1.5, -1. , -0.5,  0. ,  0.5,  1. ,  1.5]
file_path = "../dataFiles/IC_expressions.txt"

def write_vals(my_string, my_float):
    # Writing a string and a float to a file
    with open("data.txt", "w") as f:
        my_float = float(my_float)
        f.write(f"{my_string},{my_float}\n")


#for val in vals:
#    system(f"./InvertedPendulum {val} 1")

with open(file_path, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        write_vals(row['expression'], row['x_0'])
        system("python InvertedPendulum.py")

#x_0,MSE,depth,expression_type,expression,orig_expression
#-1.5,6.79952e-05,7,postfix,(ln(sech(tanh(sqrt(t)))) / (ln(sqrt(1.001952)) - sqrt(cos(acos(-(sin(4))))))),x0 sqrt tanh sech ln 1.001952 sqrt ln 4 sin ~ acos cos sqrt - /
#-1,0.00221172,7,postfix,((0.000000 - 0) + (1 / ((sqrt((1 + 2)) - cos(2)) - ((10.000000 - (-0.444347 * log(t))) * sech(t))))),0.000000 0 - 1 1 2 + sqrt 2 cos - 10.000000 -0.444347 x0 log * - x0 sech * - / +
#-0.5,0.00734351,7,postfix,((sin((t * (t + 0.979767))) - (tanh(1) / (10.000000 + (10.000000 * 10.000000)))) / sech((sech(sqrt(sech(t))) + cos((sin(cos(1)) - ln(1)))))),x0 x0 0.979767 + * sin 1 tanh 10.000000 10.000000 10.000000 * + / - x0 sech sqrt sech 1 cos sin 1 ln - cos + sech /
#0,0.0407242,7,postfix,((ln(2) + (1.563600 ** (log(9.843181) / 4))) / ((0.999996 ** (t / sech(t))) + (4 + exp((exp(cos(t)) * sqrt(2)))))),2 ln 1.563600 9.843181 log 4 / ^ + 0.999996 x0 x0 sech / ^ 4 x0 cos exp 2 sqrt * exp + + /
#0.5,0.000811432,7,postfix,(ln(10.000000) ** ((cos((4 + 0.000000)) - (t ** arcsin(-(cos(0.956603))))) / 1)),10.000000 ln 4 0.000000 + cos x0 0.956603 cos ~ arcsin ^ - 1 / ^
#1,0.0381913,7,postfix,((acos(0.000000) / (4 * 1)) - (2 ** (sqrt((4 / 2)) - exp(exp(sin((t / 4))))))),0.000000 acos 4 1 * / 2 4 2 / sqrt x0 4 / sin exp exp - ^ -
#1.5,0.0573828,7,postfix,-(tanh((sech((t + log(t))) - (tanh(tanh(t)) * ln(acos(sin(t))))))),x0 x0 log + sech x0 tanh tanh x0 sin acos ln * - tanh ~

#
