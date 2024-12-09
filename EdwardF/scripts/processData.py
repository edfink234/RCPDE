from os import system
import csv

vals = [-1.5, -1. , -0.5,  0. ,  0.5,  1. ,  1.5]

def replace_xi_val(file_path, some_var):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Update the line starting with `xi_val =`
    with open(file_path, 'w') as file:
        for line in lines:
            if line.strip().startswith("xi_val ="):
                file.write(f'xi_val = `{some_var}`\n')
            else:
                file.write(line)

for val in vals:
    system(f"./InvertedPendulum {val} 1")

with open(file_path, newline='') as csvfile:
    
    

x_0,MSE,depth,expression_type,expression,orig_expression
-1.5,0.000829325,7,postfix,arcsin((sqrt(0.696405) * (-((tanh(2) ** arccos(0.000000))) * ln(cos(tanh(sqrt(t))))))),0.696405 sqrt 2 tanh 0.000000 arccos ^ ~ x0 sqrt tanh cos ln * * arcsin
-1,0.240793,7,postfix,arcsin((sin(1) * ((cos(2) * cos((t + t))) * ln(cos(tanh(sqrt(t))))))),1 sin 2 cos x0 x0 + cos * x0 sqrt tanh cos ln * * arcsin
-0.5,0.387232,7,postfix,arcsin((-(t) * (((0 * 0.000000) - (4 - (0.452101 / 1))) * tanh((sin(3.983357) / (10.000000 ** (t - 0.000000))))))),x0 ~ 0 0.000000 * 4 0.452101 1 / - - 3.983357 sin 10.000000 x0 0.000000 - ^ / tanh * * arcsin
0.5,0.0279834,7,postfix,arcsin((tanh(t) * (4 ** -(cos(sin(cos(10.000000))))))),x0 tanh 4 10.000000 cos sin cos ~ ^ * arcsin
1,0.0205642,7,postfix,(1 / ((4 + sin(t)) ** (t ** (sqrt(2) + asin((-(1) * (1 * 1.000000))))))),1 4 x0 sin + x0 2 sqrt 1 ~ 1 1.000000 * * asin + ^ ^ /
1.5,0.103473,7,postfix,tanh(sech((sech(0) + (4 / (t + sin(-(t))))))),0 sech 4 x0 x0 ~ sin + / + sech tanh
0,0.295974,7,postfix,log((tanh(acos(cos(-(sqrt(t))))) + (10.000000 ** 0.000000))),x0 sqrt ~ cos acos tanh 10.000000 0.000000 ^ + log
