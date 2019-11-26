import numpy as np

import pandas as pd

df = pd.read_csv('Data.dat', sep='\t', header=None)

df.drop([4], axis=1, inplace=True)

df.insert(0, 'X0', 1)

df.rename(columns={0: 'V', 1: 'F', 2: 'C', 3: 'M'}, inplace=True)

data = np.asarray(df)


def trans(mas, dem=2):
    if dem == 2:
        trans = list()
        for i in range(len(mas[0])):
            l = list()
            for j in range(len(mas)):
                l.append(mas[j][i])
            trans.append(l)
        return np.asarray(trans)

    else:
        trans = list()
        a = True
        for i in range(len(mas)):
            l = list()
            l.append(mas[i])
            trans.append(l)

        return np.asarray(trans)


def mul(a, b, dem=2):
    if dem == 1:
        x = [[None for y in range(len(b))] for x in range(len(a))]
        for i in range(len(a)):
            for j in range(len(b)):
                x[i][j] = a[i] * b[j]
        return x

    elif dem == 3:
        x = [0 for i in range(len(a))]
        for i in range(len(a)):
            for j in range(len(b)):
                x[i] += a[i][j] * b[j]
        return x

    elif dem == 2 and len(a[0]) == len(b):

        x = [[None for y in range(len(b[0]))] for x in range(len(a))]

        for i in range(len(a)):

            for j in range(len(b[0])):

                x[i][j] = 0

                for k in range(len(a[0])):
                    x[i][j] += a[i][k] * b[k][j]

        return x

    else:
        return 'Wrong data dimensions'


def sum_(a, b, dem=2):
    if dem == 2 and len(a) * len(a[0]) == len(b) * len(b[0]):

        sum_mas = [[None for y in range(len(a[0]))] for x in range(len(a))]

        for i in range(len(a)):

            for j in range(len(a[0])):
                sum_mas[i][j] = a[i][j] + b[i][j]

        return sum_mas

    elif dem == 1 and len(a) == len(b):

        sum_mas = [None for i in range(len(a))]

        for i in range(len(a)):
            sum_mas[i] = a[i] + b[i]

        return sum_mas

    else:
        return 'Wrong data dimensions'


def func_a(a, k):
    x = [[None for y in range(len(a))] for x in range(len(a))]

    for i in range(len(a)):
        for j in range(len(a)):
            x[i][j] = a[i][j]

    del x[k]

    for i in x:
        del i[0]

    return x


def det(a):
    sum = 0
    if len(a) == 2:

        return a[0][0] * a[1][1] - a[0][1] * a[1][0]

    elif len(a) == 3:
        return a[0][0] * a[1][1] * a[2][2] + a[0][1] * a[1][2] * a[2][0] + a[0][2] * a[1][0] * a[2][1] - a[0][2] * a[1][
            1] * \
               a[2][0] - a[1][2] * a[2][1] * a[0][0] - a[2][2] * a[0][1] * a[1][0]

    else:
        for k in range(len(a)):
            sum += ((-1) ** (k + 2)) * a[k][0] * det(func_a(a, k))

    return sum


def func_a_(a, row, col):
    x = [[0 for y in range(len(a))] for x in range(len(a))]

    for i in range(len(a)):

        for j in range(len(a)):
            x[i][j] = a[i][j]

    del x[row]

    for i in x:
        del i[col]

    return x


def t(a):
    if (len(a) == 2):

        return a[0][0] * a[1][1] - a[0][1] * a[1][0]

    elif (len(a) == 3):

        return a[0][0] * a[1][1] * a[2][2] + a[0][1] * a[1][2] * a[2][0] + a[0][2] * a[1][0] * a[2][1] - a[0][2] * a[1][
            1] * a[2][0] - a[1][2] * a[2][1] * a[0][0] - a[2][2] * a[0][1] * a[1][0]

    else:

        x = [[0 for y in range(len(a))] for x in range(len(a))]

        for i in range(len(a)):

            for j in range(len(a)):
                x[i][j] = (((-1) ** (i + j + 2)) * t(func_a_(a, i, j)))

    return x


def obr(a):
    trans_a = t(trans(a))

    det_a = det(a)

    x = [[None for y in range(len(a))] for x in range(len(a))]

    for i in range(len(a)):

        for j in range(len(a)):
            x[i][j] = trans_a[i][j] / det_a

    return x


def regression_equations(x, y, w):
    y_ans = [0 for i in range(len(x))]

    for i in range(len(x)):

        for j in range(len(x[0])):
            y_ans[i] += x[i][j] * w[j]

    sse_ = 0

    for i in range(len(x)):
        sse_ += (y[i] - y_ans[i]) ** 2

    return sse_


# var 1

# C(k)=a0 - a1 F(k-1) + a2 M(k) + a3 V(k-1)


F = [None for x in range(len(data))]

V = [None for y in range(len(data))]

for i in range(len(data)):
    F[i] = data[i][2]

    V[i] = data[i][1]

del F[-1]

del V[-1]

F = np.append(None, F)

V = np.append(None, V)

for i in range(len(data)):
    data[i][2] = F[i]

    data[i][1] = V[i]

data = np.delete(data, (0), axis=0)
x_1 = [[None for x in range(len(data))] for y in range(len(data[0]) - 1)]

x_1 = trans(x_1)

for i in range(len(x_1)):
    x_1[i][0] = data[i][0]

    x_1[i][1] = data[i][2]

    x_1[i][2] = data[i][4]

    x_1[i][3] = data[i][1]

y_1 = [None for x in range(len(data))]

for i in range(len(data)):
    y_1[i] = data[i][3]

ans_1 = mul(obr(mul(trans(x_1), x_1)), mul(trans(x_1), y_1, dem=3), dem=3)

# Var 1 end

# Var 2
# V(k)=a0 + a1 M(k-1) + a2 F(k-1) +a3 C(k)

data = np.asarray(df)

M = [None for x in range(len(data))]

F = [None for x in range(len(data))]

for i in range(len(data)):
    M[i] = data[i][4]

    F[i] = data[i][2]

del M[-1]

del F[-1]

M = np.append(None, M)

F = np.append(None, F)

for i in range(len(data)):
    data[i][4] = M[i]

    data[i][2] = F[i]

data = np.delete(data, 0, axis=0)

x_2 = [[None for x in range(len(data))] for y in range(len(data[0]) - 1)]

x_2 = trans(x_2)

for i in range(len(x_2)):
    x_2[i][0] = data[i][0]

    x_2[i][1] = data[i][4]

    x_2[i][2] = data[i][2]

    x_2[i][3] = data[i][3]

y_2 = [None for x in range(len(data))]

for i in range(len(data)):
    y_2[i] = data[i][1]

ans_2 = mul(obr(mul(trans(x_2), x_2)), mul(trans(x_2), y_2, dem=3), dem=3)

# Var 2 end

# Var 3

# M(k)=a0 + a1 C(k-1) + a2 V(k) + a3 F(k)

data = np.asarray(df)

C = [None for x in range(len(data))]

for i in range(len(data)):
    C[i] = data[i][3]

del C[-1]

C = np.append(None, C)

for i in range(len(data)):
    data[i][3] = C[i]

data = np.delete(data, (0), axis=0)

x_3 = [[None for x in range(len(data))] for y in range(len(data[0]) - 1)]

x_3 = trans(x_3)

for i in range(len(x_3)):
    x_3[i][0] = data[i][0]

    x_3[i][1] = data[i][3]

    x_3[i][2] = data[i][1]

    x_3[i][3] = data[i][2]

y_3 = [None for x in range(len(data))]

for i in range(len(data)):
    y_3[i] = data[i][4]

ans_3 = mul(obr(mul(trans(x_3), x_3)), mul(trans(x_3), y_3, dem=3), dem=3)

# Var 3 end

# Var 4

# F(k)=a0 + a1 M(k-1) + a2 V(k) + a3 C(k-1)

data = np.asarray(df)

C = [None for x in range(len(data))]

M = [None for x in range(len(data))]

for i in range(len(data)):
    C[i] = data[i][3]

    M[i] = data[i][4]

del C[-1]

del M[-1]

C = np.append(None, C)

M = np.append(None, M)

for i in range(len(data)):
    data[i][3] = C[i]

    data[i][4] = M[i]

data = np.delete(data, 0, axis=0)

x_4 = [[None for x in range(len(data))] for y in range(len(data[0]) - 1)]

x_4 = trans(x_4)

for i in range(len(x_4)):
    x_4[i][0] = data[i][0]

    x_4[i][1] = data[i][4]

    x_4[i][2] = data[i][1]

    x_4[i][3] = data[i][3]

y_4 = [None for x in range(len(data))]

for i in range(len(data)):
    y_4[i] = data[i][2]

ans_4 = mul(obr(mul(trans(x_4), x_4)), mul(trans(x_4), y_4, dem=3), dem=3)

# Var 4 end


# var 5 C(k)=a0+a1 M(k-1)+a2 F(k)+a3 V(k)

# prepare for 5 var.

data = np.asarray(df)

M = [None for x in range(len(data))]

for i in range(len(data)):
    M[i] = data[i][4]

del M[-1]

M = np.append(None, M)

for i in range(len(data)):
    data[i][4] = M[i]

data = np.delete(data, (0), axis=0)

x_5 = [[None for x in range(len(data))] for y in range(len(data[0]) - 1)]

x_5 = trans(x_5)

for i in range(len(x_5)):
    x_5[i][0] = data[i][0]

    x_5[i][1] = data[i][4]

    x_5[i][2] = data[i][2]

    x_5[i][3] = data[i][1]

y_5 = [None for x in range(len(data))]

for i in range(len(data)):
    y_5[i] = data[i][3]

ans_5 = mul(obr(mul(trans(x_5), x_5)), mul(trans(x_5), y_5, dem=3), dem=3)

# 5 var end

# Var 6

# F(k)=a0 + a1 V(k-1) + a2 C(k) + a3 M(k)

data = np.asarray(df)

V = [None for x in range(len(data))]

for i in range(len(data)):
    V[i] = data[i][1]

del V[-1]

V = np.append(None, V)

for i in range(len(data)):
    data[i][1] = V[i]

data = np.delete(data, (0), axis=0)

x_6 = [[None for x in range(len(data))] for y in range(len(data[0]) - 1)]

x_6 = trans(x_6)

for i in range(len(x_6)):
    x_6[i][0] = data[i][0]

    x_6[i][1] = data[i][1]

    x_6[i][2] = data[i][3]

    x_6[i][3] = data[i][4]

y_6 = [None for x in range(len(data))]

for i in range(len(data)):
    y_6[i] = data[i][2]

ans_6 = mul(obr(mul(trans(x_6), x_6)), mul(trans(x_6), y_6, dem=3), dem=3)

# #Var 6 end


print('Variant #1 Koef: ', ans_1)

print('RESULT FOR VARIANT #1: ', regression_equations(x_1, y_1, ans_1))
print('---------------------------------------------')

print('Variant #2 Koef: ', ans_2)

print('RESULT FOR VARIANT #2:  ', regression_equations(x_2, y_2, ans_2))
print('---------------------------------------------')

print('Variant #3 Koef: ', ans_3)

print('RESULT FOR VARIANT #3:  ', regression_equations(x_3, y_3, ans_3))
print('---------------------------------------------')

print('Variant #4 Koef: ', ans_4)

print('RESULT FOR VARIANT #4:  ', regression_equations(x_4, y_4, ans_4))
print('---------------------------------------------')

print('Variant #5 Koef: ', ans_5)

print('RESULT FOR VARIANT #5:  ', regression_equations(x_5, y_5, ans_5))
print('---------------------------------------------')

print('Variant #6 Koef: ', ans_6)

print('RESULT FOR VARIANT #6:  ', regression_equations(x_6, y_6, ans_6))
