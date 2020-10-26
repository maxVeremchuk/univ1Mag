import re
import operator
import numpy
import matplotlib.pyplot as plt
from math import sin, cos, tan, exp, sqrt

functions = ["sin", "cos", "tg", "abs", "exp", "pow", "sqrt"]
operators = ["*", "/", "+", "-"]
constants = ["E", "PI"]
constant_values = ["2.7", "3.14"]
ops = {
    '+': operator.add,
    '-': operator.sub,
    '*': operator.mul,
}
funcs = {'sin': sin, 'cos': cos, 'tg': tan, 'exp': exp, 'abs': abs}


def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def tokenize(formula):
    return re.findall(r"(\b\w*[\.]?\w+\b|[\(\)\+\*\-\/])", formula)


def parse_formula_to_rpn(tokens):
    output_queue = []
    stack = []
    for token in tokens:
        if is_float(token) or token in constants or token == "x":
            output_queue.append(token)
        elif token in functions or token == "(":
            stack.append(token)
        elif token in operators:
            while (len(stack) and
                   ((stack[-1] in functions) or ((stack[-1] in operators) and ((int)(operators.index(token) / 2) >= (int)(operators.index(stack[-1]) / 2)))) and
                   (stack[-1] != "(")):
                output_queue.append(stack.pop())
            stack.append(token)
        elif token == ")":
            while len(stack) and stack[-1] != "(":
                output_queue.append(stack.pop())
            if len(stack):
                stack.pop()  # pop left parentheses
            else:
                raise Exception("Wrong right parentheses in formula")
        else:
            raise Exception("Wrong function name")
    while len(stack):
        output_queue.append(stack.pop())
    return output_queue


def make_tabulation(formula_tokens, x1, x2, step=0.01):
    values = []
    for x in numpy.linspace(x1, x2, int(1 / step)):
        formula = list(formula_tokens)
        for i, item in enumerate(formula):
            if item == "x":
                formula[i] = str(x)
        values.append(calclulate_rpn(formula))
    return values


def calclulate_rpn(rpn):
    stack = []
    for token in rpn:
        if is_float(token):
            stack.append(token)
        elif token in constants:
            stack.append(constant_values[constants.index(token)])
        elif token in operators:
            value_1 = float(stack.pop())
            value_2 = float(stack.pop())
            if token == "/":
                if value_1 == 0:
                    raise Exception("Devide by 0")
                else:
                    stack.append(value_2 / value_1)
            else:
                stack.append(ops[token](value_2, value_1))
        elif token in functions:
            if token == "pow":
                value_1 = float(stack.pop())
                value_2 = float(stack.pop())
                if value_1 == 0 and value_2 == 0:
                    raise Exception("Zero pow zero")
                else:
                    stack.append(pow(value_2, value_1))
            elif token == "sqrt":
                value = float(stack.pop())
                if value < 0:
                    raise Exception("Negative value under sqrt")
                else:
                    stack.append(sqrt(value))
            else:
                stack.append(funcs[token](float(stack.pop())))
        elif token == "(":
            raise Exception("Wrong left parentheses in formula")
        else:
            raise Exception("Unknown exception")

    if len(stack) > 1:
        raise Exception("Wrong number order")
    return stack.pop()


if __name__ == "__main__":

    formulas = ["2*sin(1/(exp(3.5*x))+1)-tg(x+PI/2)",
                "1+sin(2.9*sin(E*x)/exp(tg(pow(4,5)+(pow(4,5)))-6*7*8))-9*tg(abs(10-cos(PI+E))*sqrt(abs(tg(11+PI)+tg(12))))",
                "sin(x)",
                "1+sin(tg(2)))",
                "1+sin((tg(2))",
                "1+sin(tg(2))/0",
                "1+sin(tg(2))*pow(0, 0)",
                "1+sqrt(6-3*4)",
                "1+sqt(6-4)",
                "sqrt(abs(x))",
                "1.2.3*x",]

    for formula in formulas:
        try:
            plt.plot(make_tabulation(
                parse_formula_to_rpn(tokenize(formula)), -6, 6, 0.01))
            plt.show()
        except Exception as e:
            print(e)
