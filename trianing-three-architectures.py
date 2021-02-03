import numpy
import pandas
import matplotlib.pyplot as plt


def main():
    train_data_1 = pandas.read_csv('train_data_1.txt', header=None)
    train_data_1 = normalize_data(train_data_1)
    train_data_2 = pandas.read_csv('train_data_2.txt', header=None)
    train_data_2 = normalize_data(train_data_2)
    train_data_3 = pandas.read_csv('train_data_3.txt', header=None)
    train_data_3 = normalize_data(train_data_3)
    test_data = pandas.read_csv('test_data_4.txt', header=None)
    test_data = normalize_data(test_data)

    # combine all training datasets into one
    training = [train_data_1, train_data_2, train_data_3]
    train_data_all = pandas.concat(training)

    weight1 = 0.1796
    bias = 0.13329
    result = neuron_one_train(train_data_all, weight1, bias)  # train with all training data from 3 days
    training_error_one(train_data_1, result[0], result[1])  # calculate total error for day 1
    training_error_one(train_data_2, result[0], result[1])  # calculate total error for day 2
    training_error_one(train_data_3, result[0], result[1])  # calculate total error for day 3
    neuron_one_test(test_data, test_data, result[0], result[1])  # calculate total error for testing data

    weight1 = -0.4155
    weight2 = 0.3433
    bias = 0.2259
    result = neuron_two_train(train_data_all, weight1, weight2, bias)
    training_error_two(train_data_1, result[0], result[1], result[2])
    training_error_two(train_data_2, result[0], result[1], result[2])
    training_error_two(train_data_3, result[0], result[1], result[2])
    neuron_two_test(test_data, test_data, result[0], result[1], result[2])

    weight1 = -0.3737
    weight2 = -0.0506
    weight3 = 0.1637
    bias = 0.09976
    result = neuron_three_train(train_data_all, weight1, weight2, weight3, bias)
    training_error_three(train_data_1, result[0], result[1], result[2], result[3])
    training_error_three(train_data_2, result[0], result[1], result[2], result[3])
    training_error_three(train_data_3, result[0], result[1], result[2], result[3])
    neuron_three_test(test_data, test_data, result[0], result[1], result[2], result[3])


# training for architecture 1
def neuron_one_train(train_data, weight1, bias):
    train = numpy.asarray(train_data)

    alpha = 0.001
    for j in range(3000):
        for i in range(len(train)):
            pattern = train[i]
            net = (pattern[0] * weight1) + bias
            delta = alpha * (pattern[1] - net)
            weight1 = weight1 + (delta * pattern[0])
            bias = bias + delta
    return weight1, bias


# test for architecture 1
def neuron_one_test(train_data, test_data, weight1, bias):
    test = numpy.asarray(test_data)
    error = 0
    values = []
    print("testing function 1 weights:", weight1, bias)
    for i in range(len(test)):
        pattern = test[i]
        net = weight1 * pattern[0] + bias
        values.append(net)
        error += (pattern[1] - net) * (pattern[1] - net)
    print_graph(train_data, values)
    print("testing function 1 error", error)
    print("")


# training for architecture 2
def neuron_two_train(train_data, weight1, weight2, bias):
    train = numpy.asarray(train_data)

    alpha = 0.01
    for j in range(1000):
        for i in range(len(train)):
            pattern = train[i]
            net = (pattern[0] * weight1) + ((pattern[0] * pattern[0]) * weight2) + bias
            delta = alpha * (pattern[1] - net)
            weight1 = weight1 + (delta * pattern[0])
            weight2 = weight2 + (delta * (pattern[0] * pattern[0]))
            bias = bias + delta
    return weight1, weight2, bias


# test for architecture 2
def neuron_two_test(train_data, test_data, weight1, weight2, bias):
    test = numpy.asarray(test_data)
    error = 0
    values = []
    print("testing function 2 weights:", weight1, weight2, bias)
    for i in range(len(test)):
        pattern = test[i]
        net = (weight1 * pattern[0]) + ((pattern[0] * pattern[0]) * weight2) + bias
        values.append(net)
        error += (pattern[1] - net) * (pattern[1] - net)
    print_graph(train_data, values)
    print("testing function 2 error", error)
    print("")


# training for architecture 3
def neuron_three_train(train_data, weight1, weight2, weight3, bias):
    train = numpy.asarray(train_data)

    alpha = 0.2
    for j in range(5000):
        for i in range(len(train)):
            pattern = train[i]
            net = (pattern[0] * weight1) + ((pattern[0] * pattern[0]) * weight2) + (
                        (pattern[0] * pattern[0] * pattern[0]) * weight3) + bias
            delta = alpha * (pattern[1] - net)
            weight1 = weight1 + (delta * pattern[0])
            weight2 = weight2 + (delta * (pattern[0] * pattern[0]))
            weight3 = weight3 + (delta * (pattern[0] * pattern[0] * pattern[0]))
            bias = bias + delta
    return weight1, weight2, weight3, bias


# test for architecture 3
def neuron_three_test(train_data, test_data, weight1, weight2, weight3, bias):
    test = numpy.asarray(test_data)
    error = 0
    print("testing function 3 weights:", weight1, weight2, weight3, bias)
    values = []
    for i in range(len(test)):
        pattern = test[i]
        net = (weight1 * pattern[0]) + ((pattern[0] * pattern[0]) * weight2) + (
                    (pattern[0] * pattern[0] * pattern[0]) * weight3) + bias
        values.append(net)
        error += (pattern[1] - net) * (pattern[1] - net)
    print_graph(train_data, values)
    print("testing day 3 error", error)


# print graphs for training and testing errors
def print_graph(train_data, values):
    day = numpy.asarray(train_data[0])
    power = numpy.asarray(train_data[1])
    mymodel2 = numpy.poly1d(numpy.polyfit(day, values, 3))
    myline = numpy.linspace(0, 1, 100)
    plt.scatter(day, power)
    plt.scatter(day, values)
    plt.plot(myline, mymodel2(myline))
    plt.xlabel('Time (hour)')
    plt.ylabel('Energy Consumed (kW)')
    plt.show()


# calculate training error for architecture 1
def training_error_one(train_data, weight1, bias):
    train = numpy.asarray(train_data)
    values = []
    error = 0
    for i in range(len(train)):
        pattern = train[i]
        net = pattern[0] * weight1 + bias
        values.append(net)
        error += (pattern[1] - net) * (pattern[1] - net)
    print_graph(train_data, values)
    print("training function 1 error", error)



# calculate training error for architecture 2
def training_error_two(train_data, weight1, weight2, bias):
    train = numpy.asarray(train_data)
    values = []
    error = 0
    for i in range(len(train)):
        pattern = train[i]
        net = (weight1 * pattern[0]) + ((pattern[0] * pattern[0]) * weight2) + bias
        values.append(net)
        error += (pattern[1] - net) * (pattern[1] - net)
    print_graph(train_data, values)
    print("training function 2 error", error)


# calculate training error for architecture 3
def training_error_three(train_data, weight1, weight2, weight3, bias):
    train = numpy.asarray(train_data)
    values = []
    error = 0
    for i in range(len(train)):
        pattern = train[i]
        net = (weight1 * pattern[0]) + ((pattern[0] * pattern[0]) * weight2) + (
                    (pattern[0] * pattern[0] * pattern[0]) * weight3) + bias
        values.append(net)
        error += (pattern[1] - net) * (pattern[1] - net)
    print_graph(train_data, values)
    print("training function 3 error", error)


def normalize_data(data):
    data[0] = (data[0] - data[0].min()) / (data[0].max() - data[0].min())
    data[1] = (data[1] - data[1].min()) / (data[1].max() - data[1].min())
    return data


if __name__ == "__main__":
    main()
