import numpy
import pandas


def main():
    atrain75 = pandas.read_csv('random_a75.csv', header=None)
    atrain25 = pandas.read_csv('random_a25.csv', header=None)
    btrain75 = pandas.read_csv('B_75%.csv', header=None)
    btrain25 = pandas.read_csv('random_b25.csv', header=None)
    ctrain75 = pandas.read_csv('random_c75.csv', header=None)
    ctrain25 = pandas.read_csv('random_c25.csv', header=None)

    ae = 0.00001
    be = 200
    ce = 700

    ni = 5000  # number of iterations
    alpha = 0.1
    gain = 0.1
    error = 0

    alphaloop = 9
    gainloop = 9
    # WEIGHTS
    x = 0.1
    y = 0.4
    z = -0.2

    path = ""

    a75 = numpy.asarray(atrain75)
    for n in range(0, alphaloop):
       for m in range(0, gainloop):
    for j in range(0, ni):
        if (j == 0 or error > ae):
            error = 0
            for i in range(len(a75)):
                pattern = a75[i]
    
                net = pattern[0] * x + pattern[1] * y + z
                out = (1 / (1 + numpy.math.exp(-1 * gain * net)))
                if pattern[2] == 0:
                    desired = 1
                else:
                    desired = 0
                delta = alpha * (desired - out)
                error += (desired - out) * (desired - out)
                pattern0 = delta * pattern[0]
                pattern1 = delta * pattern[1]
                pattern2 = delta
                x = x + pattern0
                y = y + pattern1
                z = z + pattern2
        else:
            break
    calculate(a75, x, y, z, 'This is A75 for gain', gain, 'and alpha', alpha)
    print(j, '  error for a75', error)
    
    error = 0
    alpha = 0.9
    gain = 0.9
    # WEIGHTS
    x = 0.2
    y =-0.1
    z = -0.3

    a25 = numpy.asarray(atrain25)
    
    for j in range(0, ni):
        if (j == 0 or error > ae):
            error = 0
            for i in range(len(a25)):
                pattern = a25[i]
    
                net = pattern[0] * x + pattern[1] * y + z
    
                out = 1 / (1 + numpy.math.exp(-1 * gain * net))
    
                if (pattern[2] == 0):
                    desired = 1
                else:
                    desired = 0
                delta = alpha * (desired - out)
    
                error += (desired - out) * (desired - out)
    
                pattern0 = delta * pattern[0]
                pattern1 = delta * pattern[1]
                pattern2 = delta
                x = x + pattern0
                y = y + pattern1
                z = z + pattern2
        else:
            break
    calculate(a25, x, y, z, 'This is A25 for gain', gain, 'and alpha', alpha)
    print(j, '   error for a25', error)
    error = 0
    alpha = 0.1
    gain = 0.1
    x = 0.49
    y = 0.49
    z = -0.49

    b75 = numpy.asarray(btrain75)

    for j in range(0, ni):
        if (j == 0 or error > be):
            error = 0
            for i in range(len(b75)):
                pattern = b75[i]

                net = pattern[0] * x + pattern[1] * y + z
                out = 1 / (1 + numpy.math.exp(-1 * gain * net))

                if (pattern[2] == 0):
                    desired = 1
                else:
                    desired = 0
                delta = alpha * (desired - out)
                error += (desired - out) * (desired - out)
                pattern0 = delta * pattern[0]
                pattern1 = delta * pattern[1]
                pattern2 = delta
                x = x + pattern0
                y = y + pattern1
                z = z + pattern2
        else:
            break
    calculate(b75, x, y, z, 'This is B75 for gain', gain, 'and alpha', alpha)
    print(j, 'error for b75', error)
    alpha = 0.2
    gain = 0.1
    error = 0
    x = 0.076
    y = -0.3
    z = -0.24

    b25 = numpy.asarray(btrain25)

    for j in range(0, ni):
        if (j == 0 or error > be):
            error = 0
            for i in range(len(b25)):
                pattern = b25[i]

                net = pattern[0] * x + pattern[1] * y + z
                out = 1 / (1 + numpy.math.exp(-1 * gain * net))

                if (pattern[2] == 0):
                    desired = 1
                else:
                    desired = 0
                delta = alpha * (desired - out)
                error += (desired - out) * (desired - out)
                pattern0 = delta * pattern[0]
                pattern1 = delta * pattern[1]
                pattern2 = delta
                x = x + pattern0
                y = y + pattern1
                z = z + pattern2
        else:
            break
    calculate(b25, x, y, z, 'This is B25 for gain', gain, 'and alpha', alpha)
    print(j, 'error for b25', error)
    error = 0
    alpha = 0.3
    gain = 0.6
    x = 0.49
    y = 0.49
    z = -0.49

    c75 = numpy.asarray(ctrain75)

    for j in range(0, ni):
        if j == 0 or error > ce:
            error = 0
            for i in range(len(c75)):
                pattern = c75[i]
                net = pattern[0] * x + pattern[1] * y + z
                out = 1 / (1 + numpy.math.exp(-1 * gain * net))

                if pattern[2] == 0:
                    desired = 1
                else:
                    desired = 0
                delta = alpha * (desired - out)
                error += (desired - out) * (desired - out)

                pattern0 = delta * pattern[0]
                pattern1 = delta * pattern[1]
                pattern2 = delta
                x = x + pattern0
                y = y + pattern1
                z = z + pattern2
        else:
            break
    calculate(c75, x, y, z, 'This is C75 for gain', gain, 'and alpha', alpha)
    print(j, 'error c75', error)

    error = 0
    alpha = 0.2
    gain = 0.6
    x = 0.49
    y = 0.49
    z = -0.49

    c25 = numpy.asarray(ctrain25)

    for j in range(0, ni):
        if j == 0 or error > ce:
            error = 0
            for i in range(len(c25)):
                pattern = c25[i]
                net = pattern[0] * x + pattern[1] * y + z
                out = 1 / (1 + numpy.math.exp(-1 * gain * net))

                if pattern[2] == 0:
                    desired = 1
                else:
                    desired = 0
                delta = alpha * (desired - out)
                error += (desired - out) * (desired - out)

                pattern0 = delta * pattern[0]
                pattern1 = delta * pattern[1]
                pattern2 = delta
                x = x + pattern0
                y = y + pattern1
                z = z + pattern2
        else:
            break
    calculate(c25, x, y, z, 'This is C25 for gain', gain, 'and alpha', alpha)
    print(j, 'error c25', error)
    error = 0
    x = numpy.random.uniform(-0.5, 0.5)
    y = numpy.random.uniform(-0.5, 0.5)
    z = numpy.random.uniform(-0.5, 0.5)


def calculate(dataset, x, y, z, name, g, name2, a):
    truths = {'true negative': 0, 'true positive': 0, 'false negative': 0, 'false positive': 0}
    counter = 0
    for i in range(len(dataset)):
        pattern = dataset[i]
        counter += 1
        value = pattern[0] * x + pattern[1] * y + z
        if value > 0 and pattern[2] == 0:
            truths['true positive'] = truths.get('true positive') + 1
        elif value > 0 and pattern[2] == 1:
            truths['false positive'] = truths.get('false positive') + 1
        elif value <= 0 and pattern[2] == 1:
            truths['true negative'] = truths.get('true negative') + 1
        else:
            truths['false negative'] = truths.get('false negative') + 1
    print('')
    print(name, g, name2, a)
    print('False positives: ', truths['false positive'], 'False negatives: ', truths['false negative'])
    print('true positives: ', truths['true positive'], 'False negatives: ', truths['true negative'])
    accuracy = (truths['true positive'] + truths['true negative']) / counter
    error = 1 - accuracy
    tp = truths['true positive'] / (truths['true positive'] + truths['false negative'])
    tn = truths['true negative'] / (truths['false positive'] + truths['true negative'])
    fp = truths['false positive'] / (truths['false positive'] + truths['true negative'])
    fn = truths['false negative'] / (truths['true positive'] + truths['false negative'])
    print('Accuracy', numpy.around(accuracy, 4))
    print('error', numpy.around(error, 4))
    print('True Positive Rate', numpy.around(tp, 4))
    print('False Positive Rate', numpy.around(fp, 4))
    print('True Negative Rate', numpy.around(tn, 4))
    print('False Negative Rate', numpy.around(fn, 4))
    print('')


if __name__ == "__main__":
    main()
