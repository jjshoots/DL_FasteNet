from matplotlib import pyplot as plt
import numpy as np

FN_recall = [
    0.9685,
    0.9609,
    0.9556,
    0.9485,
    0.9399,
    0.9333,
    0.9288,
    0.9157,
    0.8997,
    0.8698,
]

FN_precision = [
    0.9271,
    0.9467,
    0.9634,
    0.9723,
    0.9768,
    0.982,
    0.9881,
    0.9894,
    0.9916,
    0.9945,
]

LN_recall = [
    0.9559,
    0.9411,
    0.9272,
    0.9225,
    0.9179,
    0.9132,
    0.9102,
    0.9063,
    0.9022,
    0.8918,
]

LN_precision = [
    0.9556,
    0.9730,
    0.9783,
    0.9793,
    0.9806,
    0.9823,
    0.9848,
    0.9871,
    0.9891,
    0.9905,
]

VN_recall = [
    0.9367,
    0.9247,
    0.9181,
    0.9131,
    0.9105,
    0.8971,
    0.8927, 
    0.8906,
    0.8861,
    0.8819,
]

VN_precision = [
    0.9586,
    0.9770,
    0.9805,
    0.9813,
    0.9835,
    0.9858,
    0.9886,
    0.9893,
    0.9913,
    0.9921,
]

FN_recall = np.array(FN_recall) * 100
FN_precision = np.array(FN_precision) * 100

LN_recall = np.array(LN_recall) * 100
LN_precision = np.array(LN_precision) * 100

VN_recall = np.array(VN_recall) * 100
VN_precision = np.array(VN_precision) * 100

threshold = 9

for recall_value, precision_value in zip(FN_recall, FN_precision):
    if threshold > 0:
        plt.text(recall_value, precision_value,f'    \u03B8 = {threshold/10}')
    else:
        plt.text(recall_value, precision_value,f'    \u03B8 = {0.01}')
    threshold -= 1

for recall_value, precision_value in zip(LN_recall, LN_precision):
    plt.text(recall_value, precision_value,f' ')

for recall_value, precision_value in zip(VN_recall, VN_precision):
    plt.text(recall_value, precision_value,f' ')

plt.plot(FN_recall, FN_precision,'ro-')
plt.plot(LN_recall, LN_precision,'go-')
plt.plot(VN_recall, VN_precision,'bo-')
plt.legend([f'FasteNet', f'LargeNet', f'VanillaNet'])
plt.xlim(86, 100)
plt.ylim(92, 100)
plt.grid(linestyle='dotted')
plt.title(f'Precision vs. Recall for FasteNet')
plt.xlabel(f'Recall (%)')
plt.ylabel(f'Precision (%)')
plt.show()