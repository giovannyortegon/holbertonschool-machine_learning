#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

person = ('Farrah', 'Fred', 'Felicia')
n_fruits = ('apples', 'bananas', 'oranges', 'peaches')
colors = ('red', 'yellow', '#ff8000', '#ffe5b4')
width = 0.5

for i in range(0, len(fruit)):
    plt.bar(person, fruit[i], width, bottom=np.sum(fruit[:i], axis=0),
            color=colors[i], label=n_fruits[i])

plt.yticks(np.arange(0, 81, 10))

plt.legend()
plt.title('Number of Fruit per Person')
plt.ylabel('Quantity of Fruit')

plt.show()
