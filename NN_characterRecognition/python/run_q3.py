import numpy as np
import scipy.io
from nn import *
import matplotlib.pyplot as plt

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']

max_iters = 50
# pick a batch size, learning rate
batch_size = 8
learning_rate = 1e-2
hidden_size = 64

batches = get_random_batches(train_x,train_y,batch_size)
batch_num = len(batches)

params = {}

# initialize layers here
initialize_weights(train_x.shape[1], hidden_size, params, 'layer1')
initialize_weights(hidden_size, train_y.shape[1], params, 'output')
assert(params['Wlayer1'].shape == (train_x.shape[1], hidden_size))
assert(params['blayer1'].shape == (hidden_size,))

# 3.3
from mpl_toolkits.axes_grid1 import ImageGrid

# learned nerwork

grid = ImageGrid(plt.figure(), 111,  # similar to subplot(111)
                 nrows_ncols=(8, 8),  # creates 2x2 grid of axes
                 axes_pad=0.1,  # pad between axes in inch.
                 )

W = params['Wlayer1']
for i in range(hidden_size):
    grid[i].imshow(np.reshape(W[:, i], (32, 32)))  # The AxesGrid object work as a list of axes.
    plt.axis('off')

plt.savefig('../results/intitial_grid.png')


train_loss, train_accuracy, valid_loss, valid_accuracy = list(), list(), list(), list()

# with default settings, you should get loss < 150 and accuracy > 80%
for itr in range(max_iters):
    total_loss = 0
    total_acc = 0
    for xb,yb in batches:
        h1 = forward(xb, params, 'layer1')
        probs = forward(h1, params, 'output', softmax)

        # loss
        # be sure to add loss and accuracy to epoch totals
        loss, acc = compute_loss_and_acc(yb, probs)
        total_loss += loss
        total_acc += acc

        # backward
        delta1 = probs - yb
        delta2 = backwards(delta1, params, 'output', linear_deriv)
        backwards(delta2, params, 'layer1', sigmoid_deriv)

        # apply gradient
        params['Wlayer1'] -= learning_rate * params['grad_Wlayer1']
        params['blayer1'] -= learning_rate * params['grad_blayer1']
        params['Woutput'] -= learning_rate * params['grad_Woutput']
        params['boutput'] -= learning_rate * params['grad_boutput']

    total_acc = total_acc / batch_num
    if itr % 1 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,total_loss,total_acc))

    train_loss.append(total_loss)
    train_accuracy.append(total_acc)

    #valid_y = valid_y.astype(int)
    h1 = forward(valid_x, params, 'layer1')
    probs = forward(h1, params, 'output', softmax)
    loss, acc = compute_loss_and_acc(valid_y, probs)
    valid_loss.append(loss)
    valid_accuracy.append(acc)


plt.figure('accuracy iters:{}, batch size:{}, learning rate:{:e}'.format(max_iters,batch_size,learning_rate))
plt.plot(range(max_iters), train_accuracy, color='g')
plt.plot(range(max_iters), valid_accuracy, color='b')
plt.legend(['train', 'validation'])
plt.title('accuracy iters:{}, batch size:{}, learning rate:{:e}'.format(max_iters,batch_size,learning_rate))
plt.savefig('../results/accuracy{}-{}-{:e}.png'.format(max_iters,batch_size,learning_rate))


plt.figure('loss iters:{}, batch size:{}, learning rate:{:e}'.format(max_iters,batch_size,learning_rate))
plt.plot(range(max_iters), train_loss, color='g')
plt.plot(range(max_iters), valid_loss, color='b')
plt.legend(['train', 'validation'])
plt.title('loss iters:{}, batch size:{}, learning rate:{:e}'.format(max_iters,batch_size,learning_rate))
plt.savefig('../results/loss{}-{}-{:e}.png'.format(max_iters,batch_size,learning_rate))



# run on validation set and report accuracy! should be above 75%
#valid_acc = None
h1 = forward(valid_x, params, 'layer1')
probs = forward(h1, params, 'output', softmax)
valid_loss, valid_acc = compute_loss_and_acc(valid_y, probs)
print('Validation accuracy: ',valid_acc)



if False: # view the data
    for crop in xb:
        import matplotlib.pyplot as plt
        plt.imshow(crop.reshape(32,32).T)

import pickle
saved_params = {k:v for k,v in params.items() if '_' not in k}
with open('q3_weights.pickle', 'wb') as handle:
    pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Q3.3
import matplotlib
#matplotlib.use('agg')

from mpl_toolkits.axes_grid1 import ImageGrid

# learned nerwork

grid = ImageGrid(plt.figure(), 111,  # similar to subplot(111)
                 nrows_ncols=(8, 8),  # creates 2x2 grid of axes
                 axes_pad=0.1,  # pad between axes in inch.
                 )

W = params['Wlayer1']
for i in range(hidden_size):
    grid[i].imshow(np.reshape(W[:, i], (32, 32)))  # The AxesGrid object work as a list of axes.
    plt.axis('off')


plt.savefig('../results/Learned_grid.png')


plt.show()

# Q3.4
confusion_matrix = np.zeros((train_y.shape[1],train_y.shape[1]))

y = train_y.astype(int)
h1 = forward(train_x, params, 'layer1')
probs = forward(h1, params, 'output', softmax)

pred_y = (probs == np.expand_dims(np.max(probs, axis=1), axis=1))
with_same_prob = np.where(np.sum(pred_y, axis=1) > 1)[0]
for i in range(with_same_prob.shape[0]):
    pred_y[i, np.where(pred_y[i, :] == np.max(pred_y[i, :]))[0][0] + 1:] = False

yl = [np.where(y[i, :] == 1)[0][0] for i in range(y.shape[0])]
pred_yl = [np.where(pred_y[i, :] == 1)[0][0] for i in range(pred_y.shape[0])]

for a, p in zip(yl, pred_yl):
    confusion_matrix[a][p] += 1


import string
plt.imshow(confusion_matrix, interpolation='nearest')
plt.grid(True)
plt.xticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.yticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))

#plt.title('confusion matrix')

plt.savefig('../results/confusion_matrix.png')

plt.pause(10)

