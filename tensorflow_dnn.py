import tensorflow as tf
import numpy as np
import pandas as pd

# to make this notebook's output stable across runs
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

# 1. Data Stuff
from sklearn.datasets import load_iris
data = load_iris()

X_data = data.data
y_data = data.target

# train and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, random_state=42)

# 1.1 Next batch data function
def next_batch(num, data, labels):
    '''
    Return a total of `num` random samples and labels.
    '''
    idx = np.arange(0 , len(data.data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

#reset tf graph
reset_graph()

# Functions to do accuracy:
with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))


# 2. Placeholders for input and output data

n_features = 4

X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")
y = tf.placeholder(tf.int64, shape=(None), name="y")

# 3. Define the deep network
n_hidden1 = 400
n_hidden2 = 200
n_outputs = 3

with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(X, n_hidden1, name="hidden1", activation=tf.nn.relu)
    hidden2 = tf.layers.dense(hidden1, n_hidden2, name="hidden2", activation=tf.nn.relu)
    logits  = tf.layers.dense(hidden2, n_outputs, name="outputs")

# 4. What is the loss function
with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

# 5. What is the optimizer
learning_rate = 0.001

with tf.name_scope("Training"):
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9, use_nesterov=True)
    training_op = optimizer.minimize(loss)

# 6. Execution Phase
n_epochs=500
batch_size=10

init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(int(np.ceil(len(X_train)/batch_size))):
            X_batch, y_batch = next_batch(batch_size, X_train, y_train)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        print(epoch, "Train accuracy:", acc_train)
    save_path = saver.save(sess, "./my_model_final,ckpt")
