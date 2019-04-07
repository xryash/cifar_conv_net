import tensorflow as tf

from net import net
from utils import load_datasets, load_hyperparams, loss_plot, accuracy_plot


def get_layer_weights():
    return {
        'w_conv_1': tf.get_variable('W0', shape=(3, 3, 3, 64), initializer=tf.contrib.layers.xavier_initializer()),
        'w_conv_2': tf.get_variable('W1', shape=(3, 3, 64, 128), initializer=tf.contrib.layers.xavier_initializer()),
        'w_conv_3': tf.get_variable('W2', shape=(3, 3, 128, 256), initializer=tf.contrib.layers.xavier_initializer()),
        'w_conv_4': tf.get_variable('W3', shape=(3, 3, 256, 512), initializer=tf.contrib.layers.xavier_initializer()),
        'w_full_1': tf.get_variable('W4', shape=(4 * 512, 128), initializer=tf.contrib.layers.xavier_initializer()),
        'w_full_2': tf.get_variable('W5', shape=(128, 256), initializer=tf.contrib.layers.xavier_initializer()),
        'w_full_3': tf.get_variable('W6', shape=(256, 512), initializer=tf.contrib.layers.xavier_initializer()),
        'w_full_4': tf.get_variable('W7', shape=(512, 1024), initializer=tf.contrib.layers.xavier_initializer()),
        'out': tf.get_variable('W8', shape=(1024, 10), initializer=tf.contrib.layers.xavier_initializer()),
    }


def get_layer_biases():
    return {
        'b_conv_1': tf.get_variable('B0', shape=(64,), initializer=tf.contrib.layers.xavier_initializer()),
        'b_conv_2': tf.get_variable('B1', shape=(128,), initializer=tf.contrib.layers.xavier_initializer()),
        'b_conv_3': tf.get_variable('B2', shape=(256,), initializer=tf.contrib.layers.xavier_initializer()),
        'b_conv_4': tf.get_variable('B3', shape=(512,), initializer=tf.contrib.layers.xavier_initializer()),
        'b_full_1': tf.get_variable('B4', shape=(128,), initializer=tf.contrib.layers.xavier_initializer()),
        'b_full_2': tf.get_variable('B5', shape=(256,), initializer=tf.contrib.layers.xavier_initializer()),
        'b_full_3': tf.get_variable('B6', shape=(512,), initializer=tf.contrib.layers.xavier_initializer()),
        'b_full_4': tf.get_variable('B7', shape=(1024,), initializer=tf.contrib.layers.xavier_initializer()),
        'out': tf.get_variable('B8', shape=(10,), initializer=tf.contrib.layers.xavier_initializer()),
    }


def save_model(session, model_replica_path):
    """Save model"""
    saver = tf.train.Saver()
    saver.save(session, model_replica_path)


def main():
    # load train and test datasets
    train_x, train_y, test_x, test_y = load_datasets()

    # load model hyperparams
    epochs, batch_size, learning_rate, model_replica_path, dropout_rate = load_hyperparams()

    # init dropout params
    train_dropout_rate = dropout_rate
    test_dropout_rate = 0.0

    # remove previous weights, bias, inputs, etc..
    tf.reset_default_graph()

    # init inputs
    x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3), name='x')
    y = tf.placeholder(tf.float32, shape=(None, 10), name='y')

    dropout_rate = tf.placeholder(tf.float32, name='dropout_rate')

    # init layer biases and weights
    weights, biases = get_layer_weights(), get_layer_biases()

    # init model
    model = net(x, biases=biases, weights=weights, dropout_rate=dropout_rate)

    # init optimization function
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # accuracy function
    correct_prediction = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # create tensorflow session
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        train_loss, test_loss, train_accuracy, test_accuracy = [], [], [], []

        summary_writer = tf.summary.FileWriter('./output', session.graph)

        print('Training')

        for i in range(epochs):

            print('Epoch {} :'.format(i + 1))
            for batch in range(len(train_x) // batch_size):
                batch_x = train_x[batch * batch_size:min((batch + 1) * batch_size, len(train_x))]
                batch_y = train_y[batch * batch_size:min((batch + 1) * batch_size, len(train_y))]

                # start training
                session.run(optimizer, feed_dict={x: batch_x,
                                                  y: batch_y,
                                                  dropout_rate: train_dropout_rate})

                # compute metrics
                train_loss_batch, train_accuracy_batch = session.run([cost, accuracy], feed_dict={x: batch_x,
                                                                                                  y: batch_y,
                                                                                                  dropout_rate: train_dropout_rate})
                print(
                    'Batch range:{} - {}  Loss: {:>10.4f}  Accuracy: {:.6f}'.format(batch * batch_size,
                                                                                    min((batch + 1) * batch_size,
                                                                                        len(train_x)), train_loss_batch,
                                                                                    train_accuracy_batch))

            test_accuracy_batch, test_loss_batch = session.run([accuracy, cost], feed_dict={x: test_x,
                                                                                            y: test_y,
                                                                                            dropout_rate: test_dropout_rate})
            print(
                'Epoch {} finished, Loss: {:>10.4f} Validation Accuracy: {:.6f}'.format((i + 1), test_loss_batch,
                                                                             test_accuracy_batch))

            train_loss.append(train_loss_batch)
            test_loss.append(test_loss_batch)
            train_accuracy.append(train_accuracy_batch)
            test_accuracy.append(test_accuracy_batch)

        save_model(session, model_replica_path)

        # draw plots
        loss_plot(train_loss, test_loss)
        accuracy_plot(train_accuracy, test_accuracy)

        summary_writer.close()


if __name__ == "__main__":
    main()
    # load_datasets()
