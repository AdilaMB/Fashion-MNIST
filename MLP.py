s.run(tf.global_variables_initializer())
for epoch in range(epochs):
    arr = np.arange(X_train.shape[0])
    np.random.shuffle(arr)
    for i in range(0, X_train.shape[0], batch_size):
        s.run(optimizer, {input_X: X_train[arr[index:index + batch_size]],
                          input_y: y_train[arr[index:index + batch_size]],
                          keep_prob: dropout_prob})
    training_accuracy.append(s.run(accuracy, feed_dict={input_X: X_train,
                                                        input_y: y_train, keep_prob: 1}))
    training_loss.append(s.run(loss, {input_X: X_train,
                                      input_y: y_train, keep_prob: 1}))

    ## Evaluation of model
    testing_accuracy.append(accuracy_score(y_test.argmax(1),
                                           s.run(predicted_y, {input_X: X_test, keep_prob: 1}).argmax(1)))
    print("Epoch:{0}, Train loss: {1:.2f} Train acc: {2:.3f}, Test acc:{3:.3f}".format(epoch,
                                                                                       training_loss[epoch],
                                                                                       training_accuracy[epoch],
                                                                                       testing_accuracy[epoch]))