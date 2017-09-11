from resnet import * 
import tensorflow as tf

class Train:

    def __init__(self, 
            trainDir='/tmp/resnet_train', 
            learningRate=.0001, 
            batchSize=16, 
            maxSteps=500000, 
            resume=False, 
            minimalSummaries=True):
        self.trainDir = trainDir
        self.learningRate=learningRate
        self.batchSize = batchSize
        self.maxSteps = maxSteps
        self.resume = resume
        self.minimalSummaries = minimalSummaries

    def loss(self, logits, labels):
        #cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
     
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

        loss_ = tf.add_n([cross_entropy_mean] + regularization_losses)
        tf.summary.tensor_summary('loss', loss_)

        return loss_

    def train(self, is_training, logits, images, labels):
        global_step = tf.get_variable('global_step', [],
                initializer=tf.constant_initializer(0),
                trainable=False)
        val_step = tf.get_variable('val_step', [],
                initializer=tf.constant_initializer(0),
                trainable=False)

        loss_ = self.loss(logits, labels)
        predictions = tf.nn.softmax(logits)

        a = tf.argmax(predictions, 1)
        b = tf.argmax(labels,1)
        correctPredictions = tf.equal(a, b)
        accuracy = tf.reduce_mean(tf.cast(correctPredictions, tf.float32))

        # loss_avg
        ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
        tf.add_to_collection(UPDATE_OPS_COLLECTION, ema.apply([loss_]))
        tf.summary.scalar('loss_avg', ema.average(loss_))


        # validation stats
        ema = tf.train.ExponentialMovingAverage(0.9, val_step)
        val_op = tf.group(val_step.assign_add(1), ema.apply([accuracy]))
        accuracy_avg = ema.average(accuracy)
        tf.summary.scalar('val_accuracy_avg', accuracy_avg)

        tf.summary.scalar('learning_rate', self.learningRate)

        opt = tf.train.AdamOptimizer(self.learningRate, .9, .999, epsilon=1e-8)
        grads = opt.compute_gradients(loss_)
        for grad, var in grads:
            if grad is not None and not self.minimalSummaries:
                tf.histogram_summary(var.op.name + '/gradients', grad)
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        if not self.minimalSummaries:
            # Display the training images in the visualizer.
            tf.image_summary('images', images)

            for var in tf.trainable_variables():
                tf.histogram_summary(var.op.name, var)

        batchnorm_updates = tf.get_collection(UPDATE_OPS_COLLECTION)
        batchnorm_updates_op = tf.group(*batchnorm_updates)
        train_op = tf.group(apply_gradient_op, batchnorm_updates_op)

        saver = tf.train.Saver(tf.global_variables())

        summary_op = tf.summary.merge_all()

        init = tf.global_variables_initializer()

        sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
        sess.run(init)
        tf.train.start_queue_runners(sess=sess)

        summary_writer = tf.summary.FileWriter(self.trainDir, sess.graph)

        if self.resume:
            latest = tf.train.latest_checkpoint(self.trainDir)
            if not latest:
                print("No checkpoint to continue from in", self.trainDir)
                sys.exit(1)
            print("resume", latest)
            saver.restore(sess, latest)

        for x in range(self.maxSteps + 1):
            start_time = time.time()

            step = sess.run(global_step)
            i = [train_op, loss_]

            write_summary = step % 100 and step > 1
            if write_summary:
                i.append(summary_op)

            o = sess.run(i, { is_training: True })

            loss_value = o[1]

            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step%10 == 0:
                examples_per_sec = self.batchSize / float(duration)
                format_str = ('step %d, loss = %.2f (%.1f examples/sec; %.3f '
                        'sec/batch)')
                print(format_str % (step, loss_value, examples_per_sec, duration))

            if write_summary:
                summary_str = o[2]
                summary_writer.add_summary(summary_str, step)

            # Save the model checkpoint periodically.
            if step > 1 and step % 100 == 0:
                checkpoint_path = os.path.join(self.trainDir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=global_step)

            # Run validation periodically
            if step > 1 and step % 100 == 0:
                _, accuracy_value = sess.run([val_op, accuracy], { is_training: False })
                print('Validation accuracy %.2f' % accuracy_value)



'''    # unused kept for future reference
    def top_k_error(predictions, labels, k):
        batch_size = float(self.batchSize) #tf.shape(predictions)[0]
        in_top1 = tf.to_float(tf.nn.in_top_k(predictions, labels, k))
        num_correct = tf.reduce_sum(in_top1)
        return (batch_size - num_correct) / batch_size
'''

