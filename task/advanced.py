from tensorflow.contrib.keras.api.keras.callbacks import TensorBoard



class TensorBoardBatch(TensorBoard):
    #TODO: Fix Tensorboard
    def __init__(self, log_dir='data/logs',
                 histogram_freq=0,
                 batch_size=32,
                 write_graph=True,
                 write_grads=False,
                 write_images=False):
        super(TensorBoardBatch, self).__init__(log_dir,
                                               histogram_freq=histogram_freq,
                                               batch_size=batch_size,
                                               write_graph=write_graph,
                                               write_grads=write_grads,
                                               write_images=write_images)

        # conditionally import tensorflow iff TensorBoardBatch is created
        self.tf = __import__('tensorflow')
        self.global_step = 1

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}

        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            summary = self.tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.writer.add_summary(summary, self.global_step)
        self.global_step += 1

        self.writer.flush()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            summary = self.tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.writer.add_summary(summary, self.global_step)

        self.global_step += 1
        self.writer.flush()
