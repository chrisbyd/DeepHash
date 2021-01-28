import numpy as np

class Dataset(object):
    def __init__(self, dataset, config):
        print ("Initializing Dataset")
        self._dataset = dataset
        self.n_samples = dataset.n_samples
        self._train = dataset.train
        self.output_dim = config.output_dim
        self.dataset_name = config.dataset
        self._output = np.zeros((self.n_samples, self.output_dim), dtype=np.float32)

        self._perm = np.arange(self.n_samples)
        np.random.shuffle(self._perm)
        self._index_in_epoch = 0
        self._epochs_complete = 0
        self.special_datasets = ["vehicleID", "VeRi"]
        self.label_dim = config.label_dim
        self.label_to_one_hot =   np.eye(self.label_dim)
        print ("Dataset already")
        return

    def next_batch(self, batch_size):
        """
        Args:
          batch_size
        Returns:
          [batch_size, (n_inputs)]: next batch images
          [batch_size, n_class]: next batch labels
        """
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        # Another epoch finish
        if self._index_in_epoch > self.n_samples:
            if self._train:
                # Training stage need repeating get batch
                self._epochs_complete += 1
                # Shuffle the data
                np.random.shuffle(self._perm)
                # Start next epoch
                start = 0
                self._index_in_epoch = batch_size
            else:
                # Validation stage only process once
                start = self.n_samples - batch_size
                self._index_in_epoch = self.n_samples
        end = self._index_in_epoch

        data, label = self._dataset.data(self._perm[start:end])
        if self.dataset_name in self.special_datasets:
            label = np.squeeze(label)
            label = self.label_to_one_hot[label]

        return (data, label)

    def feed_batch_output(self, batch_size, output):
        """
        Args:
          batch_size
          [batch_size, n_output]
        """
        start = self._index_in_epoch - batch_size
        end = self._index_in_epoch
        self.output[self._perm[start:end], :] = output
        return

    @property
    def output(self):
        return self._output

    @property
    def label(self):
        return self._dataset.get_labels()

    @property
    def cam(self):
        return self._dataset.get_cams()

    def finish_epoch(self):
        self._index_in_epoch = 0
        np.random.shuffle(self._perm)

