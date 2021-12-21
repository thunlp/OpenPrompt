from openprompt.data_utils.data_processor import *
import os, json, csv

class OptionProcessor(DataProcessor):
    def __init__(self):
        self.labels = []

    def update_options(self, example):
        options = example.meta["options"]
        if (len(options) > len(self.labels)):
            self.labels = [i for i in range(len(options))]

class CLSProcessor(DataProcessor):
    def __init__(self, labels_origin, labels_mapped):
        self.labels_origin = labels_origin
        self.labels_mapped = labels_mapped
        self.labels_mapping = {
            origin: kth for kth, (origin, mapped) in enumerate(zip(labels_origin, labels_mapped))
        }

    @property
    def labels(self):
        return self.labels_mapped

    def get_label(self, origin):
        return self.labels_mapping[origin]