from openprompt.data_utils.data_processor import *
import os, json, csv

class CLSProcessor(DataProcessor):
    def __init__(self, labels_origin, labels_mapped):
        self.labels_origin = labels_origin
        self.labels_mapped = labels_mapped
        self.labels_mapping = {
            origin: kth for kth, (origin, mapped) in enumerate(zip(labels_origin, labels_mapped))
        }

    def get_label(self, origin):
        return self.labels_mapping[origin]