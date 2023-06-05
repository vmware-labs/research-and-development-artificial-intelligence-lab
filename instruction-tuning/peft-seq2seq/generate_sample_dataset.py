# Copyright 2023 VMware, Inc.
# SPDX-License-Identifier: Apache-2.0

from datasets import load_dataset

dataset = load_dataset(
    'bigscience/P3', 'dream_answer_to_dialogue', split='train')

sample_dataset = dataset.select([i for i in range(1000)])
print(len(dataset))
sample_dataset = sample_dataset.rename_column("inputs_pretokenized", "input")
sample_dataset = sample_dataset.rename_column("targets_pretokenized", "output")

sample_dataset = sample_dataset.remove_columns(["inputs", "targets"])
print(len(sample_dataset))
sample_dataset.to_csv('sample.csv')
