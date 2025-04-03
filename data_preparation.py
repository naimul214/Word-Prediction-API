from datasets import load_dataset
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
import tensorflow as tf
import json

def load_and_filter_dataset():
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    def is_valid_line(line):
        return line.strip() != '' and not line.strip().startswith('=')
    return dataset.filter(lambda example: is_valid_line(example['text']))

def tokenize_dataset(dataset):
    nltk.download('punkt_tab')
    return dataset.map(
        lambda examples: {'tokens': [word_tokenize(text) for text in examples['text']]},
        batched=True
    )

def create_vocabulary(tokenized_dataset, vocab_size=20000):
    token_counter = Counter()
    for example in tokenized_dataset['train']:
        token_counter.update(example['tokens'])
    most_common = token_counter.most_common(vocab_size)
    vocab = {word: idx + 2 for idx, (word, _) in enumerate(most_common)}
    vocab['<unk>'] = 1
    return vocab

def save_vocabulary(vocab, file_path='vocab.json'):
    with open(file_path, 'w') as f:
        json.dump(vocab, f)

def map_tokens_to_ids(tokenized_dataset, vocab):
    return tokenized_dataset.map(
        lambda example: {'input_ids': [vocab.get(token, 1) for token in example['tokens']]}
    )

def create_sequences(indexed_dataset):
    return indexed_dataset.map(
        lambda example: {
            'input_sequence': example['input_ids'][:-1],
            'target_sequence': example['input_ids'][1:]
        } if len(example['input_ids']) >= 2 else {
            'input_sequence': [], 'target_sequence': []
        }
    ).filter(lambda x: len(x['input_sequence']) > 0)

def create_tf_dataset(sequence_dataset, split='train', batch_size=32):
    return tf.data.Dataset.from_generator(
        lambda: ({'input_sequence': ex['input_sequence'], 'target_sequence': ex['target_sequence']} 
                 for ex in sequence_dataset[split]),
        output_signature={
            'input_sequence': tf.TensorSpec(shape=(None,), dtype=tf.int32),
            'target_sequence': tf.TensorSpec(shape=(None,), dtype=tf.int32)
        }
    ).padded_batch(batch_size, padded_shapes={'input_sequence': [None], 'target_sequence': [None]}).map(
        lambda x: (x['input_sequence'], x['target_sequence'])
    )

def prepare_data(vocab_size=20000, batch_size=32):
    dataset = load_and_filter_dataset()
    tokenized = tokenize_dataset(dataset)
    vocab = create_vocabulary(tokenized, vocab_size)
    save_vocabulary(vocab)
    indexed = map_tokens_to_ids(tokenized, vocab)
    sequenced = create_sequences(indexed)
    train_dataset = create_tf_dataset(sequenced, 'train', batch_size)
    val_dataset = create_tf_dataset(sequenced, 'validation', batch_size)
    return train_dataset, val_dataset