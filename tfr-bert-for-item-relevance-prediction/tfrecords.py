import os

import numpy as np
import pandas as pd
import tensorflow as tf
import tqdm
from official.nlp.bert import tokenization
from tensorflow_serving.apis import input_pb2
import string

from config import max_seq_length, bert_path

IMG_HEIGHT = 40
IMG_WIDTH = 40
NORMALIZATION_LAYER = tf.keras.layers.Rescaling(1./255)


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    try:
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    except:
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    try:
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
    except:
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    try:
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    except:
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def category_example(category_name):
    feature = {
        'category_name': _bytes_feature(bytes(str(category_name), 'utf-8')),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def truncate_seq(tokens_sequences, max_length):
    """Truncates a list of sequences with tokens in place to the maximum length."""
    assert max_length > 0
    assert isinstance(max_length, int)
    assert len(tokens_sequences) > 0
    total_length = np.sum([len(seq) for seq in tokens_sequences])
    curr_length = 0
    if total_length > max_length:
        # Truncation is needed.
        for i, seq in enumerate(tokens_sequences):
            del seq[(max_length - curr_length) // (len(tokens_sequences) - i):]
            curr_length += len(seq)


def to_bert_ids(tokenizer, sequences, max_length):
    """Converts a list of sentences to related Bert ids.

    Args:
      tokenizer - Bert tokenizer
      sequences - Iterable of sequences

    Returns:
      A tuple (`input_ids`, `input_masks`, `segment_ids`) for Bert finetuning.
    """
    assert len(sequences) > 0
    tokens_sequences = [tokenizer.tokenize(seq) for seq in sequences]
    assert max_length > 0
    truncate_seq(tokens_sequences, max_length - (len(sequences) + 1))

    # The convention in BERT for sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP] got it [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1     0   0   0
    #
    # The `type_ids` (aka. `segment_ids`) are used to indicate whether this is
    # the first sequence, the second sequence and etc. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    #
    # When there is only one sentence given, the sequence pair would be:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0
    #

    tokens = ["[CLS]"]
    for seq in tokens_sequences:
        tokens = tokens + seq + ["[SEP]"]

    segment_ids = [0] + [0] * len(tokens_sequences[0]) + [0]
    for i, seq in enumerate(tokens_sequences):
        if i != 0:
            segment_ids = segment_ids + [i % 2] * len(seq) + [i % 2]

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    if len(input_ids) < max_length:
        padding_len = max_length - len(input_ids)
        input_ids.extend([0] * padding_len)
        input_mask.extend([0] * padding_len)
        segment_ids.extend([0] * padding_len)

    assert len(input_ids) == max_length
    assert len(input_mask) == max_length
    assert len(segment_ids) == max_length

    return input_ids, input_mask, segment_ids


def item_example(tokenizer, category_name, item_title, item_description, item_brand, item_price, rank,
                 image, image_shape, image_count):
    item_description = item_description.translate(str.maketrans('', '', string.punctuation)).replace("\n",
                                                                                                     "").replace("???",
                                                                                                                 "").lower()
    item_title = item_title.translate(str.maketrans('', '',
                                                    string.punctuation)).replace("\n", "").replace("???", "").lower()

    (input_ids, input_mask, segment_ids) = to_bert_ids(tokenizer,
                                                       [category_name, item_brand, item_title, item_description],
                                                       max_length=max_seq_length)

    feature = {
        'item_price': _float_feature(item_price),
        'input_ids': _int64_feature(input_ids),
        'input_mask': _int64_feature(input_mask),
        'segment_ids': _int64_feature(segment_ids),
        'rank': _float_feature(rank),
        'image_shape': _int64_feature(image_shape),
        'image_count': _int64_feature(image_count),
        'image': _float_feature(image),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def process_df(df):
    """
    Takes pandas dataframe and returns lists of
    feature columns
    """
    item_titles = df['title'].values.tolist()
    item_descriptions = df['description'].values.tolist()
    item_brands = df['brand'].values.tolist()
    item_prices = df['price'].values.tolist()
    ranks = df['rank_scaled'].values.tolist()
    image_urls = df['image_url'].values.tolist()
    image_urls_high_res = df["image_url_high_res"].values.tolist()
    image_counts = df['image_count'].values.tolist()

    images = []
    image_shapes = []
    skipped_count = 0
    for url, high_res_url, image_count in zip(image_urls, image_urls_high_res, image_counts):
        if image_count == 0:
            image_shapes.append(0)
            images.append(np.zeros([IMG_HEIGHT * IMG_WIDTH * 3]).astype(np.float))
            continue
        image_path = "../" + url
        image_path_high_res = "../" + high_res_url
        try:
            image = tf.io.decode_jpeg(tf.io.read_file(image_path))
        except:
            skipped_count += 1
            image_shapes.append(0)
            images.append(np.zeros([IMG_HEIGHT * IMG_WIDTH * 3]).astype(np.float))
            continue
        image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
        image = NORMALIZATION_LAYER(image)
        image_high_res = tf.io.decode_jpeg(tf.io.read_file(image_path_high_res))
        image_shapes.append(image_high_res.shape[0])
        try:
            images.append(np.array(image).reshape(IMG_HEIGHT * IMG_WIDTH * 3))
        except:
            images.append(np.zeros([IMG_HEIGHT * IMG_WIDTH * 3]).astype(np.float))
            print(image.shape)
            continue
    return item_titles, item_descriptions, item_brands, item_prices, ranks, images, image_shapes, image_counts


def create_records(df, tokenizer, output_dir, num_of_records=5, prefix=None):
    """
    Takes a pandas dataframe and number of records to create and creates TFRecords.
    Saves records in output_dir
    """
    df = df.fillna('')

    all_categories = list(set(df.category_id.values.tolist()))

    record_prefix = os.path.join(output_dir, prefix)
    files_per_record = int(len(all_categories) / num_of_records)  # approximate number of examples per record
    chunk_number = 0

    for i in range(0, len(all_categories), files_per_record):
        print("Writing chunk ", str(chunk_number))
        category_chunk = all_categories[i:i + files_per_record]

        if num_of_records == 1:
            record_file = record_prefix + ".tfrecords"
        else:
            record_file = record_prefix + str(chunk_number).zfill(3) + ".tfrecords"

        with tf.io.TFRecordWriter(record_file) as writer:
            for category in tqdm.tqdm(category_chunk):
                category_df = df.loc[df["category_id"] == category]
                category_name = category_df["category"].values.tolist()[0]
                CONTEXT = category_example(category_name)

                EXAMPLES = []
                titles, descriptions, brands, prices, ranks, images, image_shapes, image_counts = process_df(
                    category_df)
                for j in range(len(titles)):
                    EXAMPLES.append(item_example(tokenizer, category_name, titles[j], descriptions[j],
                                                 brands[j], prices[j], ranks[j], images[j], image_shapes[j],
                                                 image_counts[j]))

                ELWC = input_pb2.ExampleListWithContext()
                ELWC.context.CopyFrom(CONTEXT)
                for example in EXAMPLES:
                    example_features = ELWC.examples.add()
                    example_features.CopyFrom(example)

                writer.write(ELWC.SerializeToString())
            chunk_number += 1


def read_and_print_tf_record(target_filename, num_of_examples_to_read):
    filenames = [target_filename]
    tf_record_dataset = tf.data.TFRecordDataset(filenames)
    all_examples = []

    for raw_record in tf_record_dataset.take(num_of_examples_to_read):
        example_list_with_context = input_pb2.ExampleListWithContext()
        example_list_with_context.ParseFromString(raw_record.numpy())
        all_examples.append(example_list_with_context)

    return all_examples


def main():
    # train_data_path = "./data/train.csv"
    # test_data_path = "./data/test.csv"
    train_data_path = "./data/train_with_images.csv"
    test_data_path = "./data/test_with_images.csv"

    train_df = pd.read_csv(train_data_path)
    test_df = pd.read_csv(test_data_path)
    output_dir = f"./tfrecords_with_images_{max_seq_length}/"

    write_records = True

    if write_records:
        tokenizer = tokenization.FullTokenizer(vocab_file=os.path.join(bert_path, "vocab.txt"), do_lower_case=True)
        num_of_records = 10
        print("Creating records for train")
        create_records(train_df, tokenizer, output_dir, num_of_records, prefix="train")
        print("Creating records for test")
        create_records(test_df, tokenizer, output_dir, num_of_records, prefix="test")
    else:
        examples = read_and_print_tf_record(output_dir + "train.tfrecords", 1)
        print(examples)


if __name__ == "__main__":
    main()
