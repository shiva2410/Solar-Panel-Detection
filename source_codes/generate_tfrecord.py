"""
Usage:
  # From tensorflow/models/
  # Create train data:
  python generate_tfrecord.py --csv_input=images/train_labels.csv --image_dir=images/train --output_path=train.record

  # Create test data:
  python generate_tfrecord.py --csv_input=images/test_labels.csv  --image_dir=images/test --output_path=test.record
"""

# Import necessary libraries for future compatibility, file handling, and image processing
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import pandas as pd
import tensorflow as tf

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple

# Define command line arguments for input CSV, image directory, and output path
command_line_flags = tf.app.flags
command_line_flags.DEFINE_string("csv_input", "", "Path to the CSV input")
command_line_flags.DEFINE_string("image_dir", "", "Path to the image directory")
command_line_flags.DEFINE_string("output_path", "", "Path to output TFRecord")
FLAGS = command_line_flags.FLAGS


# Function to convert class text to int
def class_text_to_int(row_label):
    if row_label == "solar":
        return 1
    elif row_label == "roof":
        return 2
    else:
        return None


# Function to split the dataframe into groups based on filename
def split_dataframe(df, group):
    data = namedtuple("data", ["filename", "object"])
    grouped_data = df.groupby(group)
    return [
        data(filename, grouped_data.get_group(x))
        for filename, x in zip(grouped_data.groups.keys(), grouped_data.groups)
    ]


# Function to create a tensorflow example from grouped data
def create_tf_record(group, image_path):
    # Open the image file
    with tf.gfile.GFile(os.path.join(image_path, "{}".format(group.filename)), "rb") as file:
        encoded_image_file = file.read()
    encoded_image_data = io.BytesIO(encoded_image_file)
    image = Image.open(encoded_image_data)
    image_width, image_height = image.size

    # Encode the filename
    encoded_filename = group.filename.encode("utf8")
    image_format = b"jpg"
    x_min_values = []
    x_max_values = []
    y_min_values = []
    y_max_values = []
    class_texts = []
    class_ids = []

    # Iterate over each row in the group object
    for index, row in group.object.iterrows():
        x_min_values.append(row["xmin"] / image_width)
        x_max_values.append(row["xmax"] / image_width)
        y_min_values.append(row["ymin"] / image_height)
        y_max_values.append(row["ymax"] / image_height)
        class_texts.append(row["class"].encode("utf8"))
        class_ids.append(class_text_to_int(row["class"]))

    # Create a tensorflow example
    tf_record = tf.train.Example(
        features=tf.train.Features(
            feature={
                "image/height": dataset_util.int64_feature(image_height),
                "image/width": dataset_util.int64_feature(image_width),
                "image/filename": dataset_util.bytes_feature(encoded_filename),
                "image/source_id": dataset_util.bytes_feature(encoded_filename),
                "image/encoded": dataset_util.bytes_feature(encoded_image_file),
                "image/format": dataset_util.bytes_feature(image_format),
                "image/object/bbox/xmin": dataset_util.float_list_feature(x_min_values),
                "image/object/bbox/xmax": dataset_util.float_list_feature(x_max_values),
                "image/object/bbox/ymin": dataset_util.float_list_feature(y_min_values),
                "image/object/bbox/ymax": dataset_util.float_list_feature(y_max_values),
                "image/object/class/text": dataset_util.bytes_list_feature(class_texts),
                "image/object/class/label": dataset_util.int64_list_feature(class_ids),
            }
        )
    )
    return tf_record


def main(_):
    # Create a TFRecordWriter
    tf_record_writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

    # Get the path to the image directory
    image_directory_path = os.path.join(os.getcwd(), FLAGS.image_dir)

    # Read the CSV input
    csv_input_data = pd.read_csv(FLAGS.csv_input)

    # Split the CSV input data by filename
    grouped_input_data = split(csv_input_data, "filename")

    # For each group in the grouped input data, create a TF example and write it to the TFRecord
    for group in grouped_input_data:
        tf_example = create_tf_example(group, image_directory_path)
        tf_record_writer.write(tf_example.SerializeToString())

    # Close the TFRecordWriter
    tf_record_writer.close()

    # Print the path to the output TFRecord
    output_tf_record_path = os.path.join(os.getcwd(), FLAGS.output_path)
    print("Successfully created the TFRecords: {}".format(output_tf_record_path))


if __name__ == "__main__":
    tf.app.run()
