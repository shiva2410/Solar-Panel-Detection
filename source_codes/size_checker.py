# Import necessary libraries
import argparse
import os
import shutil
from xml.etree import ElementTree
from termcolor import colored
from glob import glob


# Define a class to check bounding box sizes
class BoundingBoxChecker:
    def __init__(self, train_directory, test_directory):
        # Initialize with training and testing directories and set up argument parser for command line options
        self.train_directory = train_directory
        self.test_directory = test_directory
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument(
            "--move",
            help="Move incorrect xml and images to a wrong_data folder inside each folder",
            action="store_true",
        )
        self.args = self.parser.parse_args()
        self.is_data_correct = True

    def check_directories(self):
        # Method to check if directories exist
        if not os.path.isdir(self.train_directory) or not os.path.isdir(self.test_directory):
            print(colored("[!]", "yellow", attrs=["bold"]), colored("The training or test directories do not exist"))
            exit(1)
        else:
            print(colored("[Ok]", "green"), colored("Directories exist"))

    def check_sizes(self):
        # Method to check bounding box sizes in xml files
        for directory in [self.train_directory, self.test_directory]:
            if self.args.move and not os.path.isdir(directory + "/wrong_data"):
                os.makedirs(directory + "/wrong_data")

            for file in glob(directory + "/*.xml"):
                xml_file = ElementTree.parse(file)
                boxes = xml_file.findall("object/bndbox")
                for box in boxes:
                    xmin, ymin, xmax, ymax = [int(child.text) for child in box]
                    x_value = xmax - xmin
                    y_value = ymax - ymin

                    if x_value < 33 or y_value < 33:
                        print(
                            colored("[!]", "red"),
                            f"File {file} contains a bounding box smaller than 32 in height or width",
                        )
                        print(colored("xmax - xmin", "yellow", attrs=["bold"]), x_value)
                        print(colored("ymax - ymin", "yellow", attrs=["bold"]), y_value)
                        self.is_data_correct = False

                        if self.args.move:
                            wrong_picture = xml_file.find("filename").text
                            try:
                                shutil.move(file, directory + "/wrong_data/")
                                shutil.move(directory + "/" + wrong_picture, directory + "/wrong_data/")
                                print(colored("Files moved to" + directory + "/wrong_data", "blue"))
                            except Exception as e:
                                print(colored(e, "blue"))

    def run(self):
        # Method to run the checks
        self.check_directories()
        self.check_sizes()

        if self.is_data_correct:
            print(colored("[Ok]", "green"), "All bounding boxes are equal or larger than 32 :-)")
            try:
                os.rmdir(self.train_directory + "/wrong_data")
                os.rmdir(self.test_directory + "/wrong_data")
            except OSError:
                print(
                    colored("[Info]", "blue"), "Directories wrong_data were not removed because they contain some files"
                )
        else:
            print()
            print(colored("[Error]", "red"), " (╯°□°)╯ ┻━┻")


if __name__ == "__main__":
    # Main function to create an instance of the BoundingBoxChecker class and run the checks
    checker = BoundingBoxChecker("./images/train", "./images/test")
    checker.run()
