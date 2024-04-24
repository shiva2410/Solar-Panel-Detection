import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET


class XMLtoCSVConverter:
    def __init__(self, path):
        # Initialize the path
        self.path = path

    def xml_to_csv(self):
        # Initialize the list for XML data
        xml_list = []
        # Loop through each XML file in the directory
        for xml_file in glob.glob(self.path + "/*.xml"):
            # Parse the XML file
            tree = ET.parse(xml_file)
            root = tree.getroot()
            # Loop through each 'object' element in the XML file
            for member in root.findall("object"):
                # Extract the necessary information
                value = (
                    root.find("filename").text,
                    int(root.find("size")[0].text),
                    int(root.find("size")[1].text),
                    member[0].text,
                    int(member[4][0].text),
                    int(member[4][1].text),
                    int(member[4][2].text),
                    int(member[4][3].text),
                )
                # Append the information to the list
                xml_list.append(value)
        # Define the column names for the DataFrame
        column_name = ["filename", "width", "height", "class", "xmin", "ymin", "xmax", "ymax"]
        # Create a DataFrame from the list
        xml_df = pd.DataFrame(xml_list, columns=column_name)
        return xml_df


def main():
    # Define the folders to process
    folders = ["train", "test"]
    # Process each folder
    for folder in folders:
        image_path = os.path.join(os.getcwd(), ("images/" + folder))
        converter = XMLtoCSVConverter(image_path)
        xml_df = converter.xml_to_csv()
        xml_df.to_csv(("images/" + folder + "_labels.csv"), index=None)
        print("Successfully converted xml to csv.")


# Run the main function
if __name__ == "__main__":
    main()
