# Satellite Weather Event Image Dataset

## Data Summary
The data used in this project is the **“9,081 images dataset for: Detection of weather
events in optical satellite data using deep convolutional neural networks”**,
which is hosted publicly on Harvard Dataverse [1]. It contains 9,081 color
optical satellite images labeled into five weather-event categories: tropical
cyclones (hurricanes), cell convection, roll convection, wildfires, and dust storms.
Each image represents a single weather event instance, captured by Earth-observing
satellites at varying spatial resolutions.

The dataset is provided as 5 folders of JPEG image files, where each subfolder 
corresponds to one weather event category. Image resolutions vary substantially
depending on the satellite of origin, ranging from coarse MODIS-level imagery to
finer-resolution sensors. No additional metadata (beyond filenames and folder labels)
is provided.

For this project, we used a preprocessing and dataset-splitting script
(01_preprocess_and_split.py in the SCRIPTS directory) to (1) remove duplicate
images, (2) rescale images to uniform dimensions, (3) convert files to RGB format,
and (4) generate an 80/10/10 train–validation–test directory structure. The 6224 unique,
cleaned, and standardized images are saved to `DATA/cleaned_data/`, which maintains the same
five-folder class structure. The 80/10/10 train–validation–test split is stored in
`DATA/dataset_split/`.

## Provenance
The original images were captured by seven NASA, ESA, and NOAA satellites beginning
in 1999. The curated dataset hosted on Harvard Dataverse was assembled by researchers
Li & Momen, who manually selected images from publicly available satellite archives.
They filtered raw satellite images for quality and manually assigned each image to 
one of five weather event categories. 

## License
The dataset is distributed under the Creative Commons Public Domain Dedication (CC0 1.0).
This license grants unrestricted rights to copy, modify, distribute, and use the
dataset for any purpose, including research and commercial applications. 

## Ethical Statements
Although the dataset is composed of openly available satellite imagery, several
ethical considerations still apply. First, transparency around data provenance is
important, as the images come from multiple government agencies and may have varying
usage restrictions or attribution requirements. Second, because the dataset was
manually selected and labeled, users should be aware of potential human bias or
inconsistency in classifications. Third, any models developed from this dataset
should be used responsibly, particularly in contexts involving disaster response
or public communication, where misclassification of weather events could have
real-world consequences. Finally, because satellite imagery can inadvertently
capture human activity or sensitive locations, researchers should remain mindful
of privacy and security implications even when working with seemingly benign
environmental data. To account for these limitations, we will report any assumptions
made during preprocessing, document known sources of labeling uncertainty, and
communicate model performance with appropriate caveats. We will also ensure all
data use complies with licensing requirements and conduct periodic reviews to
confirm that no sensitive content is inadvertently included in the analysis.


## Data Dictionary
*We use the original folder-based image organization for modeling, rather than implementing
a manifest csv as initially intended. Thus, the dataset does not contain structured
tabular features. The following elements effectively constitute the metadata:*

| **Element**   | **Type** | **Description**                                                                                                                                                   |
| --------------| ---------| ------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `image file`  | JPG      | Raw or cleaned RGB satellite image representing a single weather event instance.                                                                                  |
| `filepath`    | String   | Relative path to each image within the dataset directory (e.g., `DATA/cleaned_data/duststorm/501.jpg`).                                                           |
| `folder name` | String   | Weather event category associated with the image (`cellconvection`, `duststorm`, `hurricane`, `rollconvection`, `wildfires`). The folder acts as the class label. |

## Explanatory Plots
![OUTPUT/RGB_by_class.png](OUTPUT/RGB_by_class.png)

**Figure 1**. Mean RGB channel values for each weather event category in the dataset.

![contrast_by_class.png](contrast_by_class.png)

**Figure 2**. Pixel-level contrast distributions across weather event categories.

## References
[1]	Y. Li and M. Momen, “9081 images dataset for: Detection of weather events in optical
satellite data using deep convolutional neural networks.” Harvard Dataverse, 2024.
doi: 10.7910/DVN/PUIHVC. 
