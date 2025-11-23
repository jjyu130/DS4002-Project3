# Investigating Reproducibility of Weather Event Image Classifier


## 1. Software and Platform
- **Platform Used:** Mac was used to write and run scripts. The scripts can be run across platforms.
- **Software Used**: Python 3
 
- **Add-on Packages:**  
  - `pathlib` – filesystem paths and directory handling
  - `shutil` – moving, copying, and deleting files during preprocessing
  - `random` – random sampling for dataset splitting
  - `numpy` – numerical operations
  - `PIL` – image loading, conversion, resizing, preprocessing  
  - `tensorflow` – end-to-end deep learning framework
    - `keras` → `layers`, `models`, `optimizers`, `callbacks`
      - `applications` → `InceptionV3`, `preprocess_input`
    

## 2. Documentation Map
The hierarchy of folders and files contained in this project are as follows:

```text
DS4002-Project3
├── DATA
│   ├── dataset_split/               # Final 80/10/10 train–val–test split for modeling
│   │   ├── train/
│   │   │   ├── cellconvection/
│   │   │   ├── duststorm/
│   │   │   ├── hurricane/
│   │   │   ├── rollconvection/
│   │   │   └── wildfires/
│   │   ├── val/
│   │   │   ├── cellconvection/
│   │   │   ├── duststorm/
│   │   │   ├── hurricane/
│   │   │   ├── rollconvection/
│   │   │   └── wildfires/
│   │   └── test/
│   │       ├── cellconvection/
│   │       ├── duststorm/
│   │       ├── hurricane/
│   │       ├── rollconvection/
│   │       └── wildfires/
│   ├── cleaned_data/                # Standardized RGB images after preprocessing
│   │   ├── cellconvection/
│   │   ├── duststorm/
│   │   ├── hurricane/
│   │   ├── rollconvection/
│   │   └── wildfires/
│   ├── raw_weather_images/          # Original satellite imagery from Harvard Dataverse
│   │   ├── cellconvection/
│   │   ├── duststorm/
│   │   ├── hurricane/
│   │   ├── rollconvection/
│   │   └── wildfires/
│   └── README.md
├── OUTPUT
│   ├── RGB_by_class.png   
│   ├── contrast_by_class.png
│   ├── evaluation_inceptionv3_best.png
│   ├── inceptionv3_best.keras
│   └── inceptionv3_final.png  
├── SCRIPTS
│   ├── 01_preprocess_and_split.py 
│   └── 02_InceptionV3.py 
├── LICENSE.md
└── README.md
```

## 3. Reproducing Our Results
  1. **Set up Python and install required add-on packages**
     - Clone this repository: https://github.com/jjyu130/DS4002-Project3/
     - Ensure you have Python 3 installed on your system.
     - See section 1 for packages needed.
  2. **(Optional) Prepare the dataset**
     - Run `01_preprocess_and_split.py` from the `SCRIPTS` folder, which creates
          `cleaned_data` by processing `raw_weather_images` and splits into `dataset_split`
          in the `DATA` folder.
     - Otherwise, proceed with preprocessed data, which is already provided.
3. **Run model training script**
     - Navigate to the `SCRIPTS` folder.
     - Scripts 2-3 build and train 2 CNN models on the prepared `DATA/dataset_split/` directory.
       Models are saved to `OUTPUT` folder.


 

     
