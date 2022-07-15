# Normative Cranial Growth
This is a repository for the [Data-driven Normative Reference of Pediatric Cranial Bone Development](https://github.com/cuMIP/normativeCranialGrowth).

This repository contains the normative intensity (``IntensityModel``), thickness (``ThicknessModel``), and shape (``ShapeModel``) models described in the manuscript. 

This repository also provides example scripts to generate synthetic normative cranial bone surface meshes based on age, sex, and number of the number of standard deviations away from the the average PCA coefficients. 

The Excels files contains the model-generated average temporal development of bone thickness (``BoneThickness.csv``), cephalic index (``CI.csv``), intracranial volume (``ICV.csv``), and bone surface areas (``SurfaceArea.csv``) for males and females, respectively.


## Dependencies:
- [Python](python.org)
- [NumPy](https://numpy.org/install/)
- [SimpleITK](https://simpleitk.org/)
- [VTK](https://pypi.org/project/vtk/)

    *Once Python is installed, each of these packages can be downloaded using [Python pip](https://pip.pypa.io/en/stable/installation/)*


## Using the code

### Quick summary
**Input**: age (in years); sex (binary indicator for male); nStdThickness, nStdIntensity, and nStdShape (arrays where elements represent the number of standard deviations to each of the average PCA coefficient).

**Output**: VTK PolyData for the external and internal cranial surface, with information of local bone thickness and density.

### Code example
Following is an example for generating an cranial surface instance for a female with 1 year of age, and +1 std to the second PCA coefficient for all three metrics:
```python
import Tools
import pickle
import SimpleITK as sitk
import vtk
import numpy as np
import os

inputPath = './'
## import Model data
with open(os.path.join(inputPath, 'ThicknessModel'), "rb") as fp:
    ThicknessModel = pickle.load(fp)

with open(os.path.join(inputPath, 'IntensityModel'), "rb") as fp:
    IntensityModel = pickle.load(fp)

with open(os.path.join(inputPath, 'ShapeModel'), "rb") as fp:
    ShapeModel = pickle.load(fp)

## target age and sex, and standard deviations from average principal components
age = 1 ## age in years
sex = 0 # 1 for male, 0 for female 
nStdThickness = np.zeros(ThicknessModel[2].shape[0])
nStdIntensity = np.zeros(IntensityModel[2].shape[0])
nStdShape = np.zeros(ShapeModel[2].shape[0])
## + 1 std away for the second component
nStdThickness[1] = 1
nStdIntensity[1] = 1
nStdShape[1] = 1

## Read mask image and average bone segmentation image
MaskImage = sitk.ReadImage(os.path.join(inputPath, 'SphericalMaskImage.mha'))
AverageSegmentationImage = sitk.ReadImage(os.path.join(inputPath, 'averageBoneSegmentationSphericalImage.mha'))

## Constructing average spherical maps with standard deviation 1 for shape, thickness and intensity

CoordinateMap = Tools.ConstrucPredictionSphericalMapsFromPCAModel(ShapeModel, age, sex, MaskImage = MaskImage, nStd = nStdShape, Coordinates=True)
CoordinateMap.CopyInformation(AverageSegmentationImage)
ThicknessMap = Tools.ConstrucPredictionSphericalMapsFromPCAModel(ThicknessModel, age, sex, MaskImage = MaskImage, nStd = nStdThickness, Coordinates=False)
IntensityMap = Tools.ConstrucPredictionSphericalMapsFromPCAModel(IntensityModel, age, sex, MaskImage = MaskImage, nStd = nStdIntensity, Coordinates=False)

## Construct external cranial surface mesh with thickness and intensity information
referneceImage = MaskImage
referneceImage.CopyInformation(AverageSegmentationImage)
ExternalSurface = Tools.ConstructCranialSurfaceMeshFromSphericalMaps(CoordinateMap, referenceImage=referneceImage,
    intensityImageDict={'Density':IntensityMap, 'Thickness': ThicknessMap, 'BoneLabel': AverageSegmentationImage}, subsamplingFactor=1,verbose=True)

## Create internal cranial surface mesh with external surface and thickness map
InternalSurface = Tools.CreateInternalSurfaceFromExternalSurface(MaskImage, ExternalSurface=ExternalSurface)

## save the meshes
writer = vtk.vtkXMLPolyDataWriter()
writer.SetInputData(ExternalSurface)
writer.SetFileName(os.path.join(inputPath, 'ExternalSurface.vtp'))
writer.Update()

writer = vtk.vtkXMLPolyDataWriter()
writer.SetInputData(InternalSurface)
writer.SetFileName(os.path.join(inputPath, 'InternalSurface.vtp'))
writer.Update()
```
*When using this code, be sure not to modify ```SphericalMaskImage.mha``` and ```averageBoneSegmentationSphericalImage.mha```, as they define the area of interest for the model and store necessary information for the generation of images and meshes.*

### The workflow

- The **ConstrucPredictionSphericalMapsFromPCAModel** function creates predicted 2D spherical maps based on age, sex, and std to the average PCA coefficients.
- The **ConstructCranialSurfaceMeshFromSphericalMaps** function constructs the 3D VTP PolyData of the external cranial surface mesh based ont the 2D speherical maps.
- The **CreateInternalSurfaceFromExternalSurface** function creates the internal surface mesh based on external surface and the predicted local thicknes.

If you have any questions, please email Jiawei Liu at jiawei.liu@cuanschutz.edu
