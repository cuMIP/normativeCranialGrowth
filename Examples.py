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

## Create internal cranial surface mesh from the external surface and thickness map
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