import numpy as np
import SimpleITK as sitk
import vtk

def Arcsinh(x, gender, a, b, c, d, e, g):
    """
        Arcsinh-based regression function
    """
    y = a + b*gender + (c)* np.arcsinh(d*x) + e*x + g*x*gender
    return y, [1, gender, np.arcsinh(d*x), c*(x+d*x**2/np.sqrt(1+d**2*x**2))/(d*x+np.sqrt(1+d**2*x**2)), x, x*gender]

def ConstrucPredictionSphericalMapsFromPCAModel(Model, age, gender, MaskImage, nStd = None, functional = Arcsinh, Coordinates = False):

    """
        Generate spherical maps predictions from PCA regression model outputs
    Parameters
    ----------
    Model: list
        PCA models
    age: float
        Target age
    gender: int
        Target gender (male:1, female:0)
    MaskImage: sitk.Image
        Mask image
    nStd: np.array
        Vector representing the deviation away from the average for each principal component
    functional: function
        Regresion function
    Coordinates: bool
        Generate coordinate maps or not

    Returns
    -------
    sitk.Image
        Shape of 500x500 or 500x500x3
    """

    if nStd is None:
        nStd = np.zeros(Model[2].shape[0])
    
    if len(nStd) != Model[2].shape[0]:
        raise ValueError("nStd needs to have the same length as the number of principal components kept (n={})".format(Model[2].shape[0]))
        return()

    Mask = sitk.GetArrayFromImage(MaskImage)
    n = (Mask==1).sum()
    if Coordinates:
        PredImg = -np.ones((500,500,3))
    else:
        PredImg = -np.ones((500,500))
        # PredImg = np.array(np.tile(np.expand_dims(sitk.GetArrayViewFromImage(referenceImage), 2), (1, 1, 3) ), dtype=np.float32)
        # PredImg[:] = 0
    PCAParams = np.zeros(Model[2].shape[0])
    for i in range(Model[2].shape[0]):
        PCAParams[i] = functional(age, gender, *Model[2][i,:])[0] + nStd[i] * Model[3][i].predict(np.array([age,gender]).reshape(1,-1))

    vals = Model[0] + np.dot(PCAParams, Model[1])
    if not Coordinates:
        PredImg[Mask==1] = vals
        PredImg = sitk.GetImageFromArray(PredImg)
    else:
        PredImg[Mask==1] = vals.reshape((n,3))
        PredImg = sitk.GetImageFromArray(PredImg, isVector=True)

    return(PredImg)

def ConstructCranialSurfaceMeshFromSphericalMaps(euclideanCoordinateSphericalMapImage, referenceImage, intensityImageDict=None, subsamplingFactor=5, verbose=False):
    """
    Recsontructs a surface model using the Euclidean coordinates represented as a spherical map image 

    Parameters
    ----------
    euclideanCoordinateSphericalMapImage: sitkImage
        Spherical map image with the Euclidean coordinates of the surface model
    referenceImage: sitkImage
        A reference image with with pixels set to 0 in the background
    intensityImageDict: dictionary {arrayName: image}
        A dictionary

    Returns
    -------
    vtk.vtkPolyData:
        The reconstructed mesh    
    """

    bullsEyeImageArray = sitk.GetArrayViewFromImage(euclideanCoordinateSphericalMapImage)
    referenceImageArray = sitk.GetArrayViewFromImage(referenceImage)

    filter = vtk.vtkPlaneSource()
    filter.SetOrigin((-1, -1, 0))
    filter.SetPoint1((1, -1, 0))
    filter.SetPoint2((-1, 1, 0))
    filter.SetXResolution(int(euclideanCoordinateSphericalMapImage.GetSize()[0] / subsamplingFactor))
    filter.SetYResolution(int(euclideanCoordinateSphericalMapImage.GetSize()[1] / subsamplingFactor))
    filter.Update()
    mesh = filter.GetOutput()

    filter = vtk.vtkTriangleFilter()
    filter.SetInputData(mesh)
    filter.Update()
    mesh = filter.GetOutput()

    filter = vtk.vtkCleanPolyData()
    filter.SetInputData(mesh)
    filter.Update()
    mesh = filter.GetOutput()

    insideArray = vtk.vtkIntArray()
    insideArray.SetName('Inside')
    insideArray.SetNumberOfComponents(1)
    insideArray.SetNumberOfTuples(mesh.GetNumberOfPoints())
    mesh.GetPointData().AddArray(insideArray)


    intensityImageArrays = []
    intensityArrays = []
    if intensityImageDict is not None:

        for key, val in intensityImageDict.items():

            intensityImageArrays += [sitk.GetArrayViewFromImage(val)]
            intensityArrays += [vtk.vtkFloatArray()]
            intensityArrays[-1].SetName(key)
            intensityArrays[-1].SetNumberOfComponents(1)
            intensityArrays[-1].SetNumberOfTuples(mesh.GetNumberOfPoints())
            mesh.GetPointData().AddArray(intensityArrays[-1])

    # Figuring out what is inside or outside
    for p in range(mesh.GetNumberOfPoints()):
            
        if verbose:
            print('{} / {}.'.format(p, mesh.GetNumberOfPoints()), end='\r')

        coords = mesh.GetPoint(p)

        try:
            imageIndex = referenceImage.TransformPhysicalPointToIndex((coords[0], coords[1]))

            mesh.GetPoints().SetPoint(p, euclideanCoordinateSphericalMapImage.GetPixel(imageIndex))

            if referenceImageArray[imageIndex[1], imageIndex[0]] > 0:
                insideArray.SetTuple1(p, 1)
                
                if intensityImageDict is not None:

                    for arrayId in range(len(intensityArrays)):
                        intensityArrays[arrayId].SetTuple1(p, intensityImageArrays[arrayId][imageIndex[1], imageIndex[0]])
            else:
                insideArray.SetTuple1(p, 0)
        except:
            insideArray.SetTuple1(p, 0)

    filter = vtk.vtkThreshold()
    filter.SetInputData(mesh)
    filter.ThresholdByUpper(0.9)
    filter.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, 'Inside')
    filter.Update()
    mesh = filter.GetOutput()

    filter = vtk.vtkGeometryFilter()
    filter.SetInputData(mesh)
    filter.Update()
    mesh = filter.GetOutput()

    filter = vtk.vtkPolyDataNormals()
    filter.SetInputData(mesh)
    filter.ComputeCellNormalsOn()
    filter.ComputePointNormalsOn()
    filter.NonManifoldTraversalOff()
    filter.AutoOrientNormalsOn()
    filter.ConsistencyOn()
    filter.SplittingOff()
    filter.Update()
    mesh = filter.GetOutput()

    mesh.GetPointData().RemoveArray("Inside")

    return mesh

def CreateInternalSurfaceFromExternalSurface(MaskImage, ExternalSurface):
    """
    Calculate cephalic index from cranial bone mesh. 

    Parameters
    ----------
    MaskImage: sitk.Image
        Mask image
    ExternalSurface: vtk.vtkPolyData
        External surface mesh
    ExternalSurfaceMap: sitk.Image
        Spherical map image of external surface coordinates
    ThicknessMap: sitk.Image
        Spherical map image of thickness
    mapping: np.array
        Mapping between ExternalSurface points and ExternalSurfaceMap

    Returns
    -------
    float:
        vtk.vtkPolyData 
    """
    Mask = sitk.GetArrayFromImage(MaskImage)
    Masked = np.where(Mask == 1)
    masked = np.append(Masked[0],Masked[1])
    masked = np.reshape(masked,(-1,2),order='F')
    # MeshCoords = np.array(ExternalSurface.GetPoints().GetData())
    NormalVectors = np.array(ExternalSurface.GetPointData().GetNormals())
    
    InternalSurface = vtk.vtkPolyData()
    InternalSurface.DeepCopy(ExternalSurface)
    ThicknessArray = np.array(ExternalSurface.GetPointData().GetArray('Thickness'))
    for p in range(ExternalSurface.GetNumberOfPoints()):
        # InternalSurface.GetPoints().SetPoint(p, InternalSurface.GetPoint(p) - NormalVectors[p,:] * ThicknessMap[masked[mapping[p],0], masked[mapping[p],1]])
        InternalSurface.GetPoints().SetPoint(p, InternalSurface.GetPoint(p) - NormalVectors[p,:] * ThicknessArray[p])
    
    return(InternalSurface)

