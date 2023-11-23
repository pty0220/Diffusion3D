import sys
import os
import SimpleITK as sitk
import numpy as np
# import tkinter.filedialog as tk
import warnings
import SimpleITK as sitk
import cv2
from scipy import ndimage
import matplotlib.pyplot as plt


l2n = lambda l: np.array(l)
n2l = lambda n: list(n)

# from vtk.util import numpy_support as ns


class niiCook():

    def __init__(self):
        self.dummy = 0

    def readITK(self, image):

        spacing   = image.GetSpacing()
        origin    = image.GetOrigin()  ## bounds
        dimension = image.GetSize()
        direction = image.GetDirection()
        extent = (0, dimension[0] - 1, 0, dimension[1] - 1, 0, dimension[2] - 1)

        array = sitk.GetArrayFromImage(image)

        self.itkImage = image
        self.spacing = spacing
        self.origin = origin
        self.dimension = dimension
        self.extent = extent
        self.array = array
        self.direction = direction

    def readSavedFile(self, filePath):

        if filePath[-2:] == "gz":
            reader = sitk.ImageFileReader()
            reader.SetImageIO("NiftiImageIO")
            reader.SetFileName(filePath)

        elif filePath[-3:] == "nii":
            reader = sitk.ImageFileReader()
            reader.SetImageIO("NiftiImageIO")
            reader.SetFileName(filePath)

        elif filePath[-4:] == "nrrd":
            reader = sitk.ImageFileReader()
            reader.SetImageIO("NrrdImageIO")
            reader.SetFileName(filePath)

        else:
            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(filePath)
            reader.SetFileNames(dicom_names)

        image     = reader.Execute()
        spacing   = image.GetSpacing()
        origin    = image.GetOrigin()  ## bounds
        dimension = image.GetSize()
        direction = image.GetDirection()
        extent = (0, dimension[0] - 1, 0, dimension[1] - 1, 0, dimension[2] - 1)

        array = sitk.GetArrayFromImage(image)

        self.direction = direction
        self.itkImage = image
        self.spacing = spacing
        self.origin = origin
        self.dimension = dimension
        self.extent = extent
        self.array = array

    def cropVolume(self, dimension, origin, spacing, original_image = 0):

        reference_image = sitk.Image(int(dimension[0]), int(dimension[1]), int(dimension[2]), sitk.sitkFloat32)
        reference_image.SetSpacing(spacing)
        reference_image.SetOrigin(origin)
        reference_image[:,:,:] = 0

        if original_image == 0:
            original_image = self.itkImage

        rigid_euler = sitk.Euler3DTransform()
        interpolator = sitk.sitkCosineWindowedSinc
        default_value = -1000.0

        resampler = sitk.ResampleImageFilter()
        resampler.SetInterpolator(interpolator)
        resampler.SetReferenceImage(reference_image)
        resampler.SetOutputPixelType(sitk.sitkFloat32)
        resampler.SetDefaultPixelValue(default_value)
        resampler.SetTransform(rigid_euler)
        resampler.SetNumberOfThreads(15)
        crop_image = resampler.Execute(original_image)

        #crop_image = sitk.Resample(original_image, reference_image, rigid_euler, interpolator, default_value)

        return reference_image, crop_image

    def resample(self, original_image):

        reference_image = sitk.GetImageFromArray(self.array, sitk.sitkInt8)
        reference_image.SetSpacing(self.spacing)
        reference_image.SetOrigin(self.origin)
        reference_image[:,:,:] = 0

        original_image = original_image/max(original_image)
        original_image = original_image*5000


        rigid_euler = sitk.Euler3DTransform()
        interpolator = sitk.sitkCosineWindowedSinc
        default_value = 0
        resample = sitk.Resample(original_image, reference_image, rigid_euler, interpolator, default_value)

        cook = niiCook()
        cook.readITK(resample)
        arr = cook.array
        arr[arr<3000] = 0


        # cook = niiCook()
        # cook.readITK(reference_image)
        resample = cook.makeITK(arr)

        return resample

    def makeSimulationDomain(self, simul_spacing, focal_length, target_pos, make_even = True):

        simul_spacing = l2n(simul_spacing)*1000

        # for Optimal position 1.2

        max_point = target_pos + (focal_length * 1.2)
        min_point = target_pos - (focal_length * 1.2)

        bound = np.abs(max_point - min_point)
        domain = np.round(bound / simul_spacing)
        if make_even:
            domain = domain - domain % 10

        reference_image, crop_image = self.cropVolume(domain, min_point, simul_spacing)
        crop_array = sitk.GetArrayFromImage(crop_image)

        return crop_image, crop_array


    def makeSimulationDomain_rotate(self, simul_spacing, tran_pos, target_pos, focal_length, width, make_even=True):

        simul_spacing = l2n(simul_spacing) * 1000

        ####################################################################
        # Make reference image
        Nx = np.ceil(width/simul_spacing[0]) + 30
        Ny = Nx
        Nz = np.ceil((focal_length*1.7)/simul_spacing[2])

        domain = l2n([Nx, Ny, Nz])
        domain = domain - domain%10

        x_end = -simul_spacing[0] * domain[0] / 2
        y_end = -simul_spacing[1] * domain[1] / 2
        z_end = -simul_spacing[2] * 5

        grid_origin = (x_end, y_end, z_end)

        reference_image = sitk.Image(int(domain[0]), int(domain[1]), int(domain[2]), sitk.sitkFloat32)
        reference_image.SetSpacing(simul_spacing)
        reference_image.SetOrigin(grid_origin)
        reference_image[:, :, :] = 0


        ####################################################################
        # Make Transform matrix
        dir = tran_pos - target_pos

        n_z = np.sqrt(dir[0] * dir[0] + dir[1] * dir[1])
        if n_z == 0:
            transform_matrix = np.eye(3)
        else:
            rotate_z = l2n(((dir[0] / n_z, dir[1] / n_z, 0),
                            (-dir[1] / n_z, dir[0] / n_z, 0),
                            (0, 0, 1)))

            dir2 = np.dot(rotate_z, np.transpose(dir))
            n_y = np.sqrt(dir2[0] * dir2[0] + dir2[2] * dir2[2])
            rotate_y = l2n(((-dir2[2] / n_y, 0, dir2[0] / n_y),
                            (0, 1, 0),
                            (-dir2[0] / n_y, 0, -dir2[2] / n_y)))

            transform_matrix = np.dot(rotate_y, rotate_z)
            transform_matrix = np.linalg.inv(transform_matrix)


        move2tran = sitk.TranslationTransform(3, tran_pos)

        ####################################################################
        # Apply function
        rigid_euler = sitk.Euler3DTransform()
        rigid_euler.SetMatrix(transform_matrix.flatten())  # rotate rely on transducer axes axis
        rigid_euler.SetTranslation(move2tran.GetOffset())  # move to target
        interpolator = sitk.sitkCosineWindowedSinc

        # Resampling
        default_value = 0
        resampler = sitk.ResampleImageFilter()
        resampler.SetInterpolator(interpolator)
        resampler.SetReferenceImage(reference_image)
        resampler.SetOutputPixelType(sitk.sitkFloat32)
        resampler.SetDefaultPixelValue(default_value)
        resampler.SetTransform(rigid_euler)
        resampler.SetNumberOfThreads(10)
        if self.array.max()==1:
            crop_image = resampler.Execute(self.itkImage*2000)
            crop_image = crop_image/2000
        else:
            crop_image = resampler.Execute(self.itkImage)

        # Re-transform to origin image
        grid_origin_tran = np.dot(transform_matrix, grid_origin)
        crop_image.SetOrigin(grid_origin_tran+tran_pos)
        crop_image.SetDirection(n2l(transform_matrix.flatten()))

        # Get array
        crop_array = sitk.GetArrayFromImage(crop_image)

        return crop_array, crop_image


    def makeSimulationDomain_rotate_long(self, simul_spacing, tran_pos, target_pos, focal_length, width, make_even=True):

        simul_spacing = l2n(simul_spacing) * 1000

        ####################################################################
        # Make reference image
        Nx = np.ceil(width/simul_spacing[0]) + 30
        Ny = Nx
        Nz = np.ceil((focal_length*2.7)/simul_spacing[2])

        domain = l2n([Nx, Ny, Nz])
        domain = domain - domain%10

        x_end = -simul_spacing[0] * domain[0] / 2
        y_end = -simul_spacing[1] * domain[1] / 2
        z_end = -simul_spacing[2] * 50

        grid_origin = (x_end, y_end, z_end)

        reference_image = sitk.Image(int(domain[0]), int(domain[1]), int(domain[2]), sitk.sitkFloat32)
        reference_image.SetSpacing(simul_spacing)
        reference_image.SetOrigin(grid_origin)
        reference_image[:, :, :] = 0


        ####################################################################
        # Make Transform matrix
        dir = tran_pos - target_pos

        n_z = np.sqrt(dir[0] * dir[0] + dir[1] * dir[1])
        if n_z == 0:
            transform_matrix = np.eye(3)
        else:
            rotate_z = l2n(((dir[0] / n_z, dir[1] / n_z, 0),
                            (-dir[1] / n_z, dir[0] / n_z, 0),
                            (0, 0, 1)))

            dir2 = np.dot(rotate_z, np.transpose(dir))
            n_y = np.sqrt(dir2[0] * dir2[0] + dir2[2] * dir2[2])
            rotate_y = l2n(((-dir2[2] / n_y, 0, dir2[0] / n_y),
                            (0, 1, 0),
                            (-dir2[0] / n_y, 0, -dir2[2] / n_y)))

            transform_matrix = np.dot(rotate_y, rotate_z)
            transform_matrix = np.linalg.inv(transform_matrix)


        move2tran = sitk.TranslationTransform(3, tran_pos)

        ####################################################################
        # Apply function
        rigid_euler = sitk.Euler3DTransform()
        rigid_euler.SetMatrix(transform_matrix.flatten())  # rotate rely on transducer axes axis
        rigid_euler.SetTranslation(move2tran.GetOffset())  # move to target
        interpolator = sitk.sitkCosineWindowedSinc

        # Resampling
        default_value = 0
        resampler = sitk.ResampleImageFilter()
        resampler.SetInterpolator(interpolator)
        resampler.SetReferenceImage(reference_image)
        resampler.SetOutputPixelType(sitk.sitkFloat32)
        resampler.SetDefaultPixelValue(default_value)
        resampler.SetTransform(rigid_euler)
        resampler.SetNumberOfThreads(10)

        if self.array.max()==1:
            crop_image = resampler.Execute(self.itkImage*2000)
            crop_image = crop_image/2000
        else:
            crop_image = resampler.Execute(self.itkImage)

        # Re-transform to origin image
        grid_origin_tran = np.dot(transform_matrix, grid_origin)
        crop_image.SetOrigin(grid_origin_tran+tran_pos)
        crop_image.SetDirection(n2l(transform_matrix.flatten()))

        # Get array
        crop_array = sitk.GetArrayFromImage(crop_image)

        return crop_array, crop_image


    def makeSimulationDomain_rotate_large(self, simul_spacing, tran_pos, target_pos, focal_length, width, make_even=True):

        simul_spacing = l2n(simul_spacing) * 1000

        ####################################################################
        # Make reference image
        Nx = 200# np.ceil(width/simul_spacing[0]) + 30
        Ny = 200# Nx
        Nz = 200# np.ceil((focal_length*2.7)/simul_spacing[2])

        domain = l2n([Nx, Ny, Nz])
        domain = domain - domain%10

        x_end = -simul_spacing[0] * domain[0] / 2
        y_end = -simul_spacing[1] * domain[1] / 2
        z_end = -simul_spacing[2] * 20

        grid_origin = (x_end, y_end, z_end)

        reference_image = sitk.Image(int(domain[0]), int(domain[1]), int(domain[2]), sitk.sitkFloat32)
        reference_image.SetSpacing(simul_spacing)
        reference_image.SetOrigin(grid_origin)
        reference_image[:, :, :] = 0


        ####################################################################
        # Make Transform matrix
        dir = tran_pos - target_pos

        n_z = np.sqrt(dir[0] * dir[0] + dir[1] * dir[1])
        if n_z == 0:
            transform_matrix = np.eye(3)
        else:
            rotate_z = l2n(((dir[0] / n_z, dir[1] / n_z, 0),
                            (-dir[1] / n_z, dir[0] / n_z, 0),
                            (0, 0, 1)))

            dir2 = np.dot(rotate_z, np.transpose(dir))
            n_y = np.sqrt(dir2[0] * dir2[0] + dir2[2] * dir2[2])
            rotate_y = l2n(((-dir2[2] / n_y, 0, dir2[0] / n_y),
                            (0, 1, 0),
                            (-dir2[0] / n_y, 0, -dir2[2] / n_y)))

            transform_matrix = np.dot(rotate_y, rotate_z)
            transform_matrix = np.linalg.inv(transform_matrix)


        move2tran = sitk.TranslationTransform(3, tran_pos)

        ####################################################################
        # Apply function
        rigid_euler = sitk.Euler3DTransform()
        rigid_euler.SetMatrix(transform_matrix.flatten())  # rotate rely on transducer axes axis
        rigid_euler.SetTranslation(move2tran.GetOffset())  # move to target
        interpolator = sitk.sitkCosineWindowedSinc

        # Resampling
        default_value = 0
        resampler = sitk.ResampleImageFilter()
        resampler.SetInterpolator(interpolator)
        resampler.SetReferenceImage(reference_image)
        resampler.SetOutputPixelType(sitk.sitkFloat32)
        resampler.SetDefaultPixelValue(default_value)
        resampler.SetTransform(rigid_euler)
        resampler.SetNumberOfThreads(10)

        if self.array.max()==1:
            crop_image = resampler.Execute(self.itkImage*2000)
            crop_image = crop_image/2000
        else:
            crop_image = resampler.Execute(self.itkImage)

        # Re-transform to origin image
        grid_origin_tran = np.dot(transform_matrix, grid_origin)
        crop_image.SetOrigin(grid_origin_tran+tran_pos)
        crop_image.SetDirection(n2l(transform_matrix.flatten()))

        # Get array
        crop_array = sitk.GetArrayFromImage(crop_image)

        return crop_array, crop_image

    def applyMaskNormalized(self, mask):
        array_mask = np.multiply(mask, self.array)
        origin_peak = np.max(array_mask)
        array_norm_full = self.array/origin_peak
        array_norm_mask = array_mask/origin_peak

        self.array_mask = array_mask
        self.array_norm_full = array_norm_full
        self.array_norm_mask = array_norm_mask
        self.peak_in_brain = origin_peak

    def applyMaskNormalized_onemore(self, mask):
        array_mask = np.multiply(mask, self.array_mask)
        origin_peak = np.max(array_mask)
        array_norm_full = self.array/origin_peak
        array_norm_mask = array_mask/origin_peak

        self.array_mask = array_mask
        self.array_norm_full = array_norm_full
        self.array_norm_mask = array_norm_mask
        self.peak_in_brain = origin_peak

    def makeVTKVolume(self, inputArray = 0):

        if inputArray == 0:
            inputArray = self.array

        vtk_pressure = ns.numpy_to_vtk(inputArray.flatten())
        image = vtk.vtkImageData()
        image.SetDimensions(l2n(self.dimension) - 1)
        image.SetExtent(self.extent)
        image.SetSpacing(self.spacing)
        image.SetOrigin(self.origin[0], self.origin[1], self.origin[2])
        image.GetPointData().SetScalars(vtk_pressure)

        volumeMapper = vtk.vtkGPUVolumeRayCastMapper()
        volumeMapper.SetInputData(image)

        opacityTransfer = vtk.vtkPiecewiseFunction()
        opacityTransfer.AddPoint(0.05, 0)
        #opacityTransfer.AddPoint(0.05, 0)
        opacityTransfer.AddPoint(0.15, 0.2)
        opacityTransfer.AddPoint(0.5, 0.5)
        opacityTransfer.AddPoint(0.8, 0.8)
        opacityTransfer.AddPoint(0.95, 1)

        ctf = vtk.vtkColorTransferFunction()
        ctf.AddRGBPoint(0.3, 0.1, 0.1, 1.0)
        ctf.AddRGBPoint(0.5, 0.2, 1.0, 0.2)
        ctf.AddRGBPoint(0.65, 1.0, 1.0, 0)
        ctf.AddRGBPoint(0.8, 1.0, 0.5, 0)
        ctf.AddRGBPoint(0.95, 1.0, 0.0, 0.0)

        volumeProperty = vtk.vtkVolumeProperty()
        volumeProperty.SetColor(ctf)
        volumeProperty.SetScalarOpacity(opacityTransfer)
        volumeProperty.SetScalarOpacityUnitDistance(10)

        volume = vtk.vtkVolume()
        volume.SetMapper(volumeMapper)
        volume.SetProperty(volumeProperty)

        return image, volume

    def makeFWHM(self, array, dist_point = False, contor_percent=0.9, set_color=[1,0.3,0]):
        vtkImage = ns.numpy_to_vtk((array.flatten())/array.flatten().max())
        image = vtk.vtkImageData()
        image.SetDimensions(l2n(self.dimension) - 1)
        image.SetExtent(self.extent)
        image.SetSpacing(self.spacing)
        image.SetOrigin(self.origin[0], self.origin[1], self.origin[2])
        #image.SetDirectionMatrix(np.flatten(self.direction))
        image.GetPointData().SetScalars(vtkImage)

        contour = vtk.vtkContourFilter()
        contour.SetInputData(image)
        contour.ComputeNormalsOn()
        contour.ComputeScalarsOn()
        contour.ComputeGradientsOn()
        contour.SetValue(0, contor_percent)
        contour.Update()

        transform = vtk.vtkTransform()
        transform.Scale(1, 1, 1)

        transform_filter = vtk.vtkTransformFilter()
        transform_filter.SetInputConnection(contour.GetOutputPort())
        transform_filter.SetTransform(transform)
        transform_filter.Update()

        FWHM_poly = transform_filter.GetOutput()

        center = vtk.vtkCenterOfMass()
        center.SetInputData(FWHM_poly)
        center.Update()

        centroid_FWHM = center.GetCenter()

        mass = vtk.vtkMassProperties()
        mass.SetInputData(FWHM_poly)
        mass.Update()

        volume_FWHM = mass.GetVolume()

        cell_locator = vtk.vtkCellLocator()
        cell_locator.SetDataSet(FWHM_poly)
        cell_locator.BuildLocator()

        cellId = vtk.reference(0)
        c = [0.0, 0.0, 0.0]
        subId = vtk.reference(0)
        dist = vtk.reference(0.0)
        if np.all(dist_point):
            cell_locator.FindClosestPoint(dist_point, c, cellId, subId, dist)
        else:
            dist = None

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(transform_filter.GetOutputPort())
        mapper.ScalarVisibilityOff()

        actor = vtk.vtkLODActor()
        actor.SetNumberOfCloudPoints(100)
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(set_color)
        actor.GetProperty().SetOpacity(0.8)

        return actor, centroid_FWHM, volume_FWHM, dist

    def DSC_FWHM(self, y_true, y_pred, brain_mask):

        y_true = np.multiply(y_true, brain_mask)
        y_pred = np.multiply(y_pred, brain_mask)

        y_true_peak = y_true.max()
        y_pred_peak = y_pred.max()
        
        y_true_peak_idx = np.squeeze(np.array(np.where(y_true == y_true_peak)))
        y_pred_peak_idx = np.squeeze(np.array(np.where(y_pred == y_pred_peak)))

        try:
            idx_dist = np.linalg.norm(y_true_peak_idx - y_pred_peak_idx)
        except:
            idx_dist = -3

        y_true_contour = np.zeros(y_true.shape)
        y_pred_contour = np.zeros(y_pred.shape)

        y_true_contour[y_true >= y_true_peak * 0.9] = 1
        y_true_contour[y_true < y_true_peak * 0.9] = 0
        y_pred_contour[y_pred >= y_pred_peak * 0.9] = 1
        y_pred_contour[y_pred < y_pred_peak * 0.9] = 0

        overlab = np.count_nonzero(np.multiply(y_pred_contour, y_true_contour))

        true_FWHM = np.count_nonzero(y_true_contour)
        pred_FWHM = np.count_nonzero(y_pred_contour)

        DSC = (overlab * 2 )/ (true_FWHM + pred_FWHM)
        # loss = 1 - DSC

        return DSC, idx_dist ,y_true_peak_idx, y_pred_peak_idx

    def segmentationMask(self, dilate, threshold = 10):
        padding_level = 20
        ret, img_th = cv2.threshold(self.array, threshold, 1, 0)

        # remain largest region
        label_im, nb_labels = ndimage.label(img_th)
        sizes = ndimage.sum(img_th, label_im, range(nb_labels + 1))
        mask = sizes == np.max(sizes)
        label_img = mask[label_im]


        if threshold == -100:
            kernel = np.ones((3, 3), np.uint8)
            open_volume = cv2.morphologyEx(np.array(label_img).astype(np.uint8), cv2.MORPH_OPEN, kernel)

            label_im, nb_labels = ndimage.label(open_volume)
            sizes = ndimage.sum(img_th, label_im, range(nb_labels + 1))
            mask = sizes == np.max(sizes)
            label_img = mask[label_im]
            open_volume = np.array(label_img).astype(np.uint8)

            closed_volume = cv2.morphologyEx(open_volume, cv2.MORPH_CLOSE, kernel)
            # closed_volume = ndimage.binary_erosion(closed_volume)
            #         temp =  closed_volume
            temp = np.zeros_like(label_img)

            for idx in range(closed_volume.shape[0]):
                closed_slice = closed_volume[idx, :, :]
                mask_slice = ndimage.morphology.binary_fill_holes(closed_slice)
                temp[idx, :, :] = mask_slice
            #   mpimg.imsave(os.path.join('E:/2020/KHM_synt/seg', str(threshold)+'_'+str(idx)+'.png'), mask_slice, vmin=-1, vmax=1, cmap='gray')

        if threshold >= 1:
            kernel = np.ones((3, 3), np.uint8)
            open_volume = cv2.morphologyEx(np.array(label_img).astype(np.uint8), cv2.MORPH_OPEN, kernel)

            label_im, nb_labels = ndimage.label(open_volume)
            sizes = ndimage.sum(img_th, label_im, range(nb_labels + 1))
            mask = sizes == np.max(sizes)
            label_img = mask[label_im]
            open_volume = np.array(label_img).astype(np.uint8)

            closed_volume = cv2.morphologyEx(open_volume, cv2.MORPH_CLOSE, kernel)
            temp = closed_volume
            skull_volume = np.array(temp)

        if dilate == True:
            kernel = np.ones((2,2), np.uint8)
            skull_volume = cv2.dilate(np.array(skull_volume).astype(np.uint8),kernel, iterations=10)
        else:
            skull_volume = np.multiply(skull_volume, self.array)

        #extract brain region only
        mask = np.copy(skull_volume)
        
        mask[:padding_level, :, :] = 1
        mask[-padding_level:, :, :] = 1
        mask[:, :padding_level, :] = 1
        mask[:, -padding_level:, :] = 1
        mask[:, :, :padding_level] = 1
        mask[:, :, -padding_level:] = 1
        mask = mask - 1
        mask = -mask
        
        brain_mask = np.zeros_like(mask)
        
        for i in range(brain_mask.shape[1]):
            ret, makers = cv2.connectedComponents(mask[:, i, :])
            brain_mask[:, i, :] = makers
        
        brain_mask[brain_mask <= 1] = 0
        brain_mask[brain_mask > 0] = 1
        #brain_mask = 0
        return skull_volume, brain_mask

#     def compare2simulation(self, realSimul, synSimul):
        
#         ######################################################
#         ## Read FUll image 
#         ######################################################

#         real = realCook.array

#         synCook = niiCook()
#         synCook.readSavedFile(os.path.join(data_path, 'Syn_full.nii'))
#         syn = synCook.array


#         ######################################################
#         ## Read mask image 
#         ######################################################
#         realCTCook = niiCook()
#         realCTCook.readSavedFile(os.path.join(data_path, 'Real_crop_image.nii'))
#         real_ct = realCTCook.array

#         synCTCook = niiCook()
#         synCTCook.readSavedFile(os.path.join(data_path, 'Syn_crop_image.nii'))
#         syn_ct = synCTCook.array

#         realMaskCook = niiCook()
#         realMaskCook.readSavedFile(os.path.join(data_path, 'Real_mask.nii'))
#         real_mask = realMaskCook.array

#         synMaskCook = niiCook()
#         synMaskCook.readSavedFile(os.path.join(data_path, 'Syn_mask.nii'))
#         syn_mask = synMaskCook.array

#         ######################################################
#         ## Normalize  
#         ######################################################
#         maxR = np.max(real_mask)
#         real_m  = real/maxR
#         syn_m = syn/maxR


    def makeITK(self, array, path=False):

        result_itk = sitk.GetImageFromArray(array)
        result_itk.SetSpacing(self.spacing)
        result_itk.SetOrigin(self.origin)
        result_itk.SetDirection(self.direction)
        #result_itk.SetDirection(self.direction)

        ## set RAS
        #direction = [-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0]
        #result_itk.SetDirection(direction)

        if path:
            writer = sitk.ImageFileWriter()
            writer.SetFileName(path)
            writer.Execute(result_itk)
        return result_itk

    
    def makeITK_default(self, array, path=False):

        result_itk = sitk.GetImageFromArray(array)
        result_itk.SetSpacing(self.spacing)
        result_itk.SetOrigin(self.origin)
        #result_itk.SetDirection(self.direction)

        ## set RAS
        #direction = [-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0]
        #result_itk.SetDirection(direction)

        if path:
            writer = sitk.ImageFileWriter()
            writer.SetFileName(path)
            writer.Execute(result_itk)
        return result_itk
    
    
    
    def saveITK(self, path=False):

        writer = sitk.ImageFileWriter()
        writer.SetFileName(path)
        writer.Execute(self.itkImage)

    def makeNiiCook(self, numpy_array):

        result_itk = self.makeITK(numpy_array)
        makeCook = niiCook()
        makeCook.readITK(result_itk)

        return makeCook

    def read_1D(self, start, end):
        
        start = l2n(start)
        end = l2n(end)
        
        dir = (end - start)
        unit_dir = dir/np.linalg.norm(dir)

        sorting_len = np.linalg.norm(dir)
        sample = np.linspace(0, sorting_len, num= np.int(sorting_len*7))

        dist = np.zeros((len(sample), 3))
        dist[:, 0] = sample
        dist[:, 1] = sample
        dist[:, 2] = sample

        vectors = np.zeros((len(sample), 3))
        vectors[:, 0] = unit_dir[0]
        vectors[:, 1] = unit_dir[1]
        vectors[:, 2] = unit_dir[2]

        points = start+np.multiply(vectors, dist)
        axial_result = np.zeros(len(points))

        for i in range(len(points)):
            idx = self.itkImage.TransformPhysicalPointToIndex(points[i,:])
            if idx[0]>=self.dimension[0] or idx[1]>=self.dimension[1] or idx[2]>=self.dimension[2] :
                axial_result[i] = 0
            else:
                axial_result[i] = self.itkImage[idx]

        return axial_result, sample, points

    def mask2000(self, path):
        itkImage = self.itkImage
        itkImage = itkImage*2000

        writer = sitk.ImageFileWriter()
        writer.SetFileName(path)
        writer.Execute(itkImage)

        return itkImage

    def figure_2D(self, savename, data, vmin, vmax):
        shape = data.shape
        max_idx = [108,108,108]#np.where(data == np.max(data))
        fig = plt.figure(figsize=(10, 10))
        gs = fig.add_gridspec(1, 1)
        ax0 = fig.add_subplot(gs[0])
        ax0.imshow(np.fliplr(np.flip(np.squeeze(data[:,max_idx[1], :]))), cmap='jet', interpolation='hanning', vmin=vmin, vmax=vmax)
        ax0.scatter(max_idx[2], max_idx[0], c='red', marker='x')
        fig.savefig(savename, transparent=True)

        return max_idx

    def figure_2D_compare(savename, real, syn):
        shape = real.shape
        real_max_idx = np.where(real == np.max(real))
        syn_max_idx = np.where(syn == np.max(syn))

        fig = plt.figure(figsize=(50, 50))
        gs = fig.add_gridspec(2, 2)
        ax0 = fig.add_subplot(gs[0])
        ax0.imshow(np.rot90(real[:, real_max_idx[1][0], :]), cmap='jet', interpolation='hanning', vmin=0, vmax=np.max(real)+1)
        ax0.scatter(real_max_idx[0][0], shape[0] - real_max_idx[2][0], c='white', marker='x', s=1000)
        ax0.axis('off')

        ax1 = fig.add_subplot(gs[1])
        ax1.imshow(np.rot90(syn[:, syn_max_idx[1][0], :]), cmap='jet', interpolation='hanning', vmin=0, vmax=np.max(real)+1)
        ax1.scatter(syn_max_idx[0][0], shape[0] - syn_max_idx[2][0], c='white', marker='x', s=1000)
        ax1.axis('off')

        ax2 = fig.add_subplot(gs[2])
        ax2.imshow(np.rot90(real[:, :, real_max_idx[2][0]]), cmap='jet', interpolation='hanning', vmin=0, vmax=np.max(real)+1)
        ax2.scatter(real_max_idx[0][0], shape[0] - real_max_idx[2][0], c='white', marker='x', s=1000)
        ax2.axis('off')

        ax3 = fig.add_subplot(gs[3])
        ax3.imshow(np.rot90(syn[:, :, syn_max_idx[2][0]]), cmap='jet', interpolation='hanning', vmin=0, vmax=np.max(real)+1)
        ax3.scatter(syn_max_idx[0][0], shape[0] - syn_max_idx[1][0], c='white', marker='x', s=1000)
        ax3.axis('off')

        fig.savefig(savename, transparent=True)

        return real_max_idx, syn_max_idx

    def figure_target_2D(savename, data, color_type, vmin, vmax):
        shape = data.shape
        fig = plt.figure(figsize=(10, 10), facecolor='white')

        plt.axis('off')
        plt.imshow(np.fliplr(np.flip(np.squeeze(data[:, :, np.int(shape[1] / 2)]))), cmap=color_type,
                   interpolation='hanning', vmin=vmin, vmax=vmax)
        # plt.colorbar()
        plt.scatter(np.int(shape[1] / 2), np.int(shape[1] / 2), c='white', marker='x', s=400)
        fig.savefig(savename + '.png', transparent=True)

    def figure_skull(self, savename, data, color_type, vmin, vmax):
        shape = data.shape
        # fig = plt.figure(figsize=(10,10), facecolor = 'white')
        #
        # plt.axis('off')
        # plt.imshow((np.squeeze(data[:,np.int(shape[1]/2),:])), cmap= color_type, interpolation='quadric', vmin=vmin, vmax=vmax, origin='lower')
        # #plt.colorbar()
        # #plt.scatter(np.int(shape[1]/2), np.int(shape[1]/2), c='white', marker='x', s= 400)
        # fig.savefig(savename+'.png', transparent=True)

        fig = plt.figure(figsize=(10,10), facecolor = 'white')

        #plt.axis('off')
        plt.imshow((np.squeeze(data[:,:,np.int(shape[2]/2)])), cmap= color_type, interpolation='hanning', vmin=vmin, vmax=vmax, origin='lower')
        #plt.colorbar()
        #plt.scatter(np.int(shape[1]/2), np.int(shape[1]/2), c='white', marker='x', s= 400)
        fig.savefig(savename+'.png', transparent=True)

        # fig = plt.figure(figsize=(10,10), facecolor = 'white')
        #
        # #plt.axis('off')
        # plt.imshow((np.squeeze(data[np.int(shape[1]/2),:,:])), cmap= color_type, interpolation='hanning', vmin=vmin, vmax=vmax, origin='lower')
        # #plt.colorbar()
        # plt.scatter(np.int(shape[1]/2), np.int(shape[1]/2), c='white', marker='x', s= 400)
        # fig.savefig(savename+'.png', transparent=True)

    def figure(self,  color_type, savename= 'test'):

        data = self.array
        shape = data.shape
        fig = plt.figure(figsize=(10, 10), facecolor='white')

        vmax = data.max()
        vmin = data.min()

        # plt.axis('off')
        plt.imshow((np.squeeze(data[:, np.int(shape[1] / 2), :])), cmap=color_type, interpolation='hanning', vmin=vmin,
                   vmax=vmax, origin='lower')
        # plt.colorbar()
        plt.scatter(np.int(shape[0] / 2), np.int(shape[2] / 2), c='white', marker='x', s=400)
        fig.savefig(savename + '.png', transparent=True)

        fig = plt.figure(figsize=(10, 10), facecolor='white')

        # plt.axis('off')
        plt.imshow((np.squeeze(data[:, :, np.int(shape[2] / 2)])), cmap=color_type, interpolation='hanning', vmin=vmin,
                   vmax=vmax, origin='lower')
        # plt.colorbar()
        plt.scatter(np.int(shape[0] / 2), np.int(shape[1] / 2), c='white', marker='x', s=400)
        fig.savefig(savename + '.png', transparent=True)

        fig = plt.figure(figsize=(10, 10), facecolor='white')

        # plt.axis('off')
        plt.imshow((np.squeeze(data[np.int(shape[0] / 2), :, :])), cmap=color_type, interpolation='hanning', vmin=vmin,
                   vmax=vmax, origin='lower')
        # plt.colorbar()
        plt.scatter(np.int(shape[1] / 2), np.int(shape[2] / 2), c='white', marker='x', s=400)
        fig.savefig(savename + '.png', transparent=True)