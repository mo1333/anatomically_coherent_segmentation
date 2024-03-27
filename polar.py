# -*- coding: utf-8 -*-
"""
Polar transformation of an image and inverse transformation

@author: wvogl
"""

import numpy as np
import scipy as sp
import scipy.ndimage

import matplotlib.pyplot as plt

def _main():
    import nibabel as nib
    
    nii = nib.load('/home/wvogl/Dokumente/Uni/optima/data/RVO_Large/509/20121207/warp/thicknessToReferenceAffine.nii.gz')
    #im = Image.open('mri_demo.png')
    im = nii.get_data()
    #im = im.convert('RGB')
    plt.subplot(131)
    plt.imshow(im,vmin=70,vmax=400)
    plt.subplot(132)
    #plot_polar_image(data, origin=None,shape=(80,80),rmax=150)
    proj,_,_ = reproject_image_into_polar(im)
    plt.imshow(proj,vmin=70,vmax=400)
    
    plt.show()

def plot_directional_intensity(data, origin=None):
    """Makes a cicular histogram showing average intensity binned by direction
    from "origin" for each band in "data" (a 3D numpy array). "origin" defaults
    to the center of the image."""
    def intensity_rose(theta, band, color):
        theta, band = theta.flatten(), band.flatten()
        intensities, theta_bins = _bin_by(band, theta)
        mean_intensity = map(np.sum, intensities)
        width = np.diff(theta_bins)[0]
        plt.bar(theta_bins, mean_intensity, width=width, color=color)
        plt.xlabel(color + ' Band')
        plt.yticks([])

    # Make cartesian coordinates for the pixel indicies
    # (The origin defaults to the center of the image)
    x, y = _index_coords(data, origin)

    # Convert the pixel indices into polar coords.
    r, theta = cart2pol(x, y)

    # Plot...
    plt.figure()
    plt.axes([0.1,0.1,0.9,0.9], polar=True)
    #plt.subplot(2,2,1, projection='polar')
    
    intensity_rose(theta, data.T, 'red')

    plt.suptitle('Average intensity as a function of direction')

def plot_polar_image(data, origin=None, shape=None,rmax=None):
    """Plots an image reprojected into polar coordinages with the origin
    at "origin" (a tuple of (x0, y0), defaults to the center of the image)"""
    polar_grid, r, theta = reproject_image_into_polar(data, origin, shape,rmax)
    #plt.figure()
    plt.imshow(polar_grid, extent=(theta.min(), theta.max(), r.max(), r.min()))
    plt.axis('tight')    
    
    plt.ylim(plt.ylim()[::-1])
    plt.xlabel('Theta Coordinate (radians)')
    plt.ylabel('R Coordinate (pixels)')
    plt.title('Image in Polar Coordinates')

def _index_coords(data, origin=None):
    """Creates x & y coords for the indicies in a numpy array "data".
    "origin" defaults to the center of the image. Specify origin=(0,0)
    to set the origin to the lower left corner of the image."""
    ny, nx = data.shape[:2]
    if origin is None:
        origin_x, origin_y = nx // 2, ny // 2
    else:
        origin_x, origin_y = origin
    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    x -= origin_x
    y -= origin_y
    return x, y

def cart2pol(x,y):
    """
    Transform Cartesian to polar coordinates.
    cart2pol transforms corresponding elements of data
    stored in Cartesian coordinates X,Y to polar coordinates (angle TH
    and radius R).  The arrays X and Y must be the same size (or
    either can be scalar). TH is returned in radians. 
    """    
    
    th =  np.arctan2(y,x)
    r = np.hypot(x,y)
    return (r,th)

def pol2cart(r,th):
    """    
    Transform polar to Cartesian coordinates.
    pol2cart transforms corresponding elements of data
    stored in polar coordinates (angle TH, radius R) to Cartesian
    coordinates X,Y.  The arrays TH and R must the same size (or
    either can be scalar).  TH must be in radians.
    """
    x = r * np.cos(th)
    y = r * np.sin(th)
    return (x,y)

def _bin_by(x, y, nbins=30):
    """Bin x by y, given paired observations of x & y.
    Returns the binned "x" values and the left edges of the bins."""
    bins = np.linspace(y.min(), y.max(), nbins+1)
    # To avoid extra bin for the max value
    bins[-1] += 1 

    indicies = np.digitize(y, bins)

    output = []
    for i in np.arange(1, len(bins)):
        output.append(x[indicies==i])

    # Just return the left edges of the bins
    bins = bins[:-1]

    return output, bins

def reproject_image_into_polar(data, origin=None,shape=None,rmax=None):
    """Reprojects a 2D numpy array ("data") into a polar coordinate system.
    "origin" is a tuple of (x0, y0) and defaults to the center of the image.
    "shape" is a tuple of (ny,nx), where nx is the resolution in x and defines
    the stepsize for radius, and ny is the resolution in y and defines the 
    angular stepsize. If none, shape is equal to data
    """
    if (shape is None):
        ny, nx = data.shape[:2]
    else:
        ny, nx = shape
    if origin is None:
        
        origin = (data.shape[1]//2, data.shape[0]//2)

    # Determine that the min and max r and theta coords will be...
    x, y = _index_coords(data, origin=origin)
    r, theta = cart2pol(x, y)
    if (rmax==None):
        rmax=r.max()
    # Make a regular (in polar space) grid based on the min and max r & theta
    r_i = np.linspace(r.min(), rmax, nx)
    theta_i = np.linspace(theta.min(), theta.max(), ny)
    theta_grid, r_grid = np.meshgrid(theta_i, r_i)

    # Project the r and theta grid back into pixel coordinates
    xi, yi = pol2cart(r_grid, theta_grid)
    xi += origin[0] # We need to shift the origin back to 
    yi += origin[1] # back to the lower-left corner...
    xi, yi = xi.flatten(), yi.flatten()
    coords = np.vstack((xi, yi)) # (map_coordinates requires a 2xn array)

    # Reproject each band individually and the restack
    # (uses less memory than reprojection the 3-dimensional array in one step)

    zi = sp.ndimage.map_coordinates(data.T, coords, order=1)
    output = zi.reshape((nx, ny))

    return output, r_i, theta_i

if __name__ == '__main__':
    _main()
