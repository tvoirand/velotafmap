"""
Utils module for velotafmap.
"""

# standard imports
import os
import datetime

# third party imports
import xmltodict
import numpy as np
import pandas as pd
import xarray as xr
from scipy import signal
from scipy import ndimage
from numpy.linalg import norm
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import matplotlib.pyplot as plt
import cv2


def check_bike_commuting(input_file):
    """
    Check if a Strava activity input file corresponds to bike commuting activity.
    Input:
        -input_file         str
    Output:
        -                   bool
    """

    # parse input file
    with open(input_file, "r") as infile:
        data = xmltodict.parse(infile.read())

    # check activity type (cycling=1? no documentation found)
    if data["gpx"]["trk"]["type"] not in ["1"]:
        return False

    # read coordinates of first and last points
    first_point = data["gpx"]["trk"]["trkseg"]["trkpt"][0]
    last_point = data["gpx"]["trk"]["trkseg"]["trkpt"][-1]
    longitudes = [float(first_point["@lon"]), float(last_point["@lon"])]
    latitudes = [float(first_point["@lat"]), float(last_point["@lat"])]

    # commuting track coordinates bounds
    XMIN = -0.585652
    XMAX = -0.486084
    YMIN = 44.777028
    YMAX = 44.860108

    # check first and last points are within bounds
    if (
        any(lon < XMIN for lon in longitudes)
        or any(lon > XMAX for lon in longitudes)
        or any(lat < YMIN for lat in latitudes)
        or any(lat > YMAX for lat in latitudes)
    ):
        return False

    return True


def read_gpx(input_file):
    """
    Read GPX files.
    Input:
        -input_file     str
    Output:
        -activity_date  datetime.datetime object
        -activity_name  str
        -xmin           float
        -xmax           float
        -ymin           float
        -ymax           float
        -points_df      pandas dataframe
            index: timestamp as datetime object
            columns:
                "lon" (float) (longitude)
                "lat" (float) (latitude)
                "ele" (float) (elevation)
                "time" (float) (time elapsed since activity start in seconds)
                "vel"   (float)     (velocity)
                "x" (float)         (projected x coord)
                "y" (float)         (projected y coord)
    """

    # parse input file
    with open(input_file, "r") as infile:
        data = xmltodict.parse(infile.read())

    # read activity date
    activity_date = datetime.datetime.strptime(
        data["gpx"]["metadata"]["time"], "%Y-%m-%dT%H:%M:%SZ"
    )

    # read activity name
    activity_name = data["gpx"]["trk"]["name"]

    # read points
    points = data["gpx"]["trk"]["trkseg"]["trkpt"]

    # read coordinates bounds
    xmin = 180
    xmax = -180
    ymin = 90
    ymax = -90
    for item in points:

        # convert longitude, latitude, and elevation to float
        item["@lon"] = float(item["@lon"])
        item["@lat"] = float(item["@lat"])
        item["ele"] = float(item["ele"])

        # convert time to datetime object
        item["time"] = datetime.datetime.strptime(item["time"], "%Y-%m-%dT%H:%M:%SZ")

        if item["@lon"] < xmin:
            xmin = item["@lon"]
        if item["@lon"] > xmax:
            xmax = item["@lon"]
        if item["@lat"] < ymin:
            ymin = item["@lat"]
        if item["@lat"] > ymax:
            ymax = item["@lat"]

    # get time elapsed in seconds for each point
    time_elapsed = [(point["time"] - points[0]["time"]).seconds for point in points]

    # store longitude, latitude, and elevation in a numpy array
    array = np.asarray(
        [
            [point["@lon"], point["@lat"], point["ele"], time_elapsed[i]]
            for i, point in enumerate(points)
        ]
    )

    # store points in pandas data frame, with datetime index
    points_df = pd.DataFrame(
        array,
        index=[point["time"] for point in points],
        columns=["lon", "lat", "ele", "time"],
    )

    # get coordinates of points in meters (projected in EPSG 32630)
    projected_coords = ccrs.epsg(32630).transform_points(
        ccrs.PlateCarree(), np.asarray(points_df["lon"]), np.asarray(points_df["lat"])
    )

    # add velocity (in km/h) to points dataframe
    gradient = np.gradient(  # compute gradient
        projected_coords[:, :2], points_df["time"], axis=0
    )
    gradient *= 3.6  # convert form m/s to km/h
    points_df["vel"] = np.array([norm(v) for v in gradient])  # add to dataframe

    # filter velocity
    b, a = signal.butter(3, 0.01)  # get Butterworth filter coefficients
    points_df["vel"] = signal.filtfilt(
        b, a, points_df["vel"]
    )  # apply forward and backward filter

    # add projected coordinates to points dataframe
    points_df["x"] = projected_coords[:, 0]
    points_df["y"] = projected_coords[:, 1]

    return activity_date, activity_name, xmin, xmax, ymin, ymax, points_df


def create_map(array, output_file, projection):
    """
    Create map with cartopy based on input xarray DataArray.
    The DataArray must have "x" and a "y" dimensions.
    Input:
        -array          xarray DataArray instance
        -output_file    str
        -projection     cartopy.crs.Projection instance
    """

    # create figure
    fig = plt.figure(figsize=(16, 12), dpi=300)

    # create geo axes
    geo_axes = plt.subplot(projection=projection)

    # add open street map background
    osm_background = cimgt.OSM()
    geo_axes.add_image(osm_background, 13)

    # plot dataset
    xr.plot.imshow(
        darray=array,
        x="x",
        y="y",
        ax=geo_axes,
        transform=projection,
        zorder=10,
        vmin=20,
        vmax=28,
        extend="neither",
    )

    # save as image
    plt.savefig(output_file)

    # close figure
    plt.close()


def create_video(video_name, input_dir, fps=16):
    """
    Create video with opencv based on png images stored in input dir.
    Input:
        -video_name     str
        -input_dir      str
        -fps            int
    """

    # set FourCC video codec code
    fourcc = cv2.VideoWriter_fourcc(*"XVID")

    # get input images list
    images = [img for img in os.listdir(input_dir) if img.endswith(".png")]

    # read frame shape
    frame = cv2.imread(os.path.join(input_dir, images[0]))
    height, width, layers = frame.shape

    # create video writer object
    video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

    # write video using input images
    for image in images:
        video.write(cv2.imread(os.path.join(input_dir, image)))

    # close video writer and windows once finished
    cv2.destroyAllWindows()
    video.release()


def nan_filter_1d(array, sigma=1.0, axis=0):
    """
    Apply 1d gaussian filter to 3d array containing nans.
    Based on the following stackoverflow answer by user David:
    https://stackoverflow.com/questions/18697532/gaussian-filtering-a-image-with-nan-in-python
    Input:
        -array              np.array
        -sigma              float
        -axis               int
    Output:
        -                   np.array
    """

    # create copy of input array with nans replaced by zeros
    V = array.copy()
    V[np.isnan(array)] = 0

    # create arrays of ones, with zeros indicating positions of nans
    W = np.ones(array.shape)
    W[np.isnan(array)] = 0

    # apply 1d filter
    VV = ndimage.gaussian_filter1d(V, sigma=sigma, axis=axis)
    WW = ndimage.gaussian_filter1d(W, sigma=sigma, axis=axis)

    # returned combined two filtered arrays
    return VV / WW


def nan_filter(array, sigma=1.0):
    """
    Apply gaussian filter to array containing nans.
    Based on the following stackoverflow answer by user David:
    https://stackoverflow.com/questions/18697532/gaussian-filtering-a-image-with-nan-in-python
    Input:
        -array              np.array
        -sigma              float
        -axis               int
    Output:
        -                   np.array
    """

    # create copy of input array with nans replaced by zeros
    V = array.copy()
    V[np.isnan(array)] = 0

    # create arrays of ones, with zeros indicating positions of nans
    W = np.ones(array.shape)
    W[np.isnan(array)] = 0

    # apply 1d filter
    VV = ndimage.gaussian_filter(V, sigma=sigma)
    WW = ndimage.gaussian_filter(W, sigma=sigma)

    # returned combined two filtered arrays
    return VV / WW
