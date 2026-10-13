from pyfame.utils.constants import *
from pyfame.landmark import facial_landmarks
from math import atan
import numpy as np
import cv2 as cv

def get_variable_name(variable, namespace) -> str:
    """ Takes in a variable and a namespace (one of locals() or globals()), and returns the variables defined name in the 
        relevant scope.

        Parameters
        ----------

        variable: any
            Any defined variable.
        
        namespace: function
            One of locals() or globals(), defines which scope for the function to look at when searching for the variables name.
        
        Returns
        -------
        variable_name: str
            The assigned variable name at the given scope level (if any).
    """

    return [name for name, value in namespace.items() if value == variable][0]

def get_landmark_names(landmarks, module_globals = vars(facial_landmarks)):
    """ Takes in a list of landmark paths, and returns its internal variable name. 
    """
    # check for sublists
    if isinstance(landmarks[0], list):
        names = []
        for i,lm in enumerate(landmarks, start=1):
            for name, val in module_globals.items():
                if isinstance(val, list) and val == lm:
                    names.append(name)
            
            if len(names) < i:
                names.append("UNKNOWN_LANDMARK")
        
        return names
    
    else:
        for name, val in module_globals.items():
            if isinstance(val, list) and val == landmarks:
                return name
        
        return "UNKNOWN_LANDMARK"

def compute_rotation_angle(slope_1:float, slope_2:float = 0.0) -> float:
    """ Given the current and previous slope, this function uses the arctan function to compute the angle delta 
    in radians. If no previous slope is provided, the default value is zero.

    Parameters
    ----------

    slope_1: float
        The current slope.

    slope_2: float
        The initial slope (defaults to 0.0)

    returns
    -------

    rot_angle: float
        The displacement between the two slopes. 
    """

    if np.isinf(slope_1) and np.isinf(slope_2):
        return 0.0
    elif np.isinf(slope_1):
        return 90.0
    elif np.isinf(slope_2):
        return -90.0

    rad_angle = atan((slope_2-slope_1) / (1 + slope_1*slope_2))
    rot_angle = np.degrees(rad_angle)
    return rot_angle

def compute_slope(p1:tuple[int,int], p2:tuple[int,int]) -> float:
    """ Given two integer points in the form (x,y), compute the slope of the line 
    that connects the two points and return it.
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]

    if dx == 0:
        return float("inf")
    return dy/dx

def display_landmarks_face_overlay(frame, landmarker_coordinates, point_radius:int = 4):
    """ Given a frame or image containing a face, and the FaceLandmarker landmark coordinates list
    returned from applying the FaceLandmarker to the frame or image, visualize all landmark points on the 
    frame or image.

    Parameters
    ----------

    frame : ndarray
        The static image or video frame to have the
        FaceLandmarker points overlayed on.
    
    landmarker_coordinates : list[tuple[int,int]]
        The pixel coordinates of the facial landmarks returned 
        by `get_landmarker_coordinates()`.
    
    point_radius : int
        The radius of the overlayed points. 
    
    Returns
    -------
    None
    """
    for (x, y) in landmarker_coordinates:
        cv.circle(frame, (x,y), point_radius, (0, 0, 255), -1)

def _anchor_to_pixel(anchor, face_center, face_width, face_height):
    """
    Convert a normalised ``FaceAnchor`` position to absolute pixel
    coordinates in the frame.

    The anchor's normalised (x, y) value is scaled by face width and
    face height respectively and offset from the face center to produce
    the final pixel position.

    Parameters
    ----------
    anchor : FaceAnchor
        The facial anchor whose normalised position is to be converted.
    face_center : tuple of int
        The (x, y) pixel coordinates of the face center, computed from
        the bounding box of the face oval landmark.
    face_width : int
        The pixel width of the face oval bounding box.
    face_height : int
        The pixel height of the face oval bounding box.

    Returns
    -------
    px : int
        The absolute x pixel coordinate of the anchor position.
    py : int
        The absolute y pixel coordinate of the anchor position.
    """
    ax, ay = anchor.value
    px = int(face_center[0] + ax * face_width)
    py = int(face_center[1] + ay * face_height)
    return px, py