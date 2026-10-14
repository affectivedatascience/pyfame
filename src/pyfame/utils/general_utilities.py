from .constants import *
from .checks import *
from .face_anchors import FaceAnchor
from pyfame.landmark import facial_landmarks
from math import atan
import numpy as np
import cv2 as cv

def get_variable_name(variable, namespace: dict) -> str:
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
    if not isinstance(namespace, dict):
        raise TypeError("Parameter namespace must be a dictionary returned by one of locals() or globals().")
    
    return [name for name, value in namespace.items() if value is variable][0]

def get_landmark_names(landmarks: list[tuple[int,int]] | list[list[tuple[int,int]]]):
    """ Takes in a list of landmark paths, and returns its internal variable name. 
    """

    if not isinstance(landmarks, list):
        raise TypeError("Parameter landmarks must be a list or nested list of tuples.")

    if not landmarks:
        raise ValueError("Parameter landmarks must be a nonempty list.")

    if isinstance(landmarks[0], list):
        if not landmarks[0]:
            raise ValueError("Nested landmark paths must be nonempty.")

        check_type(landmarks[0][0], [int, np.integer], iterable=True)
    else:
        check_type(landmarks[0], [int, np.integer], iterable=True)

    module_globals = vars(facial_landmarks)
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
    check_type(slope_1, [float])
    check_type(slope_2, [float])

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
    check_type(p1, [int], iterable=True)
    check_type(p2, [int], iterable=True)

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
    check_type(frame, [np.ndarray, cv.typing.MatLike])
    check_type(point_radius, [int])

    for (x, y) in landmarker_coordinates:
        cv.circle(frame, (x,y), point_radius, (0, 0, 255), -1)

def _anchor_to_pixel(anchor: FaceAnchor, face_center: tuple[int,int], face_width: int, face_height: int):
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
    check_type(anchor, [FaceAnchor])
    check_type(face_center, [int], iterable=True)
    check_value()

    ax, ay = anchor.value
    px = int(face_center[0] + ax * face_width)
    py = int(face_center[1] + ay * face_height)
    return px, py