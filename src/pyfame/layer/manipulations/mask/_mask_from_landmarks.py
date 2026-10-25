from pyfame.utils.constants import *
from pyfame.landmark.get_landmark_coordinates import get_relative_landmark_coordinates
from pyfame.landmark.facial_landmarks import *
from pyfame.utils.exceptions import *
import cv2 as cv
import numpy as np

def mask_from_landmarks(frame:cv.typing.MatLike, landmark_paths:list[list[tuple[int,int]]] | list[tuple[int,int]], landmarker_coordinates) -> cv.typing.MatLike:
    """ Given a landmark path, create a binary image mask based
    on the input image or frame.

    Parameters
    ----------
    frame : MatLike
        A static image or decoded video frame used to create the binary image mask.
    landmark_paths : array_like
        A predefined single landmark path, or a list of landmark paths specifying
        the masked region in the frame.
    landmarker_coordinates : array_like
        A list of screen pixel coordinates returned by the mediapipe 
        FaceLandmarker task.

    Returns
    -------
    mask : ndarray
        A binary image with everything masked out except for the regions
        enclosed by landmark_paths.
    """
    masked_frame = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

    # Extracting all of the relevant landmark pixel coordinates
    lc_screen_coords = get_relative_landmark_coordinates(landmarker_coordinates, LANDMARK_LEFT_CHEEK)
    rc_screen_coords = get_relative_landmark_coordinates(landmarker_coordinates, LANDMARK_RIGHT_CHEEK)
    chin_screen_coords = get_relative_landmark_coordinates(landmarker_coordinates, LANDMARK_CHIN)
    nose_screen_coords = get_relative_landmark_coordinates(landmarker_coordinates, LANDMARK_NOSE)
    ler_screen_coords = get_relative_landmark_coordinates(landmarker_coordinates, LANDMARK_LEFT_EYE_REGION)
    rer_screen_coords = get_relative_landmark_coordinates(landmarker_coordinates, LANDMARK_RIGHT_EYE_REGION)
    le_screen_coords = get_relative_landmark_coordinates(landmarker_coordinates, LANDMARK_LEFT_EYE)
    re_screen_coords = get_relative_landmark_coordinates(landmarker_coordinates, LANDMARK_RIGHT_EYE)
    li_screen_coords = get_relative_landmark_coordinates(landmarker_coordinates, LANDMARK_LEFT_IRIS)
    ri_screen_coords = get_relative_landmark_coordinates(landmarker_coordinates, LANDMARK_RIGHT_IRIS)
    leb_screen_coords = get_relative_landmark_coordinates(landmarker_coordinates, LANDMARK_LEFT_EYEBROW)
    reb_screen_coords = get_relative_landmark_coordinates(landmarker_coordinates, LANDMARK_RIGHT_EYEBROW)
    lip_o_screen_coords = get_relative_landmark_coordinates(landmarker_coordinates, LANDMARK_LIPS_OUTER_CONTOUR)
    lip_i_screen_coords = get_relative_landmark_coordinates(landmarker_coordinates, LANDMARK_LIPS_INNER_CONTOUR)
    fo_screen_coords = get_relative_landmark_coordinates(landmarker_coordinates, LANDMARK_FACE_OVAL)

    # Convert all screen coords to ndarrays
    l_cheek_array = np.array(lc_screen_coords, dtype=np.int32)
    r_cheek_array = np.array(rc_screen_coords, dtype=np.int32)
    chin_array = np.array(chin_screen_coords, dtype=np.int32)
    nose_array = np.array(nose_screen_coords, dtype=np.int32)
    l_eye_reg_array = np.array(ler_screen_coords, dtype=np.int32)
    r_eye_reg_array = np.array(rer_screen_coords, dtype=np.int32)
    l_eye_array = np.array(le_screen_coords, dtype=np.int32)
    r_eye_array = np.array(re_screen_coords, dtype=np.int32)
    l_iris_array = np.array(li_screen_coords, dtype=np.int32)
    r_iris_array = np.array(ri_screen_coords, dtype=np.int32)
    l_eyebrow_array = np.array(leb_screen_coords, dtype=np.int32)
    r_eyebrow_array = np.array(reb_screen_coords, dtype=np.int32)
    lip_outer_array = np.array(lip_o_screen_coords, dtype=np.int32)
    lip_inner_array = np.array(lip_i_screen_coords, dtype=np.int32)
    face_oval_array = np.array(fo_screen_coords, dtype=np.int32)

    if isinstance(landmark_paths[0], list):
        for path in landmark_paths:
            match path:
                # Both Cheeks
                case [(0,)]:
                    lc_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                    lc_mask = cv.fillPoly(img=lc_mask, pts=[l_cheek_array], color=(255,255,255))
                    lc_mask = lc_mask.astype(bool)

                    rc_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                    rc_mask = cv.fillPoly(img=rc_mask, pts=[r_cheek_array], color=(255,255,255))
                    rc_mask = rc_mask.astype(bool)

                    masked_frame[lc_mask] = 255
                    masked_frame[rc_mask] = 255
                
                # Left Cheek Only
                case [(1,)]:
                    lc_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                    lc_mask = cv.fillPoly(img=lc_mask, pts=[l_cheek_array], color=(255,255,255))
                    lc_mask = lc_mask.astype(bool)

                    masked_frame[lc_mask] = 255
                
                # Right Cheek Only
                case [(2,)]:
                    rc_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                    rc_mask = cv.fillPoly(img=rc_mask, pts=[r_cheek_array], color=(255,255,255))
                    rc_mask = rc_mask.astype(bool)

                    masked_frame[rc_mask] = 255

                # Cheeks and Nose
                case [(3,)]: 
                    lc_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                    lc_mask = cv.fillPoly(img=lc_mask, pts=[l_cheek_array], color=(255,255,255))
                    lc_mask = lc_mask.astype(bool)

                    rc_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                    rc_mask = cv.fillPoly(img=rc_mask, pts=[r_cheek_array], color=(255,255,255))
                    rc_mask = rc_mask.astype(bool)

                    nose_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                    nose_mask = cv.fillPoly(img=nose_mask, pts=[nose_array], color=(255,255,255))
                    nose_mask = nose_mask.astype(bool)

                    masked_frame[lc_mask] = 255
                    masked_frame[rc_mask] = 255
                    masked_frame[nose_mask] = 255
                
                # Both eye regions
                case [(4,)]:
                    ler_mask = np.zeros((frame.shape[0],frame.shape[1]), dtype=np.uint8)
                    ler_mask = cv.fillPoly(img=ler_mask, pts=[l_eye_reg_array], color=(255,255,255))
                    ler_mask = ler_mask.astype(bool)

                    rer_mask = np.zeros((frame.shape[0],frame.shape[1]), dtype=np.uint8)
                    rer_mask = cv.fillPoly(img=rer_mask, pts=[r_eye_reg_array], color=(255,255,255))
                    rer_mask = rer_mask.astype(bool)

                    masked_frame[ler_mask] = 255
                    masked_frame[rer_mask] = 255

                # Face Skin
                case [(5,)]:
                    le_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                    le_mask = cv.fillPoly(img=le_mask, pts=[l_eye_array], color=(255,255,255))
                    le_mask = le_mask.astype(bool)

                    re_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                    re_mask = cv.fillPoly(img=re_mask, pts=[r_eye_array], color=(255,255,255))
                    re_mask = re_mask.astype(bool)

                    leb_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                    leb_mask = cv.fillPoly(img=leb_mask, pts=[l_eyebrow_array], color=(255,255,255))
                    leb_mask = leb_mask.astype(bool)

                    reb_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                    reb_mask = cv.fillPoly(img=reb_mask, pts=[r_eyebrow_array], color=(255,255,255))
                    reb_mask = reb_mask.astype(bool)

                    lip_mask = np.zeros((frame.shape[0],frame.shape[1]), dtype=np.uint8)
                    lip_mask = cv.fillPoly(img=lip_mask, pts=[lip_outer_array], color=(255,255,255))
                    lip_mask = lip_mask.astype(bool)

                    oval_mask = np.zeros((frame.shape[0],frame.shape[1]), dtype=np.uint8)
                    oval_mask = cv.fillPoly(img=oval_mask, pts=[face_oval_array], color=(255,255,255))
                    oval_mask = oval_mask.astype(bool)

                    # Masking the face oval
                    masked_frame[oval_mask] = 255
                    masked_frame[le_mask] = 0
                    masked_frame[re_mask] = 0
                    masked_frame[leb_mask] = 0
                    masked_frame[reb_mask] = 0
                    masked_frame[lip_mask] = 0
                
                # Chin
                case [(6,)]:    
                    chin_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                    chin_mask = cv.fillPoly(img=chin_mask, pts=[chin_array], color=(255,255,255))
                    chin_mask = chin_mask.astype(bool)

                    masked_frame[chin_mask] = 255
                
                # Both eyes (sclera)
                case [(8,)]:
                    le_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                    le_mask = cv.fillPoly(img=le_mask, pts=[l_eye_array], color=(255,255,255))
                    le_mask = le_mask.astype(bool)

                    re_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                    re_mask = cv.fillPoly(img=re_mask, pts=[r_eye_array], color=(255,255,255))
                    re_mask = re_mask.astype(bool)

                    masked_frame[le_mask] = 255
                    masked_frame[re_mask] = 255
                
                # Both irises
                case [(9,)]:
                    li_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                    li_mask = cv.fillPoly(img=li_mask, pts=[l_iris_array], color=(255,255,255))
                    li_mask = li_mask.astype(bool)

                    ri_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                    ri_mask = cv.fillPoly(img=ri_mask, pts=[r_iris_array], color=(255,255,255))
                    ri_mask = ri_mask.astype(bool)

                    masked_frame[li_mask] = 255
                    masked_frame[ri_mask] = 255

                # Both eyebrows
                case [(10,)]:
                    leb_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                    leb_mask = cv.fillPoly(img=leb_mask, pts=[l_eyebrow_array], color=(255,255,255))
                    leb_mask = leb_mask.astype(bool)

                    reb_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                    reb_mask = cv.fillPoly(img=reb_mask, pts=[r_eyebrow_array], color=(255,255,255))
                    reb_mask = reb_mask.astype(bool)

                    masked_frame[leb_mask] = 255
                    masked_frame[reb_mask] = 255

                # Lips
                case [(11,)]:
                    lip_o_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                    lip_o_mask = cv.fillPoly(img=lip_o_mask, pts=[lip_outer_array], color=(255,255,255))
                    lip_o_mask = lip_o_mask.astype(bool)

                    lip_i_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                    lip_i_mask = cv.fillPoly(img=lip_i_mask, pts=[lip_inner_array], color=(255,255,255))
                    lip_i_mask = lip_i_mask.astype(bool)

                    masked_frame[lip_o_mask] = 255
                    masked_frame[lip_i_mask] = 0

                case _:
                    cur_landmark_coords = get_relative_landmark_coordinates(landmarker_coordinates, path)
                    cur_lm_array = np.array(cur_landmark_coords, np.int32)

                    # Creating boolean masks for the facial landmarks 
                    bool_mask = np.zeros((frame.shape[0],frame.shape[1]), dtype=np.uint8)
                    bool_mask = cv.fillPoly(img=bool_mask, pts=[cur_lm_array], color=(255,255,255))
                    bool_mask = bool_mask.astype(bool)

                    masked_frame[bool_mask] = 255
    else:
        match landmark_paths:
            # Both Cheeks
            case [(0,)]:
                lc_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                lc_mask = cv.fillPoly(img=lc_mask, pts=[l_cheek_array], color=(255,255,255))
                lc_mask = lc_mask.astype(bool)

                rc_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                rc_mask = cv.fillPoly(img=rc_mask, pts=[r_cheek_array], color=(255,255,255))
                rc_mask = rc_mask.astype(bool)

                masked_frame[lc_mask] = 255
                masked_frame[rc_mask] = 255
            
            # Left Cheek Only
            case [(1,)]:
                lc_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                lc_mask = cv.fillPoly(img=lc_mask, pts=[l_cheek_array], color=(255,255,255))
                lc_mask = lc_mask.astype(bool)

                masked_frame[lc_mask] = 255
            
            # Right Cheek Only
            case [(2,)]:
                rc_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                rc_mask = cv.fillPoly(img=rc_mask, pts=[r_cheek_array], color=(255,255,255))
                rc_mask = rc_mask.astype(bool)

                masked_frame[rc_mask] = 255

            # Cheeks and Nose
            case [(3,)]: 
                lc_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                lc_mask = cv.fillPoly(img=lc_mask, pts=[l_cheek_array], color=(255,255,255))
                lc_mask = lc_mask.astype(bool)

                rc_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                rc_mask = cv.fillPoly(img=rc_mask, pts=[r_cheek_array], color=(255,255,255))
                rc_mask = rc_mask.astype(bool)

                nose_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                nose_mask = cv.fillPoly(img=nose_mask, pts=[nose_array], color=(255,255,255))
                nose_mask = nose_mask.astype(bool)

                masked_frame[lc_mask] = 255
                masked_frame[rc_mask] = 255
                masked_frame[nose_mask] = 255
            
            # Both eye regions
            case [(4,)]:
                ler_mask = np.zeros((frame.shape[0],frame.shape[1]), dtype=np.uint8)
                ler_mask = cv.fillPoly(img=ler_mask, pts=[l_eye_reg_array], color=(255,255,255))
                ler_mask = ler_mask.astype(bool)

                rer_mask = np.zeros((frame.shape[0],frame.shape[1]), dtype=np.uint8)
                rer_mask = cv.fillPoly(img=rer_mask, pts=[r_eye_reg_array], color=(255,255,255))
                rer_mask = rer_mask.astype(bool)

                masked_frame[ler_mask] = 255
                masked_frame[rer_mask] = 255

            # Face Skin
            case [(5,)]:
                le_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                le_mask = cv.fillPoly(img=le_mask, pts=[l_eye_array], color=(255,255,255))
                le_mask = le_mask.astype(bool)

                re_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                re_mask = cv.fillPoly(img=re_mask, pts=[r_eye_array], color=(255,255,255))
                re_mask = re_mask.astype(bool)

                leb_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                leb_mask = cv.fillPoly(img=leb_mask, pts=[l_eyebrow_array], color=(255,255,255))
                leb_mask = leb_mask.astype(bool)

                reb_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                reb_mask = cv.fillPoly(img=reb_mask, pts=[r_eyebrow_array], color=(255,255,255))
                reb_mask = reb_mask.astype(bool)

                lip_mask = np.zeros((frame.shape[0],frame.shape[1]), dtype=np.uint8)
                lip_mask = cv.fillPoly(img=lip_mask, pts=[lip_outer_array], color=(255,255,255))
                lip_mask = lip_mask.astype(bool)

                oval_mask = np.zeros((frame.shape[0],frame.shape[1]), dtype=np.uint8)
                oval_mask = cv.fillPoly(img=oval_mask, pts=[face_oval_array], color=(255,255,255))
                oval_mask = oval_mask.astype(bool)

                # Masking the face oval
                masked_frame[oval_mask] = 255
                masked_frame[le_mask] = 0
                masked_frame[re_mask] = 0
                masked_frame[leb_mask] = 0
                masked_frame[reb_mask] = 0
                masked_frame[lip_mask] = 0
            
            # Chin
            case [(6,)]:    
                chin_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                chin_mask = cv.fillPoly(img=chin_mask, pts=[chin_array], color=(255,255,255))
                chin_mask = chin_mask.astype(bool)

                masked_frame[chin_mask] = 255
            
            # Both eyes (sclera)
            case [(8,)]:
                le_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                le_mask = cv.fillPoly(img=le_mask, pts=[l_eye_array], color=(255,255,255))
                le_mask = le_mask.astype(bool)

                re_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                re_mask = cv.fillPoly(img=re_mask, pts=[r_eye_array], color=(255,255,255))
                re_mask = re_mask.astype(bool)

                masked_frame[le_mask] = 255
                masked_frame[re_mask] = 255
            
            # Both irises
            case [(9,)]:
                li_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                li_mask = cv.fillPoly(img=li_mask, pts=[l_iris_array], color=(255,255,255))
                li_mask = li_mask.astype(bool)

                ri_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                ri_mask = cv.fillPoly(img=ri_mask, pts=[r_iris_array], color=(255,255,255))
                ri_mask = ri_mask.astype(bool)

                masked_frame[li_mask] = 255
                masked_frame[ri_mask] = 255

            # Both eyebrows
            case [(10,)]:
                leb_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                leb_mask = cv.fillPoly(img=leb_mask, pts=[l_eyebrow_array], color=(255,255,255))
                leb_mask = leb_mask.astype(bool)

                reb_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                reb_mask = cv.fillPoly(img=reb_mask, pts=[r_eyebrow_array], color=(255,255,255))
                reb_mask = reb_mask.astype(bool)

                masked_frame[leb_mask] = 255
                masked_frame[reb_mask] = 255

            # Lips
            case [(11,)]:
                lip_o_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                lip_o_mask = cv.fillPoly(img=lip_o_mask, pts=[lip_outer_array], color=(255,255,255))
                lip_o_mask = lip_o_mask.astype(bool)

                lip_i_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                lip_i_mask = cv.fillPoly(img=lip_i_mask, pts=[lip_inner_array], color=(255,255,255))
                lip_i_mask = lip_i_mask.astype(bool)

                masked_frame[lip_o_mask] = 255
                masked_frame[lip_i_mask] = 0

            case _:
                cur_landmark_coords = get_relative_landmark_coordinates(landmarker_coordinates, landmark_paths)
                cur_lm_array = np.array(cur_landmark_coords, np.int32)
                
                # Creating boolean masks for the facial landmarks 
                bool_mask = np.zeros((frame.shape[0],frame.shape[1]), dtype=np.uint8)
                bool_mask = cv.fillPoly(img=bool_mask, pts=[cur_lm_array], color=(255,255,255))
                bool_mask = bool_mask.astype(bool)

                masked_frame[bool_mask] = 255

    return masked_frame.astype(np.uint8)

__all__ = [mask_from_landmarks]