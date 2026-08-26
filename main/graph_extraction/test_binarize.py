import cv2
import os
import numpy as np

import image
from build_linegraph import *



CASES_DIR = "testcases/input"
caselist = os.listdir(CASES_DIR)
caselist = ["11.png"]
for case in caselist: 
    if not case.endswith(".png") or case.endswith("mask.png"):
        continue
    

    img = cv2.imread(f"testcases/input/{case}", cv2.IMREAD_GRAYSCALE)


    OUT_DIR = f"debug_files/{case[:-4]}"
    os.makedirs(OUT_DIR, exist_ok=True)
    orig = get_binary(img)
    cv2.imwrite(f"{OUT_DIR}/origbin.png", orig)
    cv2.imwrite(f"{OUT_DIR}/orig.png", img)
    orig2 = get_binary_2(img.copy())
    cv2.imwrite(f"{OUT_DIR}/origbin2.png", orig2)
    
    orig3 = get_binary_3(img.copy())
    cv2.imwrite(f"{OUT_DIR}/origbin3.png", orig3)

    img[img < 10] = 0 

    binary = image.thresh_by_max_component_intensity(img, 50)
    # binary = img.copy()
    # binary[binary>0] = 255
    cv2.imwrite(f"{OUT_DIR}/bin.png", binary)


    # light = img.copy()
    # light[light>50] = 0
    # light[light>0] = 255
    # cv2.imwrite(f"{OUT_DIR}/light.png", light)


    # kernel = circ(2)
    # msk, detected = detect_and_mark(binary, kernel, kernel)
    # kernel = circ(3)
    # msk2, detected2 = detect_and_mark(binary, kernel, kernel)
    # cv2.imwrite(f"{OUT_DIR}/msk1.png", msk)
    # cv2.imwrite(f"{OUT_DIR}/msk2.png", msk2)
    # msk_sub = cv2.subtract(msk, msk2)
    # kernel = circ(2) 
    # msk_sub2, _ = detect_and_mark(msk_sub, kernel, kernel)
    # cv2.imwrite(f"{OUT_DIR}/msk_sub.png", msk_sub)
    # cv2.imwrite(f"{OUT_DIR}/msk_sub2.png", msk_sub2 )

    adjusted = image.adaptive(img, size=9)
    adjusted = 255-adjusted
    cv2.imwrite(f"{OUT_DIR}/adaptivethresh.png", adjusted)
    # adjusted = cv2.subtract(adjusted, msk)
    # cv2.imwrite(f"{DIR}/adaptivethresh2.png", adjusted)


    binary2 = cv2.subtract(binary, adjusted)
    cv2.imwrite(f"{OUT_DIR}/bin2.png", binary2)
    #clean up small regions
    binary3 = image.thresh_by_max_component_intensity(img, 50, binary2)
    cv2.imwrite(f"{OUT_DIR}/bin3.png", binary3)
    
    binary32 = image.remove_holes_by_kernel(binary3, image.circ(1))
    cv2.imwrite(f"{OUT_DIR}/bin32.png", binary32)

    
    # medial = image.skeleton(binary3)
    # cv2.imwrite(f"{OUT_DIR}/medial.png", medial)

    # nodemask = primitive_graph(binary3)
    # cv2.imwrite(f"{OUT_DIR}/nodemask.png", nodemask)
    # # nodemask = image.get_skeleton_nodes(binary3)
    # # k = (nodemask != 2 ).astype(np.uint8) *255
    
    # msk_dilated = cv2.dilate(nodemask, np.ones((5, 5), dtype=np.uint8))
    # cv2.imwrite(f"{OUT_DIR}/mask_dilated.png", msk_dilated)
    # msk_restrained = cv2.bitwise_and(msk_dilated, binary)
    # binary8 = cv2.bitwise_or(binary3, msk_restrained)
    # cv2.imwrite(f"{OUT_DIR}/bin8.png", binary8)

    # cv2.imwrite(f"{OUT_DIR}/nodemask.png", k)
 
    # holes3 = image.detect_holes(binary3, maxarea=50, connectivity_8=True)
    # cv2.imwrite(f"{OUT_DIR}/holes_bin3.png", holes3)

    # diff2 = cv2.subtract(binary, binary3)
    # cv2.imwrite(f"{OUT_DIR}/diff_bin3.png", diff2)
    # binary4 = cv2.subtract(binary2, light)
    # cv2.imwrite(f"{DIR}/bin4.png", binary4 )


    # bin_dilated = cv2.dilate(binary3, image.circ(2))
    # cv2.imwrite(f"{OUT_DIR}/bin_dilated.png", bin_dilated)
    # kernel = image.circ(4)
    # r1, detected = image.detect_and_mark(bin_dilated, kernel, kernel)
    # cv2.imwrite(f"{OUT_DIR}/r1.png", r1)
    # cv2.imwrite(f"{OUT_DIR}/detected.png", detected)
    # # light_2 = cv2.subtract(light, r1)
    
    # detected2 = image.remove_components_by_kernel(detected, image.circ(1), inverse=True)
    # cv2.imwrite(f"{OUT_DIR}/detected2.png", detected2)
    # r1_2 = cv2.dilate(detected2, kernel)
    # cv2.imwrite(f"{OUT_DIR}/r1_2.png", r1_2)

    # #and it with binary to not add too much
    # r2 = cv2.bitwise_and(binary, r1_2)
    # cv2.imwrite(f"{OUT_DIR}/r2.png", r2)
    
    # # r3 = image.remove_components_with_holes(r2)
    # # cv2.imwrite(f"{OUT_DIR}/r3.png", r3)
    # #add it back
    
    # binary4 = cv2.bitwise_or(binary3, r2)
    # binary4 = cv2.subtract(binary4, holes3)
    # cv2.imwrite(f"{OUT_DIR}/bin4.png", binary4)
    
    # bin4_inv = 255-binary4
    # kernel = image.circ(1) 
    # b4a, d = image.detect_and_mark(bin4_inv, kernel, kernel)
    # bin4_inv = cv2.subtract(bin4_inv, b4a)
    # cv2.imwrite(f"{OUT_DIR}/b4inv.png", bin4_inv)
    # gaps = image.remove_components_area(bin4_inv, 5, keepsmall=True)
    # cv2.imwrite(f"{OUT_DIR}/small.png", gaps)

    # gaps = cv2.bitwise_and(gaps, binary)
    # #rem = 255-b4a
    # #rem = cv2.subtract(rem, binary4)
    # cv2.imwrite(f"{OUT_DIR}/detect4.png", gaps)
    
    # bin_42 = cv2.bitwise_or(gaps, binary4)
    # cv2.imwrite(f"{OUT_DIR}/binary4_2.png", bin_42)
    
    # diff =cv2.subtract(binary, binary4)
    # cv2.imwrite(f"{OUT_DIR}/diff.png", diff)

    # diff2 = image.remove_components_by_kernel(diff, np.ones((3, 3)))
    # cv2.imwrite(f"{OUT_DIR}/diff2.png", diff2)

    # binary5 = cv2.bitwise_or(binary4, diff2)
    # cv2.imwrite(f"{OUT_DIR}/bin5.png", binary5)
     
    # holes = image.detect_holes(binary5, 4 , connectivity_8=True)
    # bin_cleaned = cv2.bitwise_or(binary5, holes)
    # bin_cleaned = image.remove_spurious_spikes(bin_cleaned)
    # cv2.imwrite(f"{OUT_DIR}/bin_cleaned.png", bin_cleaned)
    # light_2 = cv2.subtract(light, r2)
    # binary6 = cv2.subtract(binary5, light_2)
    # cv2.imwrite(f"{DIR}/bin6.png", binary6)



