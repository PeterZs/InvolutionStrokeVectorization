import graph_extraction2
import utils.utils as utils

PATHS = [
    # "debug_files/hat/hat_centerline.png",
    # "debug_files/balloon/balloon_centerline.png",
    # "debug_files/tc/tc_centerline.png",
    # "debug_files/star/star_centerline.png",
    # "debug_files/star2/star2_centerline.png",
    # "debug_files/motobike/motobike_centerline2.png",
    # "debug_files/motobike/motobike_centerline3.png",
    # "debug_files/motobike/motobike_centerline4.png",
    # "debug_files/motobike/motobike_centerline5.png",
    # "debug_files/motobike/motobike_centerline6.png",
    # "debug_files/adventure2/adventure2_centerline.png"
    # "debug_files/adventure2/adventure2_centerline2.png",
    # "debug_files/adventure2/adventure2_centerline3.png",
    # "debug_files/testcases/4.png"
    # "debug_files/testcases/5.png",
    # "debug_files/testcases/7.png",
    # "debug_files/testcases/10.png",
    # "debug_files/testcases/14.png",
    "debug_files/testcases/15.png",
    "debug_files/testcases/9.png",
    "debug_files/testcases/16.png",
    "debug_files/testcases/17.png",
    "debug_files/testcases/12.png",
    "debug_files/testcases/19.png",
    "debug_files/testcases/20.png",
    # "debug_files/motobike/motobike_centerline.png",
    # "debug_files/architecture/architecture_centerline.png"
    # "debug_files/wizard/wizard_centerline.png",
    # "debug_files/motobike/motobike_centerline_small.png",
    # "debug_files/test/test2_centerline.png",
    # "debug_files/star/star_centerline2.png",
    # "debug_files/carbig/carbig_centerline.png",
    # "debug_files/car/car_centerline.png",
    # "debug_files/slow/slow_centerline.png",
    # "bigtest.png",
]

for p in PATHS:
    graph_extraction2.process(p, debug_dir="debug_files")
    # with utils.timer("total"):
    #     graph_extraction2.process(p)
