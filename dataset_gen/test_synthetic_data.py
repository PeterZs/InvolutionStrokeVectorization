from dataset_loaders.synthetic_loader import get_curvy_star, get_line_star, SyntheticDataset
from itertools import islice
import svgutils


def test_manual():
    BASE_DIR = "debug_files/synthetic_data"
    MIN_ANGLE = 15
    RADIUS = 400
    SVGSIZE = 1000
    test = [3, 4, 5, 6, 7]
    for idx, v in enumerate(test):
        print("-----------------------------")
        with open(f"{BASE_DIR}/line_star{idx}.svg", "w") as f:
            f.write(get_line_star(v, RADIUS, MIN_ANGLE, SVGSIZE))
    
    test = [(3, ""), (3, "evenodd"), (3, "meet_on_line"),
            (4, ""), (4, "evenodd"), (4, "meet_on_line"),
            (5, ""), (5, "evenodd"), (5, "meet_on_line"),
            (8, ""), (8, "evenodd"), (8, "meet_on_line"),
            (9, ""), (9, "evenodd"), (9, "meet_on_line"),
    ]
    for idx, v in enumerate(test):
        print("-----------------------------")
        with open(f"{BASE_DIR}/curvy_star{idx}.svg", "w") as f:
            f.write(get_curvy_star(v[0], MIN_ANGLE, v[1]))

def test_generator():
    d = SyntheticDataset()
    
    for svg in islice(d.sample_loader("train", 1000), 2):
        print(svg)
        svgutils.save_webp_from_svg(svg, "debug.svg")

def main():
    test_manual()
    test_generator()
   
    
            
    

if __name__ == "__main__":
    main()