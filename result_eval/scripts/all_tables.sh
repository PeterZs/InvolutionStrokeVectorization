DIR=result_data8
methods="Ours Ours*(smoothed) Ours*cmo2*smooth DeepSketch2024 Mo2021 Puhachov2021 Bessmeltsev2019"

uv run make_table.py "Chamfer Distance" $DIR 1024 --reorder $methods
uv run make_table.py "Length diff" $DIR 1024 --reorder $methods
uv run make_table.py "Stroke Density ratio" $DIR 1024 --reorder $methods
uv run make_intersections_table.py $DIR --reorder $methods