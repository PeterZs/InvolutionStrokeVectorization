DIR=result_data10

uv run make_metrics_table.py "Chamfer Distance" $DIR 1024
uv run make_metrics_table.py "Length diff" $DIR 1024
uv run make_metrics_table.py "Stroke Density ratio" $DIR 1024
uv run make_intersections_table.py $DIR
uv run make_intersections_table.py $DIR --skip-blank