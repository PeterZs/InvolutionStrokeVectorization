DIR=result_data10
VERSION_PICK=v110_smooth
methods="$VERSION_PICK DeepSketch2024 Mo2021 Puhachov2021 Bessmeltsev2019"
uv run plot.py "Chamfer Distance" $DIR --reorder $methods --rename $VERSION_PICK Ours --rescale_y 0.9 --ymax 9 --iqr --output "plots/paper_chamfer_iqr.pdf"
uv run plot.py "Length diff" $DIR --reorder $methods --rename $VERSION_PICK Ours --rescale_y 0.85 --ymax 2400 --iqr --output "plots/paper_length_iqr.pdf"
uv run plot.py "Stroke Density ratio" $DIR --reorder $methods --rename $VERSION_PICK Ours --break-y 0 10 10.5 40 --break-ratio 5.0 --rescale_y 0.85 --hline 1.0 --iqr --output "plots/paper_density_iqr.pdf"
