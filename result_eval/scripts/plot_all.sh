DIR=result_data8
methods="Ours Ours*(smoothed) DeepSketch2024 Mo2021 Puhachov2021 Bessmeltsev2019"
# uv run plot.py "Stroke Density ratio" $DIR --reorder $methods --ymax 12.5 --hline 1.0 --output "density.pdf"
# uv run plot.py "Chamfer Distance" $DIR --reorder $methods --ymax 10 --output "chamfer.pdf"
# uv run plot.py "Length diff" $DIR --reorder $methods --ymax 2500 --output "length.pdf"

# uv run plot.py "Length diff" $DIR --reorder $methods --ymax 2500 --iqr --output "length_iqr.pdf"
# uv run plot.py "Stroke Density ratio" $DIR --reorder $methods --ymax 12.5 --hline 1.0 --iqr --output "density_iqr.pdf"
# uv run plot.py "Chamfer Distance" $DIR --reorder $methods --ymax 10 --iqr --output "chamfer_iqr.pdf"
# uv run plot.py "Chamfer Distance" $DIR --reorder $methods --rescale_y 0.9 --ymax 9 --iqr --output "chamfer_iqr2.pdf"
# uv run plot.py "Length diff" $DIR --reorder $methods --rescale_y 0.85 --ymax 2400 --iqr --output "length_iqr2.pdf"
#uv run plot.py "Stroke Density ratio" $DIR --reorder $methods --rescale_y 0.9 --ymax 12.5 --hline 1.0 --iqr --output "density_iqr2.pdf"
#uv run plot.py "Stroke Density ratio" $DIR --reorder $methods --ymax 15 --hline 1.0 --iqr --output "density_iqr0.pdf"
#uv run plot.py "Stroke Density ratio" $DIR --reorder $methods --break-y 0 10 10.5 40 --break-ratio 5.0 --hline 1.0 --iqr --output "density_iqr1.pdf"
# uv run plot.py "Stroke Density ratio" $DIR --reorder $methods --break-y 0 10 10.5 40 --break-ratio 5.0 --rescale_y 0.85 --hline 1.0 --iqr --output "density_iqr2.pdf"
# methods="Ours Ours*(smoothed) Ours*v2*nonsmooth DeepSketch2024 Mo2021 Puhachov2021 Bessmeltsev2019"
uv run plot.py "Chamfer Distance" $DIR --reorder $methods --rescale_y 0.9 --ymax 9 --iqr --output "plots/chamfer_iqr2.pdf"
uv run plot.py "Length diff" $DIR --reorder $methods --rescale_y 0.85 --ymax 2400 --iqr --output "plots/length_iqr2.pdf"
uv run plot.py "Stroke Density ratio" $DIR --reorder $methods --break-y 0 10 10.5 40 --break-ratio 5.0 --rescale_y 0.85 --hline 1.0 --iqr --output "plots/density_iqr2.pdf"
# methods="Ours DeepSketch2024 Puhachov2021 Bessmeltsev2019"
# uv run plot.py "Turning Angle Histogram Distance" $DIR --reorder $methods --iqr --output "turnangle.pdf"

#uv run make_table.py $DIR --reorder Ours DeepSketch2024 Mo2021 Puhachov2021 Bessmeltsev2019