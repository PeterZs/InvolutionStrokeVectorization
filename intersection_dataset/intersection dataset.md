run it

```
uv run main.py --out_dir sample_output --from_seed 42 --to_seed 42 --closeness 3 10 --npoints 10 80 --n_add 0 4
```

```
uv run main.py --out_dir sample_output --from_seed 96500 --to_seed 96500 --closeness 3 10 --npoints 10 80 --n_add 0 4
```


```
uv run main.py --out_dir debug_files/train/Dataset2 --from_seed 1 --to_seed 100 --closeness 3 10 --npoints 10 80 --n_add 0 4
```

sample test dataset
```
uv run main.py --out_dir debug_files/test/SampleDataset --from_seed 500001 --to_seed 500100 --closeness 3 10 --npoints 10 80 --n_add 0 4
```

other method
```
uv run main_copy.py --out_dir debug_files/method2 --from_seed 42 --to_seed 42 --n_add 0 2 --debug
```

on cluster
```
bash scripts/submit.sh
```




#### analyse data
```
srun uv run python -u scripts/analyze_dataset.py /cluster/project/sorkine/pgerstner/intersection/Dataset5
```

#### remove files
```
cd /cluster/project/sorkine/pgerstner/intersection/NAME
for f in ./*; do echo "Removing $f"; rm -rf -- "$f"; done
or
find . -maxdepth 1 ! -name '.' ! -name '..' -print0 | xargs -0 -P 4 -I {} bash -c 'echo "Removing {}"; rm -rf -- "{}"'
```
#### tar creation
```
bash scripts/submit_make_tars.sh Dataset5 train/Dataset5Archive
```
check
```
ls /cluster/project/sorkine/pgerstner/intersection
```

```
bash scripts/run_python.sh scripts/make_tar.py --base_dir /cluster/project/sorkine/pgerstner/intersection/Iteration3 --tar_dir /cluster/project/sorkine/pgerstner/intersection/Dataset3.tar --level 1
```
locally
```
uv run scripts/make_tar.py --base_dir debug_files/train --tar_dir debug_files/SampleDataset.tar --level 1
```

Sample single polyline

a polyline is valid if all of this holds:
- all self loops have a maximum inscribed circle of radius at least d/2
- all intersection points are at least distance d apart
- it satisfies the red points and black points criterion

todo
- [x] sampling with backtracking strategy
- [ ] remove all coordinates that overlap -> not necessary if we check
- [x] add red-black criterion back
- [x] add check
- [x] refactor a bit
- [x] add triangulation thing to sampler
- [x] implement case where there is more than 1 end node and 1 through node
- [ ] change path sampling so that "central" node is selected with probability 0.5

it occured to me:
- [x] The diagonals all need to be 1, since a path might end at any intersection
- [x] we need more information to reconstruct the path
	- [x] -> add pairs of half-edges
- [ ] paths that are adjacent should be merged
- [ ] if there is a "weird" ordering, reorder it to a canonical ordering.
- [ ] reject samples with only 1 path


We have to reverify the paths after tangents have been computed
- if at a node there is a non-crossing path and 2 crossing paths, it is invalid
- if node has degree 2 and paths start/end there, they have to be merged -> maybe?
- 



results
```
--- Column Sums ---
total_3             : 331867
total_4             : 1111350
total_5             : 99246
total_6             : 352142
total_7             : 7749
total_8             : 32343
total_9             : 206
total_10            : 1041
all_tangent_3       : 0
all_tangent_4       : 547247
all_tangent_5       : 16242
all_tangent_6       : 199509
all_tangent_7       : 2082
all_tangent_8       : 21850
all_tangent_9       : 83
all_tangent_10      : 796
no_tangent_3        : 331867
no_tangent_4        : 514603
no_tangent_5        : 44621
no_tangent_6        : 45384
no_tangent_7        : 898
no_tangent_8        : 727
no_tangent_9        : 3
no_tangent_10       : 5
terminal_3          : 331867
terminal_4          : 184513
terminal_5          : 99246
terminal_6          : 53007
terminal_7          : 7749
terminal_8          : 4416
terminal_9          : 206
terminal_10         : 131
```