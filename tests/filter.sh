#!/bin/bash

for mod in train val 
do
    sed -i.orig "s/'/\"/g" solutions_$mod.txt
    sed -i 's/ignore_color"/ignore_colors"/g' solutions_$mod.txt
    sed -i 's/intersec"/intersect"/g' solutions_$mod.txt
    sed -i 's/directon/direction/g' solutions_$mod.txt
    sed -i 's/bottomright/bottom_right/g' solutions_$mod.txt
    sed -i 's/bottomleft/bottom_left/g' solutions_$mod.txt
    sed -i 's/topright/top_right/g' solutions_$mod.txt
    sed -i 's/topleft/top_left/g' solutions_$mod.txt
    sed -i 's/down/bottom/g' solutions_$mod.txt
    sed -i 's/up/top/g' solutions_$mod.txt
done
