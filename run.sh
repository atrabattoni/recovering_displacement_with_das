wget https://zenodo.org/record/8152132/files/recovering_displacement_with_das.zip
tar -xzf recovering_displacement_with_das.zip

mkdir -p results
mkdir -p figs

echo "Figure 1"
python 1_illustrative_simulation.py

echo "Figure 3"
python 3_das_sensitivity_to_displacement.py

echo "Figure 4"
python 4_full_waveform_simulation.py

echo "Figure 5"
python 5_comparison_with_colocated_seismometers.py

echo "Figure 6"
python 6_optimal_window_length_search.py

echo "Figure 7"
python 7_direct_p_wave_enhancement.py

echo "Figure 8"
python info_catalog.py
python das_catalog.py
python 8_magnitude_estimation.py
