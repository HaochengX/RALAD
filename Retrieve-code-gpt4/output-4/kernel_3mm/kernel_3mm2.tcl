open_project -reset /home/haotian/RALAD/Retrieve-code-gpt4/kernel_3mm/kernel_3mm
set_top kernel_3mm
add_files /home/haotian/RALAD/Retrieve-code-gpt4/kernel_3mm/kernel_3mm2.c -cflags "-std=c++11"  ;# Use appropriate C++ standard

open_solution -reset "solution1"
set_part {xcu250-figd2104-2L-e}
create_clock -period 5.0ns  ;# Adjust clock period as needed based on your design

# Run synthesis and export design
csynth_design
export_design -format ip_catalog -version "1.0"

exit
