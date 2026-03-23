docker exec -it peaceful_arliu bash
cd fmha
hipcc -std=c++17 --offload-arch=gfx942 -O3 fmha_host.cpp -o fmha
./fmha 0
