source run_rocm7.1.sh
cd fmha
hipcc fmha_host.cpp -o fmha
./fmha 4
exit