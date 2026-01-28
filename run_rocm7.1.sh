docker run -it --privileged  --group-add sudo -w /workspace -v ${HOME}/workspace:/workspace  rocm/composable_kernel:ck_ub24.04_rocm7.0.1_amd-mainline /bin/bash
#--net=host rocm/composable_kernel:ck_ub24.04_rocm7.0.1_amd-mainline
