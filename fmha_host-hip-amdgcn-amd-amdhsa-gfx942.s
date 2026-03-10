	.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
	.amdhsa_code_object_version 6
	.text
	.protected	_Z9fmha_mfmaPKDF16bS0_S0_PDF16_PKiiiiiiif ; -- Begin function _Z9fmha_mfmaPKDF16bS0_S0_PDF16_PKiiiiiiif
	.globl	_Z9fmha_mfmaPKDF16bS0_S0_PDF16_PKiiiiiiif
	.p2align	8
	.type	_Z9fmha_mfmaPKDF16bS0_S0_PDF16_PKiiiiiiif,@function
_Z9fmha_mfmaPKDF16bS0_S0_PDF16_PKiiiiiiif: ; @_Z9fmha_mfmaPKDF16bS0_S0_PDF16_PKiiiiiiif
.Lfunc_begin0:
	.file	0 "/workspace/fmha" "fmha_host.cpp" md5 0xd84070ad1ea8f68c6c2c4e28764d0e03
	.file	1 "." "fmha_mfma_kernel.hpp"
	.loc	1 34 0                          ; ./fmha_mfma_kernel.hpp:34:0
	.cfi_sections .debug_frame
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; 
	.cfi_undefined 16
	s_load_dwordx2 s[6:7], s[0:1], 0x20
.Ltmp0:
	.file	2 "/opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail" "amd_hip_runtime.h"
	.loc	2 261 116 prologue_end          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_runtime.h:261:116 @[ /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_runtime.h:289:160 @[ ./fmha_mfma_kernel.hpp:38:27 ] ]
	s_load_dwordx4 s[8:11], s[0:1], 0x2c
.Ltmp1:
	.loc	1 49 30                         ; ./fmha_mfma_kernel.hpp:49:30
	s_ashr_i32 s5, s4, 31
	s_lshl_b64 s[12:13], s[4:5], 2
	.loc	1 79 5                          ; ./fmha_mfma_kernel.hpp:79:5
	v_lshlrev_b32_e32 v6, 2, v0
	.loc	1 49 30                         ; ./fmha_mfma_kernel.hpp:49:30
	s_waitcnt lgkmcnt(0)
	s_add_u32 s6, s6, s12
	s_addc_u32 s7, s7, s13
	s_load_dwordx2 s[18:19], s[6:7], 0x0
.Ltmp2:
	.loc	2 261 116                       ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_runtime.h:261:116 @[ /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_runtime.h:289:160 @[ ./fmha_mfma_kernel.hpp:38:27 ] ]
	s_load_dwordx2 s[16:17], s[0:1], 0x3c
	s_load_dwordx4 s[12:15], s[0:1], 0x10
	v_mov_b32_e32 v5, 0
.Ltmp3:
	.loc	1 80 5                          ; ./fmha_mfma_kernel.hpp:80:5
	v_lshlrev_b32_e32 v7, 1, v0
	.loc	1 55 38                         ; ./fmha_mfma_kernel.hpp:55:38
	s_waitcnt lgkmcnt(0)
	s_ashr_i32 s2, s18, 31
	.loc	1 55 51 is_stmt 0               ; ./fmha_mfma_kernel.hpp:55:51
	s_mul_i32 s25, s16, s9
	s_mul_hi_i32 s6, s16, s9
	.loc	1 55 66                         ; ./fmha_mfma_kernel.hpp:55:66
	s_mul_hi_u32 s7, s25, s18
	s_mul_i32 s2, s25, s2
	s_add_i32 s2, s7, s2
	s_mul_i32 s6, s6, s18
	.loc	1 56 40 is_stmt 1               ; ./fmha_mfma_kernel.hpp:56:40
	s_mul_i32 s7, s16, s3
	.loc	1 51 38                         ; ./fmha_mfma_kernel.hpp:51:38
	s_sub_i32 s24, s19, s18
	.loc	1 55 66                         ; ./fmha_mfma_kernel.hpp:55:66
	s_add_i32 s2, s2, s6
	s_mul_i32 s6, s25, s18
	.loc	1 56 31                         ; ./fmha_mfma_kernel.hpp:56:31
	s_ashr_i32 s9, s7, 31
	.loc	1 56 29 is_stmt 0               ; ./fmha_mfma_kernel.hpp:56:29
	s_add_u32 s6, s6, s7
	s_addc_u32 s7, s2, s9
	.loc	1 64 30 is_stmt 1               ; ./fmha_mfma_kernel.hpp:64:30
	s_lshl_b64 s[20:21], s[6:7], 1
	s_add_u32 s18, s12, s20
	s_addc_u32 s19, s13, s21
	.loc	1 83 44                         ; ./fmha_mfma_kernel.hpp:83:44
	s_ashr_i32 s2, s11, 31
	s_lshr_b32 s2, s2, 29
	s_add_i32 s2, s11, s2
	s_ashr_i32 s9, s2, 3
	.loc	1 84 28                         ; ./fmha_mfma_kernel.hpp:84:28
	s_abs_i32 s6, s9
	v_cvt_f32_u32_e32 v1, s6
	s_ashr_i32 s16, s2, 31
	s_sub_i32 s2, 0, s6
	s_load_dwordx2 s[12:13], s[0:1], 0x0
	v_rcp_iflag_f32_e32 v1, v1
	v_mov_b32_e32 v4, 0
	.loc	1 79 17                         ; ./fmha_mfma_kernel.hpp:79:17
	ds_write_b32 v6, v5 offset:8320
	.loc	1 80 25                         ; ./fmha_mfma_kernel.hpp:80:25
	ds_write_b16 v7, v5 offset:9344
	.loc	1 84 28                         ; ./fmha_mfma_kernel.hpp:84:28
	v_mul_f32_e32 v1, 0x4f7ffffe, v1
	v_cvt_u32_f32_e32 v1, v1
	s_nop 0
	v_readfirstlane_b32 s7, v1
	s_mul_i32 s2, s2, s7
	s_mul_hi_u32 s2, s7, s2
	s_add_i32 s7, s7, s2
	.loc	1 85 28                         ; ./fmha_mfma_kernel.hpp:85:28
	v_mul_hi_u32 v1, v0, s7
	v_mul_lo_u32 v2, v1, s6
	.loc	1 84 28                         ; ./fmha_mfma_kernel.hpp:84:28
	s_lshr_b32 s2, s7, 24
	.loc	1 85 28                         ; ./fmha_mfma_kernel.hpp:85:28
	v_sub_u32_e32 v2, v0, v2
	.loc	1 84 28                         ; ./fmha_mfma_kernel.hpp:84:28
	s_mul_i32 s22, s2, s6
	.loc	1 85 28                         ; ./fmha_mfma_kernel.hpp:85:28
	v_add_u32_e32 v3, 1, v1
	v_cmp_le_u32_e32 vcc, s6, v2
	.loc	1 84 28                         ; ./fmha_mfma_kernel.hpp:84:28
	s_sub_i32 s22, 0x100, s22
	s_add_i32 s23, s2, 1
	.loc	1 85 28                         ; ./fmha_mfma_kernel.hpp:85:28
	v_cndmask_b32_e32 v1, v1, v3, vcc
	v_subrev_u32_e32 v3, s6, v2
	.loc	1 84 28                         ; ./fmha_mfma_kernel.hpp:84:28
	s_sub_i32 s26, s22, s6
	.loc	1 85 28                         ; ./fmha_mfma_kernel.hpp:85:28
	v_cndmask_b32_e32 v2, v2, v3, vcc
	.loc	1 84 28                         ; ./fmha_mfma_kernel.hpp:84:28
	s_cmp_ge_u32 s22, s6
	.loc	1 85 28                         ; ./fmha_mfma_kernel.hpp:85:28
	v_add_u32_e32 v3, 1, v1
	v_cmp_le_u32_e32 vcc, s6, v2
	.loc	1 84 28                         ; ./fmha_mfma_kernel.hpp:84:28
	s_cselect_b32 s2, s23, s2
	s_cselect_b32 s22, s26, s22
	.loc	1 85 28                         ; ./fmha_mfma_kernel.hpp:85:28
	v_cndmask_b32_e32 v1, v1, v3, vcc
	.loc	1 84 28                         ; ./fmha_mfma_kernel.hpp:84:28
	s_add_i32 s23, s2, 1
	.loc	1 85 28                         ; ./fmha_mfma_kernel.hpp:85:28
	v_xor_b32_e32 v9, s16, v1
	.loc	1 84 28                         ; ./fmha_mfma_kernel.hpp:84:28
	s_cmp_ge_u32 s22, s6
	.loc	1 85 28                         ; ./fmha_mfma_kernel.hpp:85:28
	v_subrev_u32_e32 v1, s16, v9
	.loc	1 84 28                         ; ./fmha_mfma_kernel.hpp:84:28
	s_cselect_b32 s2, s23, s2
	.loc	1 86 29                         ; ./fmha_mfma_kernel.hpp:86:29
	v_mul_lo_u32 v8, v1, s9
	.loc	1 84 28                         ; ./fmha_mfma_kernel.hpp:84:28
	s_xor_b32 s26, s2, s16
	.loc	1 86 29                         ; ./fmha_mfma_kernel.hpp:86:29
	v_sub_u32_e32 v2, v0, v8
	.loc	1 84 28                         ; ./fmha_mfma_kernel.hpp:84:28
	s_sub_i32 s2, s26, s16
	.loc	1 86 48                         ; ./fmha_mfma_kernel.hpp:86:48
	v_lshlrev_b32_e32 v14, 3, v2
	.loc	1 88 28                         ; ./fmha_mfma_kernel.hpp:88:28
	v_cmp_gt_i32_e64 s[6:7], s24, v1
	v_mov_b32_e32 v3, 0
	v_mov_b32_e32 v2, 0
	.loc	1 88 5 is_stmt 0                ; ./fmha_mfma_kernel.hpp:88:5
	s_and_saveexec_b64 s[22:23], s[6:7]
	s_cbranch_execz .LBB0_4
; %bb.1:                                ; %.lr.ph.preheader
	.loc	1 0 5                           ; ./fmha_mfma_kernel.hpp:0:5
	s_load_dwordx2 s[0:1], s[0:1], 0x8
	s_movk_i32 s28, 0x208
	.loc	1 88 5                          ; ./fmha_mfma_kernel.hpp:88:5
	v_mul_lo_u32 v3, v9, s28
	v_lshl_add_u32 v3, v0, 4, v3
	v_lshlrev_b32_e32 v4, 4, v8
	.loc	1 63 30 is_stmt 1               ; ./fmha_mfma_kernel.hpp:63:30
	s_waitcnt lgkmcnt(0)
	s_add_u32 s0, s0, s20
	s_addc_u32 s1, s1, s21
	.loc	1 88 5                          ; ./fmha_mfma_kernel.hpp:88:5
	s_lshl_b32 s9, s9, 3
	s_sub_i32 s9, s25, s9
	v_mul_lo_u32 v2, v1, s9
	v_sub_u32_e32 v3, v3, v4
	s_mulk_i32 s16, 0x208
	s_mulk_i32 s26, 0x208
	s_mul_i32 s27, s25, s2
	v_lshl_add_u32 v2, v0, 3, v2
	v_subrev_u32_e32 v4, s16, v3
	s_sub_i32 s9, s26, s16
	s_mov_b64 s[20:21], 0
	v_mov_b32_e32 v5, v1
.LBB0_2:                                ; %.lr.ph
                                        ; =>This Inner Loop Header: Depth=1
	.loc	1 89 74                         ; ./fmha_mfma_kernel.hpp:89:74
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[8:9], v[2:3], 1, s[0:1]
	.loc	1 89 56 is_stmt 0               ; ./fmha_mfma_kernel.hpp:89:56
	global_load_dwordx4 v[8:11], v[8:9], off
	.loc	1 88 43 is_stmt 1               ; ./fmha_mfma_kernel.hpp:88:43
	v_add_u32_e32 v5, s2, v5
	.loc	1 88 28 is_stmt 0               ; ./fmha_mfma_kernel.hpp:88:28
	v_cmp_le_i32_e32 vcc, s24, v5
	v_add_u32_e32 v2, s27, v2
	.loc	1 88 5                          ; ./fmha_mfma_kernel.hpp:88:5
	s_or_b64 s[20:21], vcc, s[20:21]
	.loc	1 89 54 is_stmt 1               ; ./fmha_mfma_kernel.hpp:89:54
	s_waitcnt vmcnt(0)
	ds_write2_b64 v4, v[8:9], v[10:11] offset1:1
	.loc	1 88 28                         ; ./fmha_mfma_kernel.hpp:88:28
	v_add_u32_e32 v4, s9, v4
	.loc	1 88 5 is_stmt 0                ; ./fmha_mfma_kernel.hpp:88:5
	s_andn2_b64 exec, exec, s[20:21]
	s_cbranch_execnz .LBB0_2
; %bb.3:                                ; %._crit_edge
	.loc	1 0 5                           ; ./fmha_mfma_kernel.hpp:0:5
	s_or_b64 exec, exec, s[20:21]
	v_mad_u64_u32 v[2:3], s[0:1], v1, s25, v[14:15]
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[2:3], v[2:3], 1, s[18:19]
	.loc	1 94 19 is_stmt 1               ; ./fmha_mfma_kernel.hpp:94:19
	global_load_dwordx4 v[2:5], v[2:3], off
.LBB0_4:                                ; %Flow308
	.loc	1 0 19 is_stmt 0                ; ./fmha_mfma_kernel.hpp:0:19
	s_or_b64 exec, exec, s[22:23]
	v_add_u32_e32 v25, 0x2080, v6
	.loc	1 96 33 is_stmt 1               ; ./fmha_mfma_kernel.hpp:96:33
	v_add_u32_e32 v23, s2, v1
	v_mov_b32_e32 v6, 0
	v_add_u32_e32 v24, 0x2480, v7
	.loc	1 97 18                         ; ./fmha_mfma_kernel.hpp:97:18
	v_cmp_gt_i32_e64 s[0:1], s24, v23
	v_mov_b32_e32 v7, v6
	v_mov_b32_e32 v8, v6
	v_mov_b32_e32 v9, v6
	s_and_saveexec_b64 s[20:21], s[0:1]
	s_cbranch_execz .LBB0_6
; %bb.5:
	.loc	1 0 18 is_stmt 0                ; ./fmha_mfma_kernel.hpp:0:18
	v_mad_u64_u32 v[6:7], s[22:23], v23, s25, v[14:15]
	v_ashrrev_i32_e32 v7, 31, v6
	v_lshl_add_u64 v[6:7], v[6:7], 1, s[18:19]
	.loc	1 98 19 is_stmt 1               ; ./fmha_mfma_kernel.hpp:98:19
	global_load_dwordx4 v[6:9], v[6:7], off
.LBB0_6:
	.loc	1 0 19 is_stmt 0                ; ./fmha_mfma_kernel.hpp:0:19
	s_or_b64 exec, exec, s[20:21]
	.loc	1 61 50 is_stmt 1               ; ./fmha_mfma_kernel.hpp:61:50
	s_mul_hi_i32 s9, s11, s10
	s_mul_i32 s10, s11, s10
	.loc	1 61 64 is_stmt 0               ; ./fmha_mfma_kernel.hpp:61:64
	s_mul_i32 s5, s10, s5
	s_mul_hi_u32 s16, s10, s4
	.loc	1 61 52                         ; ./fmha_mfma_kernel.hpp:61:52
	s_ashr_i32 s2, s8, 31
	.loc	1 61 64                         ; ./fmha_mfma_kernel.hpp:61:64
	s_add_i32 s5, s16, s5
	s_mul_i32 s16, s9, s4
	s_mul_i32 s4, s10, s4
	s_add_i32 s5, s5, s16
	.loc	1 61 75                         ; ./fmha_mfma_kernel.hpp:61:75
	s_mul_i32 s2, s4, s2
	s_mul_hi_u32 s16, s4, s8
	s_mul_i32 s5, s5, s8
	s_add_i32 s2, s16, s2
	s_add_i32 s5, s2, s5
	.loc	1 62 40 is_stmt 1               ; ./fmha_mfma_kernel.hpp:62:40
	s_ashr_i32 s2, s3, 31
	.loc	1 61 75                         ; ./fmha_mfma_kernel.hpp:61:75
	s_mul_i32 s4, s4, s8
	.loc	1 62 60                         ; ./fmha_mfma_kernel.hpp:62:60
	s_mul_i32 s2, s10, s2
	s_mul_hi_u32 s8, s10, s3
	s_add_i32 s2, s8, s2
	s_mul_i32 s9, s9, s3
	.loc	1 44 34                         ; ./fmha_mfma_kernel.hpp:44:34
	v_bfe_u32 v22, v0, 4, 2
	.loc	1 45 34                         ; ./fmha_mfma_kernel.hpp:45:34
	v_and_b32_e32 v15, 15, v0
	.loc	1 62 60                         ; ./fmha_mfma_kernel.hpp:62:60
	s_add_i32 s9, s2, s9
	s_mul_i32 s8, s10, s3
	.loc	1 106 17                        ; ./fmha_mfma_kernel.hpp:106:17
	v_cmp_gt_u32_e32 vcc, 64, v0
.Ltmp4:
	.file	3 "/opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail" "amd_device_functions.h"
	.loc	3 691 5                         ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_device_functions.h:691:5 @[ /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_device_functions.h:707:63 @[ /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_device_functions.h:714:3 @[ ./fmha_mfma_kernel.hpp:101:5 ] ] ]
	s_waitcnt lgkmcnt(0)
	.loc	3 692 5                         ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_device_functions.h:692:5 @[ /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_device_functions.h:707:63 @[ /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_device_functions.h:714:3 @[ ./fmha_mfma_kernel.hpp:101:5 ] ] ]
	s_barrier
.Ltmp5:
	.loc	1 106 17                        ; ./fmha_mfma_kernel.hpp:106:17
	s_and_saveexec_b64 s[18:19], vcc
	s_cbranch_execz .LBB0_17
; %bb.7:
	.loc	1 109 27                        ; ./fmha_mfma_kernel.hpp:109:27
	s_cmp_lt_i32 s11, 1
	.loc	1 109 9 is_stmt 0               ; ./fmha_mfma_kernel.hpp:109:9
	s_cbranch_scc1 .LBB0_14
; %bb.8:                                ; %.lr.ph211
	.loc	1 0 9                           ; ./fmha_mfma_kernel.hpp:0:9
	v_lshrrev_b32_e32 v11, 4, v0
	.loc	1 107 39 is_stmt 1              ; ./fmha_mfma_kernel.hpp:107:39
	v_lshlrev_b32_e32 v10, 2, v11
	.loc	1 107 43 is_stmt 0              ; ./fmha_mfma_kernel.hpp:107:43
	v_mad_u64_u32 v[16:17], s[2:3], s11, v15, v[10:11]
	s_movk_i32 s2, 0x104
	.loc	1 108 43 is_stmt 1              ; ./fmha_mfma_kernel.hpp:108:43
	s_nop 0
	v_mad_u32_u24 v26, v15, s2, v10
	.loc	1 61 30                         ; ./fmha_mfma_kernel.hpp:61:30
	s_lshl_b64 s[2:3], s[4:5], 1
	s_add_u32 s10, s12, s2
	s_addc_u32 s12, s13, s3
	.loc	1 62 30                         ; ./fmha_mfma_kernel.hpp:62:30
	s_lshl_b64 s[2:3], s[8:9], 1
	.loc	1 108 54                        ; ./fmha_mfma_kernel.hpp:108:54
	v_mul_u32_u24_e32 v12, 0x104, v15
	.loc	1 62 30                         ; ./fmha_mfma_kernel.hpp:62:30
	s_add_u32 s2, s10, s2
	v_mov_b32_e32 v17, 0
	s_addc_u32 s3, s12, s3
	s_add_i32 s10, s11, 15
	.loc	1 109 9                         ; ./fmha_mfma_kernel.hpp:109:9
	v_lshlrev_b32_e32 v10, 1, v12
	v_mov_b32_e32 v18, v17
	v_mov_b32_e32 v19, v17
	s_lshr_b32 s10, s10, 4
	s_mul_i32 s16, s24, s25
	v_lshl_add_u32 v27, v11, 3, v10
	v_mov_b64_e32 v[20:21], v[18:19]
	v_mov_b32_e32 v10, v17
	v_mov_b32_e32 v11, v17
	v_mov_b32_e32 v12, v17
	v_mov_b32_e32 v13, v17
	s_branch .LBB0_10
.LBB0_9:                                ;   in Loop: Header=BB0_10 Depth=1
	.loc	1 0 9 is_stmt 0                 ; ./fmha_mfma_kernel.hpp:0:9
	s_or_b64 exec, exec, s[12:13]
	.loc	1 118 19 is_stmt 1              ; ./fmha_mfma_kernel.hpp:118:19
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_mfma_f32_16x16x16_bf16 v[10:13], v[18:19], v[20:21], v[10:13]
	.loc	1 109 27                        ; ./fmha_mfma_kernel.hpp:109:27
	s_add_i32 s10, s10, -1
	v_add_u32_e32 v27, 32, v27
	v_add_u32_e32 v26, 16, v26
	s_cmp_eq_u32 s10, 0
	v_add_u32_e32 v16, 16, v16
	.loc	1 109 9 is_stmt 0               ; ./fmha_mfma_kernel.hpp:109:9
	s_cbranch_scc1 .LBB0_15
.LBB0_10:                               ; =>This Inner Loop Header: Depth=1
	.loc	1 111 35 is_stmt 1              ; ./fmha_mfma_kernel.hpp:111:35
	v_cmp_gt_u32_e32 vcc, s11, v16
	s_and_saveexec_b64 s[12:13], vcc
	s_cbranch_execz .LBB0_12
; %bb.11:                               ;   in Loop: Header=BB0_10 Depth=1
	.loc	1 0 35 is_stmt 0                ; ./fmha_mfma_kernel.hpp:0:35
	v_lshl_add_u64 v[18:19], v[16:17], 1, s[2:3]
	.loc	1 112 21 is_stmt 1              ; ./fmha_mfma_kernel.hpp:112:21
	global_load_dwordx2 v[18:19], v[18:19], off
.LBB0_12:                               ;   in Loop: Header=BB0_10 Depth=1
	.loc	1 0 21 is_stmt 0                ; ./fmha_mfma_kernel.hpp:0:21
	s_or_b64 exec, exec, s[12:13]
	.loc	1 114 35 is_stmt 1              ; ./fmha_mfma_kernel.hpp:114:35
	v_cmp_gt_u32_e32 vcc, s16, v26
	s_and_saveexec_b64 s[12:13], vcc
	s_cbranch_execz .LBB0_9
; %bb.13:                               ;   in Loop: Header=BB0_10 Depth=1
	.loc	1 115 21                        ; ./fmha_mfma_kernel.hpp:115:21
	ds_read_b64 v[20:21], v27
	s_branch .LBB0_9
.LBB0_14:
	.loc	1 0 21 is_stmt 0                ; ./fmha_mfma_kernel.hpp:0:21
	v_mov_b32_e32 v10, 0
.LBB0_15:                               ; %._crit_edge212
	.loc	1 121 22 is_stmt 1              ; ./fmha_mfma_kernel.hpp:121:22
	v_cmp_eq_u32_e32 vcc, 0, v22
	v_cmp_gt_i32_e64 s[2:3], s24, v15
	.loc	1 121 27 is_stmt 0              ; ./fmha_mfma_kernel.hpp:121:27
	s_and_b64 s[2:3], vcc, s[2:3]
	s_and_b64 exec, exec, s[2:3]
; %bb.16:
	.loc	1 0 27                          ; ./fmha_mfma_kernel.hpp:0:27
	v_lshlrev_b32_e32 v11, 2, v15
	.loc	1 122 30 is_stmt 1              ; ./fmha_mfma_kernel.hpp:122:30
	ds_write_b32 v11, v10 offset:8320
.LBB0_17:                               ; %Flow307
	.loc	1 0 30 is_stmt 0                ; ./fmha_mfma_kernel.hpp:0:30
	s_or_b64 exec, exec, s[18:19]
	.loc	1 126 13 is_stmt 1              ; ./fmha_mfma_kernel.hpp:126:13
	v_cmp_gt_i32_e32 vcc, s24, v0
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB0_19
; %bb.18:
	.loc	1 127 21                        ; ./fmha_mfma_kernel.hpp:127:21
	ds_read_b32 v10, v25
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v10, s17, v10
	ds_write_b32 v25, v10
.LBB0_19:
	.loc	1 0 21 is_stmt 0                ; ./fmha_mfma_kernel.hpp:0:21
	s_or_b64 exec, exec, s[2:3]
	v_mov_b32_e32 v11, 0xff800000
.Ltmp6:
	.loc	3 691 5 is_stmt 1               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_device_functions.h:691:5 @[ /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_device_functions.h:707:63 @[ /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_device_functions.h:714:3 @[ ./fmha_mfma_kernel.hpp:130:5 ] ] ]
	s_waitcnt lgkmcnt(0)
	.loc	3 692 5                         ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_device_functions.h:692:5 @[ /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_device_functions.h:707:63 @[ /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_device_functions.h:714:3 @[ ./fmha_mfma_kernel.hpp:130:5 ] ] ]
	s_barrier
.Ltmp7:
	.loc	1 134 13                        ; ./fmha_mfma_kernel.hpp:134:13
	s_and_saveexec_b64 s[2:3], vcc
; %bb.20:
	.loc	1 135 18                        ; ./fmha_mfma_kernel.hpp:135:18
	ds_read_b32 v11, v25
; %bb.21:
	.loc	1 0 18 is_stmt 0                ; ./fmha_mfma_kernel.hpp:0:18
	s_or_b64 exec, exec, s[2:3]
	v_mbcnt_lo_u32_b32 v10, -1, 0
	v_mbcnt_hi_u32_b32 v13, -1, v10
	v_and_b32_e32 v10, 0x70, v13
	v_add_u32_e32 v17, 16, v10
.Ltmp8:
	.file	4 "/opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail" "amd_warp_functions.h"
	.loc	4 493 20 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_warp_functions.h:493:20 @[ /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_warp_functions.h:515:11 @[ ./fmha_mfma_kernel.hpp:140:32 ] ]
	v_xor_b32_e32 v10, 8, v13
	.loc	4 494 17                        ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_warp_functions.h:494:17 @[ /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_warp_functions.h:515:11 @[ ./fmha_mfma_kernel.hpp:140:32 ] ]
	v_cmp_lt_i32_e64 s[2:3], v10, v17
	.loc	4 493 20                        ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_warp_functions.h:493:20 @[ /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_warp_functions.h:515:11 @[ ./fmha_mfma_kernel.hpp:140:32 ] ]
	v_xor_b32_e32 v16, 4, v13
	v_xor_b32_e32 v18, 2, v13
	.loc	4 494 11                        ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_warp_functions.h:494:11 @[ /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_warp_functions.h:515:11 @[ ./fmha_mfma_kernel.hpp:140:32 ] ]
	v_cndmask_b32_e64 v10, v13, v10, s[2:3]
	.loc	4 495 45                        ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_warp_functions.h:495:45 @[ /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_warp_functions.h:515:11 @[ ./fmha_mfma_kernel.hpp:140:32 ] ]
	v_lshlrev_b32_e32 v10, 2, v10
	.loc	4 495 10 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_warp_functions.h:495:10 @[ /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_warp_functions.h:515:11 @[ ./fmha_mfma_kernel.hpp:140:32 ] ]
	s_waitcnt lgkmcnt(0)
	ds_bpermute_b32 v12, v10, v11
.Ltmp9:
	.file	5 "/opt/rocm-7.2.0/lib/llvm/lib/clang/22/include" "__clang_hip_math.h"
	.loc	5 454 44 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/lib/clang/22/include/__clang_hip_math.h:454:44 @[ ./fmha_mfma_kernel.hpp:140:18 ]
	v_max_f32_e32 v11, v11, v11
.Ltmp10:
	.loc	4 494 17                        ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_warp_functions.h:494:17 @[ /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_warp_functions.h:515:11 @[ ./fmha_mfma_kernel.hpp:140:32 ] ]
	v_cmp_lt_i32_e64 s[2:3], v16, v17
	.loc	4 493 20                        ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_warp_functions.h:493:20 @[ /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_warp_functions.h:515:11 @[ ./fmha_mfma_kernel.hpp:140:32 ] ]
	v_xor_b32_e32 v19, 1, v13
.Ltmp11:
	.loc	5 454 44                        ; /opt/rocm-7.2.0/lib/llvm/lib/clang/22/include/__clang_hip_math.h:454:44 @[ ./fmha_mfma_kernel.hpp:140:18 ]
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v12, v12, v12
	v_max_f32_e32 v12, v11, v12
.Ltmp12:
	.loc	4 494 11                        ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_warp_functions.h:494:11 @[ /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_warp_functions.h:515:11 @[ ./fmha_mfma_kernel.hpp:140:32 ] ]
	v_cndmask_b32_e64 v11, v13, v16, s[2:3]
	.loc	4 495 45                        ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_warp_functions.h:495:45 @[ /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_warp_functions.h:515:11 @[ ./fmha_mfma_kernel.hpp:140:32 ] ]
	v_lshlrev_b32_e32 v11, 2, v11
	.loc	4 495 10 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_warp_functions.h:495:10 @[ /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_warp_functions.h:515:11 @[ ./fmha_mfma_kernel.hpp:140:32 ] ]
	ds_bpermute_b32 v16, v11, v12
	.loc	4 494 17 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_warp_functions.h:494:17 @[ /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_warp_functions.h:515:11 @[ ./fmha_mfma_kernel.hpp:140:32 ] ]
	v_cmp_lt_i32_e64 s[2:3], v18, v17
.Ltmp13:
	.loc	5 454 44                        ; /opt/rocm-7.2.0/lib/llvm/lib/clang/22/include/__clang_hip_math.h:454:44 @[ ./fmha_mfma_kernel.hpp:140:18 ]
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v16, v16, v16
	v_max_f32_e32 v16, v12, v16
.Ltmp14:
	.loc	4 494 11                        ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_warp_functions.h:494:11 @[ /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_warp_functions.h:515:11 @[ ./fmha_mfma_kernel.hpp:140:32 ] ]
	v_cndmask_b32_e64 v12, v13, v18, s[2:3]
	.loc	4 495 45                        ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_warp_functions.h:495:45 @[ /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_warp_functions.h:515:11 @[ ./fmha_mfma_kernel.hpp:140:32 ] ]
	v_lshlrev_b32_e32 v12, 2, v12
	.loc	4 495 10 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_warp_functions.h:495:10 @[ /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_warp_functions.h:515:11 @[ ./fmha_mfma_kernel.hpp:140:32 ] ]
	ds_bpermute_b32 v18, v12, v16
	.loc	4 494 17 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_warp_functions.h:494:17 @[ /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_warp_functions.h:515:11 @[ ./fmha_mfma_kernel.hpp:140:32 ] ]
	v_cmp_lt_i32_e64 s[2:3], v19, v17
	v_mov_b32_e32 v17, 0
.Ltmp15:
	.loc	5 454 44                        ; /opt/rocm-7.2.0/lib/llvm/lib/clang/22/include/__clang_hip_math.h:454:44 @[ ./fmha_mfma_kernel.hpp:140:18 ]
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v18, v18, v18
.Ltmp16:
	.loc	4 494 11                        ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_warp_functions.h:494:11 @[ /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_warp_functions.h:515:11 @[ ./fmha_mfma_kernel.hpp:140:32 ] ]
	v_cndmask_b32_e64 v13, v13, v19, s[2:3]
.Ltmp17:
	.loc	5 454 44                        ; /opt/rocm-7.2.0/lib/llvm/lib/clang/22/include/__clang_hip_math.h:454:44 @[ ./fmha_mfma_kernel.hpp:140:18 ]
	v_max_f32_e32 v16, v16, v18
.Ltmp18:
	.loc	4 495 45                        ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_warp_functions.h:495:45 @[ /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_warp_functions.h:515:11 @[ ./fmha_mfma_kernel.hpp:140:32 ] ]
	v_lshlrev_b32_e32 v13, 2, v13
	.loc	4 495 10 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_warp_functions.h:495:10 @[ /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_warp_functions.h:515:11 @[ ./fmha_mfma_kernel.hpp:140:32 ] ]
	ds_bpermute_b32 v18, v13, v16
.Ltmp19:
	.loc	1 145 13 is_stmt 1              ; ./fmha_mfma_kernel.hpp:145:13
	s_and_saveexec_b64 s[12:13], vcc
	s_cbranch_execz .LBB0_23
; %bb.22:
	.loc	1 146 28                        ; ./fmha_mfma_kernel.hpp:146:28
	ds_read_b32 v17, v25
.Ltmp20:
	.loc	5 454 44                        ; /opt/rocm-7.2.0/lib/llvm/lib/clang/22/include/__clang_hip_math.h:454:44 @[ ./fmha_mfma_kernel.hpp:140:18 ]
	s_waitcnt lgkmcnt(1)
	v_max_f32_e32 v18, v18, v18
	v_max_f32_e32 v16, v16, v16
	v_max_f32_e32 v16, v16, v18
	s_mov_b32 s2, 0x3fb8aa3b
.Ltmp21:
	.loc	1 146 40                        ; ./fmha_mfma_kernel.hpp:146:40
	s_waitcnt lgkmcnt(0)
	v_sub_f32_e32 v16, v17, v16
.Ltmp22:
	.loc	5 431 32                        ; /opt/rocm-7.2.0/lib/llvm/lib/clang/22/include/__clang_hip_math.h:431:32 @[ ./fmha_mfma_kernel.hpp:146:23 ]
	v_mul_f32_e32 v17, 0x3fb8aa3b, v16
	v_fma_f32 v18, v16, s2, -v17
	v_rndne_f32_e32 v19, v17
	v_fmamk_f32 v18, v16, 0x32a5705f, v18
	v_sub_f32_e32 v17, v17, v19
	v_add_f32_e32 v17, v17, v18
	v_cvt_i32_f32_e32 v18, v19
	v_exp_f32_e32 v17, v17
	s_mov_b32 s2, 0xc2ce8ed0
	v_cmp_ngt_f32_e64 s[2:3], s2, v16
	v_ldexp_f32 v17, v17, v18
	s_nop 0
	v_cndmask_b32_e64 v17, 0, v17, s[2:3]
	s_mov_b32 s2, 0x42b17218
	v_mov_b32_e32 v18, 0x7f800000
	v_cmp_nlt_f32_e64 s[2:3], s2, v16
	s_nop 1
	v_cndmask_b32_e64 v17, v18, v17, s[2:3]
.Ltmp23:
	.loc	1 146 21                        ; ./fmha_mfma_kernel.hpp:146:21
	ds_write_b32 v25, v17
.LBB0_23:
	.loc	1 0 21 is_stmt 0                ; ./fmha_mfma_kernel.hpp:0:21
	s_or_b64 exec, exec, s[12:13]
.Ltmp24:
	.loc	4 495 10 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_warp_functions.h:495:10 @[ /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_warp_functions.h:515:11 @[ ./fmha_mfma_kernel.hpp:152:19 ] ]
	ds_bpermute_b32 v10, v10, v17
.Ltmp25:
	.loc	1 152 16                        ; ./fmha_mfma_kernel.hpp:152:16
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v10, v17, v10
.Ltmp26:
	.loc	4 495 10                        ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_warp_functions.h:495:10 @[ /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_warp_functions.h:515:11 @[ ./fmha_mfma_kernel.hpp:152:19 ] ]
	ds_bpermute_b32 v11, v11, v10
.Ltmp27:
	.loc	1 152 16                        ; ./fmha_mfma_kernel.hpp:152:16
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v10, v10, v11
.Ltmp28:
	.loc	4 495 10                        ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_warp_functions.h:495:10 @[ /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_warp_functions.h:515:11 @[ ./fmha_mfma_kernel.hpp:152:19 ] ]
	ds_bpermute_b32 v11, v12, v10
.Ltmp29:
	.loc	1 152 16                        ; ./fmha_mfma_kernel.hpp:152:16
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v10, v10, v11
.Ltmp30:
	.loc	4 495 10                        ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_warp_functions.h:495:10 @[ /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_warp_functions.h:515:11 @[ ./fmha_mfma_kernel.hpp:152:19 ] ]
	ds_bpermute_b32 v11, v13, v10
.Ltmp31:
	.loc	1 155 13                        ; ./fmha_mfma_kernel.hpp:155:13
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB0_25
; %bb.24:
	.loc	1 156 21                        ; ./fmha_mfma_kernel.hpp:156:21
	ds_read_b32 v12, v25
	.loc	1 152 16                        ; ./fmha_mfma_kernel.hpp:152:16
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v10, v10, v11
	s_movk_i32 s10, 0x7fff
	.loc	1 156 21                        ; ./fmha_mfma_kernel.hpp:156:21
	s_waitcnt lgkmcnt(0)
	v_div_scale_f32 v11, s[12:13], v10, v10, v12
	v_rcp_f32_e32 v13, v11
	v_div_scale_f32 v16, vcc, v12, v10, v12
	v_fma_f32 v17, -v11, v13, 1.0
	v_fmac_f32_e32 v13, v17, v13
	v_mul_f32_e32 v17, v16, v13
	v_fma_f32 v18, -v11, v17, v16
	v_fmac_f32_e32 v17, v18, v13
	v_fma_f32 v11, -v11, v17, v16
	v_div_fmas_f32 v11, v11, v13, v17
	v_div_fixup_f32 v10, v11, v10, v12
	.loc	1 157 51                        ; ./fmha_mfma_kernel.hpp:157:51
	v_bfe_u32 v11, v10, 16, 1
	v_add3_u32 v11, v11, v10, s10
	v_or_b32_e32 v12, 0x400000, v10
	v_cmp_u_f32_e32 vcc, v10, v10
	.loc	1 156 21                        ; ./fmha_mfma_kernel.hpp:156:21
	ds_write_b32 v25, v10
	.loc	1 157 51                        ; ./fmha_mfma_kernel.hpp:157:51
	s_nop 0
	v_cndmask_b32_e32 v10, v11, v12, vcc
	.loc	1 157 29 is_stmt 0              ; ./fmha_mfma_kernel.hpp:157:29
	ds_write_b16_d16_hi v24, v10
.LBB0_25:
	.loc	1 0 29                          ; ./fmha_mfma_kernel.hpp:0:29
	s_or_b64 exec, exec, s[2:3]
	.loc	1 159 13 is_stmt 1              ; ./fmha_mfma_kernel.hpp:159:13
	v_cmp_le_i32_e32 vcc, s24, v0
	v_cmp_gt_u32_e64 s[2:3], 16, v0
	.loc	1 159 26 is_stmt 0              ; ./fmha_mfma_kernel.hpp:159:26
	s_and_b64 s[12:13], vcc, s[2:3]
	s_and_saveexec_b64 s[2:3], s[12:13]
; %bb.26:
	.loc	1 0 26                          ; ./fmha_mfma_kernel.hpp:0:26
	v_mov_b32_e32 v10, 0
	.loc	1 160 29 is_stmt 1              ; ./fmha_mfma_kernel.hpp:160:29
	ds_write_b16 v24, v10
; %bb.27:
	.loc	1 0 29 is_stmt 0                ; ./fmha_mfma_kernel.hpp:0:29
	s_or_b64 exec, exec, s[2:3]
.Ltmp32:
	.loc	3 691 5 is_stmt 1               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_device_functions.h:691:5 @[ /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_device_functions.h:707:63 @[ /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_device_functions.h:714:3 @[ ./fmha_mfma_kernel.hpp:162:5 ] ] ]
	s_waitcnt lgkmcnt(0)
	.loc	3 692 5                         ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_device_functions.h:692:5 @[ /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_device_functions.h:707:63 @[ /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_device_functions.h:714:3 @[ ./fmha_mfma_kernel.hpp:162:5 ] ] ]
	s_barrier
.Ltmp33:
	.loc	1 164 16                        ; ./fmha_mfma_kernel.hpp:164:16
	s_and_saveexec_b64 s[2:3], s[6:7]
	s_cbranch_execz .LBB0_29
; %bb.28:
	.loc	1 0 16 is_stmt 0                ; ./fmha_mfma_kernel.hpp:0:16
	s_movk_i32 s6, 0x208
	v_mul_lo_u32 v1, v1, s6
	v_lshl_add_u32 v1, v14, 1, v1
	.loc	1 165 59 is_stmt 1              ; ./fmha_mfma_kernel.hpp:165:59
	s_waitcnt vmcnt(0)
	ds_write2_b64 v1, v[2:3], v[4:5] offset1:1
.LBB0_29:
	.loc	1 0 59 is_stmt 0                ; ./fmha_mfma_kernel.hpp:0:59
	s_or_b64 exec, exec, s[2:3]
	.loc	1 167 18 is_stmt 1              ; ./fmha_mfma_kernel.hpp:167:18
	s_and_saveexec_b64 s[2:3], s[0:1]
	s_cbranch_execz .LBB0_31
; %bb.30:
	.loc	1 0 18 is_stmt 0                ; ./fmha_mfma_kernel.hpp:0:18
	s_movk_i32 s0, 0x208
	v_mul_lo_u32 v1, v23, s0
	v_lshl_add_u32 v1, v14, 1, v1
	.loc	1 168 61 is_stmt 1              ; ./fmha_mfma_kernel.hpp:168:61
	s_waitcnt vmcnt(0)
	ds_write2_b64 v1, v[6:7], v[8:9] offset1:1
.LBB0_31:
	.loc	1 0 61 is_stmt 0                ; ./fmha_mfma_kernel.hpp:0:61
	s_or_b64 exec, exec, s[2:3]
	s_add_i32 s0, s11, 63
	.loc	1 179 23 is_stmt 1              ; ./fmha_mfma_kernel.hpp:179:23
	s_cmp_lt_u32 s0, 64
.Ltmp34:
	.loc	3 691 5                         ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_device_functions.h:691:5 @[ /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_device_functions.h:707:63 @[ /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_device_functions.h:714:3 @[ ./fmha_mfma_kernel.hpp:170:5 ] ] ]
	s_waitcnt lgkmcnt(0)
	.loc	3 692 5                         ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_device_functions.h:692:5 @[ /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_device_functions.h:707:63 @[ /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_device_functions.h:714:3 @[ ./fmha_mfma_kernel.hpp:170:5 ] ] ]
	s_barrier
	s_cbranch_scc1 .LBB0_44
.Ltmp35:
; %bb.32:                               ; %.lr.ph225
	.loc	1 177 21                        ; ./fmha_mfma_kernel.hpp:177:21
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v2, 3, v22
	v_lshl_or_b32 v2, v15, 5, v2
	s_lshr_b32 s12, s0, 6
	.loc	1 65 23                         ; ./fmha_mfma_kernel.hpp:65:23
	s_lshl_b64 s[0:1], s[4:5], 1
	.loc	1 177 9                         ; ./fmha_mfma_kernel.hpp:177:9
	ds_read_b64 v[4:5], v2 offset:9344
	.loc	1 65 23                         ; ./fmha_mfma_kernel.hpp:65:23
	s_add_u32 s2, s14, s0
	.loc	1 173 35                        ; ./fmha_mfma_kernel.hpp:173:35
	v_lshlrev_b32_e32 v1, 2, v22
	.loc	1 65 23                         ; ./fmha_mfma_kernel.hpp:65:23
	s_addc_u32 s3, s15, s1
	.loc	1 66 23                         ; ./fmha_mfma_kernel.hpp:66:23
	s_lshl_b64 s[0:1], s[8:9], 1
	v_lshrrev_b32_e32 v2, 2, v0
	.loc	1 179 5                         ; ./fmha_mfma_kernel.hpp:179:5
	v_lshrrev_b32_e32 v0, 1, v0
	.loc	1 66 23                         ; ./fmha_mfma_kernel.hpp:66:23
	s_add_u32 s8, s2, s0
	v_or_b32_e32 v3, 1, v1
	.loc	1 179 5                         ; ./fmha_mfma_kernel.hpp:179:5
	v_and_b32_e32 v0, 0x60, v0
	s_movk_i32 s10, 0x820
	.loc	1 66 23                         ; ./fmha_mfma_kernel.hpp:66:23
	s_addc_u32 s9, s3, s1
	v_cmp_gt_i32_e64 s[0:1], s24, v1
	v_cmp_gt_i32_e64 s[2:3], s24, v3
	v_or_b32_e32 v3, 2, v1
	v_or_b32_e32 v1, 3, v1
	.loc	1 179 5                         ; ./fmha_mfma_kernel.hpp:179:5
	v_mad_u32_u24 v0, v22, s10, v0
	v_cmp_eq_u32_e32 vcc, 0, v22
	v_cmp_gt_i32_e64 s[4:5], s24, v3
	v_cmp_gt_i32_e64 s[6:7], s24, v1
	v_lshl_or_b32 v10, v15, 1, v0
	v_and_or_b32 v6, v2, 48, v15
	v_mov_b32_e32 v8, 0
	s_mov_b32 s13, 0x5040100
	s_mov_b32 s14, 0xffff
	v_mov_b32_e32 v11, 0x5040100
	s_branch .LBB0_34
.LBB0_33:                               ;   in Loop: Header=BB0_34 Depth=1
	.loc	1 0 5 is_stmt 0                 ; ./fmha_mfma_kernel.hpp:0:5
	s_or_b64 exec, exec, s[10:11]
	.loc	1 179 23 is_stmt 1              ; ./fmha_mfma_kernel.hpp:179:23
	s_add_i32 s12, s12, -1
	v_add_u32_e32 v10, 0x80, v10
	s_cmp_eq_u32 s12, 0
	v_add_u32_e32 v6, 64, v6
	s_cbranch_scc1 .LBB0_44
.LBB0_34:                               ; =>This Inner Loop Header: Depth=1
	.loc	1 0 23 is_stmt 0                ; ./fmha_mfma_kernel.hpp:0:23
	v_mov_b32_e32 v9, v8
	v_mov_b64_e32 v[0:1], v[8:9]
	.loc	1 186 34 is_stmt 1              ; ./fmha_mfma_kernel.hpp:186:34
	s_and_saveexec_b64 s[10:11], s[0:1]
	s_cbranch_execnz .LBB0_41
; %bb.35:                               ;   in Loop: Header=BB0_34 Depth=1
	.loc	1 0 34 is_stmt 0                ; ./fmha_mfma_kernel.hpp:0:34
	s_or_b64 exec, exec, s[10:11]
	.loc	1 186 34                        ; ./fmha_mfma_kernel.hpp:186:34
	s_and_saveexec_b64 s[10:11], s[2:3]
	s_cbranch_execnz .LBB0_42
.LBB0_36:                               ;   in Loop: Header=BB0_34 Depth=1
	.loc	1 0 34                          ; ./fmha_mfma_kernel.hpp:0:34
	s_or_b64 exec, exec, s[10:11]
	.loc	1 186 34                        ; ./fmha_mfma_kernel.hpp:186:34
	s_and_saveexec_b64 s[10:11], s[4:5]
	s_cbranch_execnz .LBB0_43
.LBB0_37:                               ;   in Loop: Header=BB0_34 Depth=1
	.loc	1 0 34                          ; ./fmha_mfma_kernel.hpp:0:34
	s_or_b64 exec, exec, s[10:11]
	.loc	1 186 34                        ; ./fmha_mfma_kernel.hpp:186:34
	s_and_saveexec_b64 s[10:11], s[6:7]
	s_cbranch_execz .LBB0_39
.LBB0_38:                               ;   in Loop: Header=BB0_34 Depth=1
	.loc	1 187 24 is_stmt 1              ; ./fmha_mfma_kernel.hpp:187:24
	ds_read_u16 v2, v10 offset:1560
	.loc	1 187 22 is_stmt 0              ; ./fmha_mfma_kernel.hpp:187:22
	s_waitcnt lgkmcnt(0)
	v_perm_b32 v1, v2, v1, s13
.LBB0_39:                               ;   in Loop: Header=BB0_34 Depth=1
	.loc	1 0 22                          ; ./fmha_mfma_kernel.hpp:0:22
	s_or_b64 exec, exec, s[10:11]
	.loc	1 190 15 is_stmt 1              ; ./fmha_mfma_kernel.hpp:190:15
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x16_bf16 v[0:3], v[4:5], v[0:1], 0
	.loc	1 193 22                        ; ./fmha_mfma_kernel.hpp:193:22
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB0_33
; %bb.40:                               ;   in Loop: Header=BB0_34 Depth=1
	.loc	1 0 22 is_stmt 0                ; ./fmha_mfma_kernel.hpp:0:22
	s_nop 4
	v_cvt_f16_f32_e32 v2, v0
	v_mov_b32_e32 v7, v8
	v_lshl_add_u64 v[0:1], v[6:7], 1, s[8:9]
	.loc	1 194 39 is_stmt 1              ; ./fmha_mfma_kernel.hpp:194:39
	global_store_short v[0:1], v2, off
	s_branch .LBB0_33
.LBB0_41:                               ;   in Loop: Header=BB0_34 Depth=1
	.loc	1 187 24                        ; ./fmha_mfma_kernel.hpp:187:24
	ds_read_u16 v0, v10
	.loc	1 187 22 is_stmt 0              ; ./fmha_mfma_kernel.hpp:187:22
	v_mov_b32_e32 v1, v8
	s_waitcnt lgkmcnt(0)
	v_perm_b32 v0, 0, v0, v11
	s_or_b64 exec, exec, s[10:11]
	.loc	1 186 34 is_stmt 1              ; ./fmha_mfma_kernel.hpp:186:34
	s_and_saveexec_b64 s[10:11], s[2:3]
	s_cbranch_execz .LBB0_36
.LBB0_42:                               ;   in Loop: Header=BB0_34 Depth=1
	.loc	1 187 24                        ; ./fmha_mfma_kernel.hpp:187:24
	ds_read_u16 v2, v10 offset:520
	.loc	1 187 22 is_stmt 0              ; ./fmha_mfma_kernel.hpp:187:22
	s_waitcnt lgkmcnt(0)
	v_perm_b32 v0, v2, v0, s13
	s_or_b64 exec, exec, s[10:11]
	.loc	1 186 34 is_stmt 1              ; ./fmha_mfma_kernel.hpp:186:34
	s_and_saveexec_b64 s[10:11], s[4:5]
	s_cbranch_execz .LBB0_37
.LBB0_43:                               ;   in Loop: Header=BB0_34 Depth=1
	.loc	1 187 24                        ; ./fmha_mfma_kernel.hpp:187:24
	ds_read_u16 v2, v10 offset:1040
	.loc	1 187 22 is_stmt 0              ; ./fmha_mfma_kernel.hpp:187:22
	s_waitcnt lgkmcnt(0)
	v_bfi_b32 v1, s14, v2, v1
	s_or_b64 exec, exec, s[10:11]
	.loc	1 186 34 is_stmt 1              ; ./fmha_mfma_kernel.hpp:186:34
	s_and_saveexec_b64 s[10:11], s[6:7]
	s_cbranch_execnz .LBB0_38
	s_branch .LBB0_39
.LBB0_44:                               ; %._crit_edge226
	.loc	1 198 1                         ; ./fmha_mfma_kernel.hpp:198:1
	s_endpgm
.Ltmp36:
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z9fmha_mfmaPKDF16bS0_S0_PDF16_PKiiiiiiif
		.amdhsa_group_segment_fixed_size 9856
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 68
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 0
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 1
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 73
		.amdhsa_next_free_sgpr 91
		.amdhsa_accum_offset 28
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_tg_split 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end0:
	.size	_Z9fmha_mfmaPKDF16bS0_S0_PDF16_PKiiiiiiif, .Lfunc_end0-_Z9fmha_mfmaPKDF16bS0_S0_PDF16_PKiiiiiiif
	.cfi_endproc
                                        ; -- End function
	.set _Z9fmha_mfmaPKDF16bS0_S0_PDF16_PKiiiiiiif.num_vgpr, 28
	.set _Z9fmha_mfmaPKDF16bS0_S0_PDF16_PKiiiiiiif.num_agpr, 0
	.set _Z9fmha_mfmaPKDF16bS0_S0_PDF16_PKiiiiiiif.numbered_sgpr, 29
	.set _Z9fmha_mfmaPKDF16bS0_S0_PDF16_PKiiiiiiif.num_named_barrier, 0
	.set _Z9fmha_mfmaPKDF16bS0_S0_PDF16_PKiiiiiiif.private_seg_size, 0
	.set _Z9fmha_mfmaPKDF16bS0_S0_PDF16_PKiiiiiiif.uses_vcc, 1
	.set _Z9fmha_mfmaPKDF16bS0_S0_PDF16_PKiiiiiiif.uses_flat_scratch, 0
	.set _Z9fmha_mfmaPKDF16bS0_S0_PDF16_PKiiiiiiif.has_dyn_sized_stack, 0
	.set _Z9fmha_mfmaPKDF16bS0_S0_PDF16_PKiiiiiiif.has_recursion, 0
	.set _Z9fmha_mfmaPKDF16bS0_S0_PDF16_PKiiiiiiif.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 2328
; TotalNumSgprs: 35
; NumVgprs: 28
; NumAgprs: 0
; TotalNumVgprs: 28
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 9856 bytes/workgroup (compile time only)
; SGPRBlocks: 12
; VGPRBlocks: 9
; NumSGPRsForWavesPerEU: 97
; NumVGPRsForWavesPerEU: 73
; AccumOffset: 28
; Occupancy: 6
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 1
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 6
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.protected	_Z19fmha_mfma_coalescedPKDF16bS0_S0_PDF16_PKiiiiiiif ; -- Begin function _Z19fmha_mfma_coalescedPKDF16bS0_S0_PDF16_PKiiiiiiif
	.globl	_Z19fmha_mfma_coalescedPKDF16bS0_S0_PDF16_PKiiiiiiif
	.p2align	8
	.type	_Z19fmha_mfma_coalescedPKDF16bS0_S0_PDF16_PKiiiiiiif,@function
_Z19fmha_mfma_coalescedPKDF16bS0_S0_PDF16_PKiiiiiiif: ; @_Z19fmha_mfma_coalescedPKDF16bS0_S0_PDF16_PKiiiiiiif
.Lfunc_begin1:
	.file	6 "." "fmha_mfma_coalesced_kernel.hpp"
	.loc	6 64 0                          ; ./fmha_mfma_coalesced_kernel.hpp:64:0
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; 
	.cfi_undefined 16
	s_mov_b32 s12, s3
	s_load_dwordx2 s[6:7], s[0:1], 0x20
	s_load_dwordx2 s[2:3], s[0:1], 0x10
.Ltmp37:
	.loc	6 82 30 prologue_end            ; ./fmha_mfma_coalesced_kernel.hpp:82:30
	s_ashr_i32 s5, s4, 31
	s_lshl_b64 s[8:9], s[4:5], 2
	.loc	6 114 13                        ; ./fmha_mfma_coalesced_kernel.hpp:114:13
	v_cmp_gt_u32_e32 vcc, 16, v0
	.loc	6 82 30                         ; ./fmha_mfma_coalesced_kernel.hpp:82:30
	s_waitcnt lgkmcnt(0)
	s_add_u32 s8, s6, s8
	s_addc_u32 s9, s7, s9
	s_load_dwordx2 s[6:7], s[8:9], 0x0
	.loc	6 114 13                        ; ./fmha_mfma_coalesced_kernel.hpp:114:13
	s_and_saveexec_b64 s[8:9], vcc
; %bb.1:
	.loc	6 0 13 is_stmt 0                ; ./fmha_mfma_coalesced_kernel.hpp:0:13
	v_lshlrev_b32_e32 v1, 1, v0
	v_mov_b32_e32 v2, 0
	.loc	6 115 26 is_stmt 1              ; ./fmha_mfma_coalesced_kernel.hpp:115:26
	ds_write_b16 v1, v2 offset:8320
; %bb.2:
	.loc	6 0 26 is_stmt 0                ; ./fmha_mfma_coalesced_kernel.hpp:0:26
	s_or_b64 exec, exec, s[8:9]
.Ltmp38:
	.loc	2 261 116 is_stmt 1             ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_runtime.h:261:116 @[ /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_runtime.h:289:160 @[ ./fmha_mfma_coalesced_kernel.hpp:68:27 ] ]
	s_load_dwordx4 s[8:11], s[0:1], 0x2c
	s_load_dwordx2 s[16:17], s[0:1], 0x3c
.Ltmp39:
	.loc	6 84 38                         ; ./fmha_mfma_coalesced_kernel.hpp:84:38
	s_waitcnt lgkmcnt(0)
	s_sub_i32 s24, s7, s6
	.loc	6 86 38                         ; ./fmha_mfma_coalesced_kernel.hpp:86:38
	s_ashr_i32 s7, s6, 31
	v_mov_b32_e32 v5, 0
	v_mov_b32_e32 v4, 0
	.loc	6 86 51 is_stmt 0               ; ./fmha_mfma_coalesced_kernel.hpp:86:51
	s_mul_hi_i32 s13, s16, s9
	s_mul_i32 s9, s16, s9
	.loc	6 86 66                         ; ./fmha_mfma_coalesced_kernel.hpp:86:66
	s_mul_i32 s7, s9, s7
	s_mul_hi_u32 s14, s9, s6
	s_add_i32 s7, s14, s7
	s_mul_i32 s13, s13, s6
	s_add_i32 s7, s7, s13
	.loc	6 87 39 is_stmt 1               ; ./fmha_mfma_coalesced_kernel.hpp:87:39
	s_mul_i32 s13, s16, s12
	.loc	6 86 66                         ; ./fmha_mfma_coalesced_kernel.hpp:86:66
	s_mul_i32 s6, s9, s6
	.loc	6 87 30                         ; ./fmha_mfma_coalesced_kernel.hpp:87:30
	s_ashr_i32 s14, s13, 31
	.loc	6 87 28 is_stmt 0               ; ./fmha_mfma_coalesced_kernel.hpp:87:28
	s_add_u32 s6, s6, s13
	s_addc_u32 s7, s7, s14
	.loc	6 92 77 is_stmt 1               ; ./fmha_mfma_coalesced_kernel.hpp:92:77
	s_ashr_i32 s25, s11, 31
	.loc	6 95 30                         ; ./fmha_mfma_coalesced_kernel.hpp:95:30
	s_lshl_b64 s[22:23], s[6:7], 1
	s_add_u32 s18, s2, s22
	s_addc_u32 s19, s3, s23
	.loc	6 130 32                        ; ./fmha_mfma_coalesced_kernel.hpp:130:32
	s_lshr_b32 s2, s25, 29
	s_add_i32 s2, s11, s2
	s_ashr_i32 s26, s2, 3
	.loc	6 131 25                        ; ./fmha_mfma_coalesced_kernel.hpp:131:25
	s_abs_i32 s3, s26
	v_cvt_f32_u32_e32 v1, s3
	s_sub_i32 s6, 0, s3
	.loc	6 100 39                        ; ./fmha_mfma_coalesced_kernel.hpp:100:39
	s_add_i32 s16, s11, 4
	.loc	6 131 25                        ; ./fmha_mfma_coalesced_kernel.hpp:131:25
	s_ashr_i32 s2, s2, 31
	v_rcp_iflag_f32_e32 v1, v1
	s_load_dwordx2 s[20:21], s[0:1], 0x0
	s_load_dwordx2 s[14:15], s[0:1], 0x18
	v_mul_f32_e32 v1, 0x4f7ffffe, v1
	v_cvt_u32_f32_e32 v1, v1
	s_nop 0
	v_readfirstlane_b32 s7, v1
	s_mul_i32 s6, s6, s7
	s_mul_hi_u32 s6, s7, s6
	s_add_i32 s7, s7, s6
	.loc	6 132 28                        ; ./fmha_mfma_coalesced_kernel.hpp:132:28
	v_mul_hi_u32 v1, v0, s7
	v_mul_lo_u32 v2, v1, s3
	.loc	6 131 25                        ; ./fmha_mfma_coalesced_kernel.hpp:131:25
	s_lshr_b32 s6, s7, 24
	.loc	6 132 28                        ; ./fmha_mfma_coalesced_kernel.hpp:132:28
	v_sub_u32_e32 v2, v0, v2
	.loc	6 131 25                        ; ./fmha_mfma_coalesced_kernel.hpp:131:25
	s_mul_i32 s13, s6, s3
	.loc	6 132 28                        ; ./fmha_mfma_coalesced_kernel.hpp:132:28
	v_add_u32_e32 v3, 1, v1
	v_cmp_le_u32_e32 vcc, s3, v2
	.loc	6 131 25                        ; ./fmha_mfma_coalesced_kernel.hpp:131:25
	s_sub_i32 s13, 0x100, s13
	s_add_i32 s27, s6, 1
	.loc	6 132 28                        ; ./fmha_mfma_coalesced_kernel.hpp:132:28
	v_cndmask_b32_e32 v1, v1, v3, vcc
	v_subrev_u32_e32 v3, s3, v2
	.loc	6 131 25                        ; ./fmha_mfma_coalesced_kernel.hpp:131:25
	s_sub_i32 s28, s13, s3
	.loc	6 132 28                        ; ./fmha_mfma_coalesced_kernel.hpp:132:28
	v_cndmask_b32_e32 v2, v2, v3, vcc
	.loc	6 131 25                        ; ./fmha_mfma_coalesced_kernel.hpp:131:25
	s_cmp_ge_u32 s13, s3
	.loc	6 132 28                        ; ./fmha_mfma_coalesced_kernel.hpp:132:28
	v_add_u32_e32 v3, 1, v1
	v_cmp_le_u32_e32 vcc, s3, v2
	.loc	6 131 25                        ; ./fmha_mfma_coalesced_kernel.hpp:131:25
	s_cselect_b32 s6, s27, s6
	s_cselect_b32 s13, s28, s13
	.loc	6 132 28                        ; ./fmha_mfma_coalesced_kernel.hpp:132:28
	v_cndmask_b32_e32 v1, v1, v3, vcc
	.loc	6 131 25                        ; ./fmha_mfma_coalesced_kernel.hpp:131:25
	s_add_i32 s27, s6, 1
	.loc	6 132 28                        ; ./fmha_mfma_coalesced_kernel.hpp:132:28
	v_xor_b32_e32 v1, s2, v1
	.loc	6 131 25                        ; ./fmha_mfma_coalesced_kernel.hpp:131:25
	s_cmp_ge_u32 s13, s3
	.loc	6 132 28                        ; ./fmha_mfma_coalesced_kernel.hpp:132:28
	v_subrev_u32_e32 v1, s2, v1
	.loc	6 131 25                        ; ./fmha_mfma_coalesced_kernel.hpp:131:25
	s_cselect_b32 s6, s27, s6
	.loc	6 133 29                        ; ./fmha_mfma_coalesced_kernel.hpp:133:29
	v_mul_lo_u32 v2, v1, s26
	.loc	6 131 25                        ; ./fmha_mfma_coalesced_kernel.hpp:131:25
	s_xor_b32 s6, s6, s2
	.loc	6 133 29                        ; ./fmha_mfma_coalesced_kernel.hpp:133:29
	v_sub_u32_e32 v2, v0, v2
	.loc	6 131 25                        ; ./fmha_mfma_coalesced_kernel.hpp:131:25
	s_sub_i32 s13, s6, s2
	.loc	6 133 36                        ; ./fmha_mfma_coalesced_kernel.hpp:133:36
	v_lshlrev_b32_e32 v14, 3, v2
	.loc	6 136 28                        ; ./fmha_mfma_coalesced_kernel.hpp:136:28
	v_cmp_gt_i32_e64 s[6:7], s24, v1
	v_mov_b32_e32 v3, 0
	v_mov_b32_e32 v2, 0
	.loc	6 136 5 is_stmt 0               ; ./fmha_mfma_coalesced_kernel.hpp:136:5
	s_and_saveexec_b64 s[2:3], s[6:7]
	s_cbranch_execz .LBB1_6
; %bb.3:                                ; %.lr.ph.preheader
	.loc	6 0 5                           ; ./fmha_mfma_coalesced_kernel.hpp:0:5
	s_load_dwordx2 s[0:1], s[0:1], 0x8
	.loc	6 136 5                         ; ./fmha_mfma_coalesced_kernel.hpp:136:5
	s_mul_i32 s27, s13, s16
	v_mov_b32_e32 v5, v1
	.loc	6 94 30 is_stmt 1               ; ./fmha_mfma_coalesced_kernel.hpp:94:30
	s_waitcnt lgkmcnt(0)
	s_add_u32 s0, s0, s22
	s_addc_u32 s1, s1, s23
	.loc	6 136 5                         ; ./fmha_mfma_coalesced_kernel.hpp:136:5
	s_lshl_b32 s22, s11, 1
	s_lshl_b32 s23, s26, 4
	s_sub_i32 s22, s22, s23
	s_add_i32 s22, s22, 8
	v_mul_lo_u32 v2, s22, v1
	s_lshl_b32 s22, s26, 3
	s_sub_i32 s22, s9, s22
	v_lshl_add_u32 v4, v0, 4, v2
	v_mul_lo_u32 v2, v1, s22
	s_lshl_b32 s27, s27, 1
	v_lshl_add_u32 v2, v0, 3, v2
	s_mul_i32 s26, s9, s13
	s_mov_b64 s[22:23], 0
.LBB1_4:                                ; %.lr.ph
                                        ; =>This Inner Loop Header: Depth=1
	.loc	6 138 31                        ; ./fmha_mfma_coalesced_kernel.hpp:138:31
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[6:7], v[2:3], 1, s[0:1]
	.loc	6 138 13 is_stmt 0              ; ./fmha_mfma_coalesced_kernel.hpp:138:13
	global_load_dwordx4 v[6:9], v[6:7], off
	.loc	6 136 43 is_stmt 1              ; ./fmha_mfma_coalesced_kernel.hpp:136:43
	v_add_u32_e32 v5, s13, v5
	.loc	6 136 28 is_stmt 0              ; ./fmha_mfma_coalesced_kernel.hpp:136:28
	v_cmp_le_i32_e32 vcc, s24, v5
	v_add_u32_e32 v2, s26, v2
	.loc	6 136 5                         ; ./fmha_mfma_coalesced_kernel.hpp:136:5
	s_or_b64 s[22:23], vcc, s[22:23]
	.loc	6 137 54 is_stmt 1              ; ./fmha_mfma_coalesced_kernel.hpp:137:54
	s_waitcnt vmcnt(0)
	ds_write_b128 v4, v[6:9]
	.loc	6 136 28                        ; ./fmha_mfma_coalesced_kernel.hpp:136:28
	v_add_u32_e32 v4, s27, v4
	.loc	6 136 5 is_stmt 0               ; ./fmha_mfma_coalesced_kernel.hpp:136:5
	s_andn2_b64 exec, exec, s[22:23]
	s_cbranch_execnz .LBB1_4
; %bb.5:                                ; %._crit_edge
	.loc	6 0 5                           ; ./fmha_mfma_coalesced_kernel.hpp:0:5
	s_or_b64 exec, exec, s[22:23]
	v_mad_u64_u32 v[2:3], s[0:1], v1, s9, v[14:15]
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[2:3], v[2:3], 1, s[18:19]
	.loc	6 146 19 is_stmt 1              ; ./fmha_mfma_coalesced_kernel.hpp:146:19
	global_load_dwordx4 v[2:5], v[2:3], off
.LBB1_6:                                ; %Flow304
	.loc	6 0 19 is_stmt 0                ; ./fmha_mfma_coalesced_kernel.hpp:0:19
	s_or_b64 exec, exec, s[2:3]
	.loc	6 148 33 is_stmt 1              ; ./fmha_mfma_coalesced_kernel.hpp:148:33
	v_add_u32_e32 v23, s13, v1
	v_mov_b32_e32 v6, 0
	.loc	6 149 18                        ; ./fmha_mfma_coalesced_kernel.hpp:149:18
	v_cmp_gt_i32_e64 s[2:3], s24, v23
	v_mov_b32_e32 v7, v6
	v_mov_b32_e32 v8, v6
	v_mov_b32_e32 v9, v6
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_8
; %bb.7:
	.loc	6 0 18 is_stmt 0                ; ./fmha_mfma_coalesced_kernel.hpp:0:18
	v_mad_u64_u32 v[6:7], s[22:23], v23, s9, v[14:15]
	v_ashrrev_i32_e32 v7, 31, v6
	v_lshl_add_u64 v[6:7], v[6:7], 1, s[18:19]
	.loc	6 150 19 is_stmt 1              ; ./fmha_mfma_coalesced_kernel.hpp:150:19
	global_load_dwordx4 v[6:9], v[6:7], off
.LBB1_8:
	.loc	6 0 19 is_stmt 0                ; ./fmha_mfma_coalesced_kernel.hpp:0:19
	s_or_b64 exec, exec, s[0:1]
	.loc	6 78 34 is_stmt 1               ; ./fmha_mfma_coalesced_kernel.hpp:78:34
	v_bfe_u32 v22, v0, 4, 2
	.loc	6 79 34                         ; ./fmha_mfma_coalesced_kernel.hpp:79:34
	v_and_b32_e32 v15, 15, v0
	.loc	6 92 52                         ; ./fmha_mfma_coalesced_kernel.hpp:92:52
	s_ashr_i32 s9, s8, 31
	.loc	6 92 66 is_stmt 0               ; ./fmha_mfma_coalesced_kernel.hpp:92:66
	s_ashr_i32 s23, s10, 31
	.loc	6 93 40 is_stmt 1               ; ./fmha_mfma_coalesced_kernel.hpp:93:40
	s_ashr_i32 s13, s12, 31
	.loc	6 161 17                        ; ./fmha_mfma_coalesced_kernel.hpp:161:17
	v_cmp_gt_u32_e32 vcc, 64, v0
.Ltmp40:
	.loc	3 691 5                         ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_device_functions.h:691:5 @[ /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_device_functions.h:707:63 @[ /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_device_functions.h:714:3 @[ ./fmha_mfma_coalesced_kernel.hpp:153:5 ] ] ]
	s_waitcnt lgkmcnt(0)
	.loc	3 692 5                         ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_device_functions.h:692:5 @[ /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_device_functions.h:707:63 @[ /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_device_functions.h:714:3 @[ ./fmha_mfma_coalesced_kernel.hpp:153:5 ] ] ]
	s_barrier
.Ltmp41:
	.loc	6 161 17                        ; ./fmha_mfma_coalesced_kernel.hpp:161:17
	s_and_saveexec_b64 s[18:19], vcc
	s_cbranch_execz .LBB1_19
; %bb.9:                                ; %.preheader
	.loc	6 162 27                        ; ./fmha_mfma_coalesced_kernel.hpp:162:27
	s_cmp_lt_i32 s11, 1
	v_cmp_gt_i32_e32 vcc, s24, v15
	.loc	6 162 9 is_stmt 0               ; ./fmha_mfma_coalesced_kernel.hpp:162:9
	s_cbranch_scc1 .LBB1_16
; %bb.10:                               ; %.lr.ph194
	.loc	6 0 9                           ; ./fmha_mfma_coalesced_kernel.hpp:0:9
	s_add_i32 s0, s11, 15
	.loc	6 162 9                         ; ./fmha_mfma_coalesced_kernel.hpp:162:9
	s_mul_hi_i32 s27, s4, s8
	s_mul_i32 s26, s4, s8
	s_lshr_b32 s22, s0, 4
	s_lshl_b64 s[26:27], s[26:27], 1
	s_lshl_b64 s[28:29], s[12:13], 1
	s_add_u32 s26, s26, s28
	s_addc_u32 s27, s27, s29
	s_mul_i32 s25, s26, s25
	s_mul_hi_u32 s28, s26, s11
	s_add_i32 s25, s28, s25
	s_mul_i32 s27, s27, s11
	s_mul_i32 s26, s26, s11
	s_add_i32 s25, s25, s27
	s_mul_i32 s23, s26, s23
	s_mul_hi_u32 s27, s26, s10
	v_lshrrev_b32_e32 v10, 4, v0
	s_add_i32 s23, s27, s23
	s_mul_i32 s25, s25, s10
	v_mul_lo_u32 v11, s16, v15
	v_lshlrev_b32_e32 v10, 3, v10
	s_add_i32 s23, s23, s25
	s_mul_i32 s26, s26, s10
	v_lshl_add_u32 v24, v11, 1, v10
	v_lshrrev_b32_e32 v10, 1, v0
	s_add_u32 s20, s20, s26
	v_and_b32_e32 v16, 0x78, v10
	v_mov_b32_e32 v17, 0
	s_addc_u32 s21, s21, s23
	v_lshl_add_u64 v[18:19], s[20:21], 0, v[16:17]
	v_mov_b32_e32 v16, v17
	v_cmp_eq_u32_e64 s[0:1], 0, v15
	v_mov_b64_e32 v[20:21], v[16:17]
	v_mov_b32_e32 v10, v17
	v_mov_b32_e32 v11, v17
	v_mov_b32_e32 v12, v17
	v_mov_b32_e32 v13, v17
	s_branch .LBB1_12
.LBB1_11:                               ;   in Loop: Header=BB1_12 Depth=1
	.loc	6 0 9                           ; ./fmha_mfma_coalesced_kernel.hpp:0:9
	s_or_b64 exec, exec, s[20:21]
	.loc	6 175 19 is_stmt 1              ; ./fmha_mfma_coalesced_kernel.hpp:175:19
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_mfma_f32_16x16x16_bf16 v[10:13], v[20:21], v[16:17], v[10:13]
	.loc	6 162 27                        ; ./fmha_mfma_coalesced_kernel.hpp:162:27
	s_add_i32 s22, s22, -1
	v_lshl_add_u64 v[18:19], v[18:19], 0, 32
	s_cmp_eq_u32 s22, 0
	v_add_u32_e32 v24, 32, v24
	.loc	6 162 9 is_stmt 0               ; ./fmha_mfma_coalesced_kernel.hpp:162:9
	s_cbranch_scc1 .LBB1_17
.LBB1_12:                               ; =>This Inner Loop Header: Depth=1
	.loc	6 166 26 is_stmt 1              ; ./fmha_mfma_coalesced_kernel.hpp:166:26
	s_and_saveexec_b64 s[20:21], s[0:1]
	s_cbranch_execz .LBB1_14
; %bb.13:                               ;   in Loop: Header=BB1_12 Depth=1
	.loc	6 167 21                        ; ./fmha_mfma_coalesced_kernel.hpp:167:21
	global_load_dwordx2 v[20:21], v[18:19], off
.LBB1_14:                               ;   in Loop: Header=BB1_12 Depth=1
	.loc	6 0 21 is_stmt 0                ; ./fmha_mfma_coalesced_kernel.hpp:0:21
	s_or_b64 exec, exec, s[20:21]
	.loc	6 171 26 is_stmt 1              ; ./fmha_mfma_coalesced_kernel.hpp:171:26
	s_and_saveexec_b64 s[20:21], vcc
	s_cbranch_execz .LBB1_11
; %bb.15:                               ;   in Loop: Header=BB1_12 Depth=1
	.loc	6 172 21                        ; ./fmha_mfma_coalesced_kernel.hpp:172:21
	ds_read_b64 v[16:17], v24
	s_branch .LBB1_11
.LBB1_16:
	.loc	6 0 21 is_stmt 0                ; ./fmha_mfma_coalesced_kernel.hpp:0:21
	v_mov_b32_e32 v10, 0
.LBB1_17:                               ; %._crit_edge195
	.loc	6 186 22 is_stmt 1              ; ./fmha_mfma_coalesced_kernel.hpp:186:22
	v_cmp_eq_u32_e32 vcc, 0, v22
	v_cmp_gt_i32_e64 s[0:1], s24, v15
	.loc	6 186 27 is_stmt 0              ; ./fmha_mfma_coalesced_kernel.hpp:186:27
	v_mul_f32_e32 v10, s17, v10
	v_mov_b32_e32 v11, 0xff800000
	s_and_b64 vcc, vcc, s[0:1]
	v_cndmask_b32_e32 v10, v11, v10, vcc
	v_mbcnt_lo_u32_b32 v11, -1, 0
	v_mbcnt_hi_u32_b32 v11, -1, v11
	v_and_b32_e32 v12, 0x70, v11
	v_add_u32_e32 v12, 16, v12
.Ltmp42:
	.loc	4 493 20 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_warp_functions.h:493:20 @[ /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_warp_functions.h:515:11 @[ ./fmha_mfma_coalesced_kernel.hpp:193:36 ] ]
	v_xor_b32_e32 v13, 8, v11
	.loc	4 494 17                        ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_warp_functions.h:494:17 @[ /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_warp_functions.h:515:11 @[ ./fmha_mfma_coalesced_kernel.hpp:193:36 ] ]
	v_cmp_lt_i32_e64 s[0:1], v13, v12
	.loc	4 493 20                        ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_warp_functions.h:493:20 @[ /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_warp_functions.h:515:11 @[ ./fmha_mfma_coalesced_kernel.hpp:193:36 ] ]
	v_xor_b32_e32 v17, 4, v11
	.loc	4 494 11                        ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_warp_functions.h:494:11 @[ /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_warp_functions.h:515:11 @[ ./fmha_mfma_coalesced_kernel.hpp:193:36 ] ]
	s_nop 0
	v_cndmask_b32_e64 v13, v11, v13, s[0:1]
	.loc	4 495 45                        ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_warp_functions.h:495:45 @[ /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_warp_functions.h:515:11 @[ ./fmha_mfma_coalesced_kernel.hpp:193:36 ] ]
	v_lshlrev_b32_e32 v13, 2, v13
	.loc	4 495 10 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_warp_functions.h:495:10 @[ /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_warp_functions.h:515:11 @[ ./fmha_mfma_coalesced_kernel.hpp:193:36 ] ]
	ds_bpermute_b32 v16, v13, v10
	.loc	4 494 17 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_warp_functions.h:494:17 @[ /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_warp_functions.h:515:11 @[ ./fmha_mfma_coalesced_kernel.hpp:193:36 ] ]
	v_cmp_lt_i32_e64 s[0:1], v17, v12
.Ltmp43:
	.loc	5 454 44                        ; /opt/rocm-7.2.0/lib/llvm/lib/clang/22/include/__clang_hip_math.h:454:44 @[ ./fmha_mfma_coalesced_kernel.hpp:193:22 ]
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v16, v16, v16
.Ltmp44:
	.loc	4 494 11                        ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_warp_functions.h:494:11 @[ /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_warp_functions.h:515:11 @[ ./fmha_mfma_coalesced_kernel.hpp:193:36 ] ]
	v_cndmask_b32_e64 v17, v11, v17, s[0:1]
.Ltmp45:
	.loc	5 454 44                        ; /opt/rocm-7.2.0/lib/llvm/lib/clang/22/include/__clang_hip_math.h:454:44 @[ ./fmha_mfma_coalesced_kernel.hpp:193:22 ]
	v_max_f32_e32 v16, v10, v16
.Ltmp46:
	.loc	4 495 45                        ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_warp_functions.h:495:45 @[ /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_warp_functions.h:515:11 @[ ./fmha_mfma_coalesced_kernel.hpp:193:36 ] ]
	v_lshlrev_b32_e32 v17, 2, v17
	.loc	4 495 10 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_warp_functions.h:495:10 @[ /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_warp_functions.h:515:11 @[ ./fmha_mfma_coalesced_kernel.hpp:193:36 ] ]
	ds_bpermute_b32 v18, v17, v16
.Ltmp47:
	.loc	5 454 44 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/lib/clang/22/include/__clang_hip_math.h:454:44 @[ ./fmha_mfma_coalesced_kernel.hpp:193:22 ]
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v18, v18, v18
	v_max_f32_e32 v16, v16, v18
.Ltmp48:
	.loc	4 493 20                        ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_warp_functions.h:493:20 @[ /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_warp_functions.h:515:11 @[ ./fmha_mfma_coalesced_kernel.hpp:193:36 ] ]
	v_xor_b32_e32 v18, 2, v11
	.loc	4 494 17                        ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_warp_functions.h:494:17 @[ /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_warp_functions.h:515:11 @[ ./fmha_mfma_coalesced_kernel.hpp:193:36 ] ]
	v_cmp_lt_i32_e64 s[0:1], v18, v12
	.loc	4 494 11 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_warp_functions.h:494:11 @[ /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_warp_functions.h:515:11 @[ ./fmha_mfma_coalesced_kernel.hpp:193:36 ] ]
	s_nop 1
	v_cndmask_b32_e64 v18, v11, v18, s[0:1]
	.loc	4 495 45 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_warp_functions.h:495:45 @[ /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_warp_functions.h:515:11 @[ ./fmha_mfma_coalesced_kernel.hpp:193:36 ] ]
	v_lshlrev_b32_e32 v18, 2, v18
	.loc	4 495 10 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_warp_functions.h:495:10 @[ /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_warp_functions.h:515:11 @[ ./fmha_mfma_coalesced_kernel.hpp:193:36 ] ]
	ds_bpermute_b32 v19, v18, v16
.Ltmp49:
	.loc	5 454 44 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/lib/clang/22/include/__clang_hip_math.h:454:44 @[ ./fmha_mfma_coalesced_kernel.hpp:193:22 ]
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v19, v19, v19
	v_max_f32_e32 v16, v16, v19
.Ltmp50:
	.loc	4 493 20                        ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_warp_functions.h:493:20 @[ /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_warp_functions.h:515:11 @[ ./fmha_mfma_coalesced_kernel.hpp:193:36 ] ]
	v_xor_b32_e32 v19, 1, v11
	.loc	4 494 17                        ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_warp_functions.h:494:17 @[ /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_warp_functions.h:515:11 @[ ./fmha_mfma_coalesced_kernel.hpp:193:36 ] ]
	v_cmp_lt_i32_e64 s[0:1], v19, v12
	.loc	4 494 11 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_warp_functions.h:494:11 @[ /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_warp_functions.h:515:11 @[ ./fmha_mfma_coalesced_kernel.hpp:193:36 ] ]
	s_nop 1
	v_cndmask_b32_e64 v11, v11, v19, s[0:1]
	.loc	4 495 45 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_warp_functions.h:495:45 @[ /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_warp_functions.h:515:11 @[ ./fmha_mfma_coalesced_kernel.hpp:193:36 ] ]
	v_lshlrev_b32_e32 v12, 2, v11
	.loc	4 495 10 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_warp_functions.h:495:10 @[ /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_warp_functions.h:515:11 @[ ./fmha_mfma_coalesced_kernel.hpp:193:36 ] ]
	ds_bpermute_b32 v11, v12, v16
	s_mov_b32 s0, 0x3fb8aa3b
.Ltmp51:
	.loc	5 454 44 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/lib/clang/22/include/__clang_hip_math.h:454:44 @[ ./fmha_mfma_coalesced_kernel.hpp:193:22 ]
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v11, v11, v11
	v_max_f32_e32 v11, v16, v11
	v_sub_f32_e32 v10, v10, v11
	v_mul_f32_e32 v11, 0x3fb8aa3b, v10
	v_fma_f32 v16, v10, s0, -v11
	v_rndne_f32_e32 v19, v11
	v_fmamk_f32 v16, v10, 0x32a5705f, v16
	v_sub_f32_e32 v11, v11, v19
	v_add_f32_e32 v11, v11, v16
	v_exp_f32_e32 v11, v11
	v_cvt_i32_f32_e32 v16, v19
	s_mov_b32 s0, 0xc2ce8ed0
	v_cmp_ngt_f32_e64 s[0:1], s0, v10
	v_ldexp_f32 v11, v11, v16
	s_nop 0
	v_cndmask_b32_e64 v11, 0, v11, s[0:1]
	s_mov_b32 s0, 0x42b17218
	v_mov_b32_e32 v16, 0x7f800000
	v_cmp_nlt_f32_e64 s[0:1], s0, v10
	s_nop 1
	v_cndmask_b32_e64 v10, v16, v11, s[0:1]
.Ltmp52:
	.loc	6 198 27                        ; ./fmha_mfma_coalesced_kernel.hpp:198:27
	v_cndmask_b32_e32 v11, 0, v10, vcc
.Ltmp53:
	.loc	4 495 10                        ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_warp_functions.h:495:10 @[ /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_warp_functions.h:515:11 @[ ./fmha_mfma_coalesced_kernel.hpp:203:23 ] ]
	ds_bpermute_b32 v13, v13, v11
.Ltmp54:
	.loc	6 203 20                        ; ./fmha_mfma_coalesced_kernel.hpp:203:20
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v11, v11, v13
.Ltmp55:
	.loc	4 495 10                        ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_warp_functions.h:495:10 @[ /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_warp_functions.h:515:11 @[ ./fmha_mfma_coalesced_kernel.hpp:203:23 ] ]
	ds_bpermute_b32 v13, v17, v11
.Ltmp56:
	.loc	6 203 20                        ; ./fmha_mfma_coalesced_kernel.hpp:203:20
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v11, v11, v13
.Ltmp57:
	.loc	4 495 10                        ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_warp_functions.h:495:10 @[ /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_warp_functions.h:515:11 @[ ./fmha_mfma_coalesced_kernel.hpp:203:23 ] ]
	ds_bpermute_b32 v13, v18, v11
.Ltmp58:
	.loc	6 203 20                        ; ./fmha_mfma_coalesced_kernel.hpp:203:20
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v11, v11, v13
.Ltmp59:
	.loc	4 495 10                        ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_warp_functions.h:495:10 @[ /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_warp_functions.h:515:11 @[ ./fmha_mfma_coalesced_kernel.hpp:203:23 ] ]
	ds_bpermute_b32 v12, v12, v11
.Ltmp60:
	.loc	6 207 27                        ; ./fmha_mfma_coalesced_kernel.hpp:207:27
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB1_19
; %bb.18:
	.loc	6 203 20                        ; ./fmha_mfma_coalesced_kernel.hpp:203:20
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v11, v11, v12
	.loc	6 208 65                        ; ./fmha_mfma_coalesced_kernel.hpp:208:65
	v_div_scale_f32 v12, s[0:1], v11, v11, v10
	v_rcp_f32_e32 v13, v12
	v_div_scale_f32 v16, vcc, v10, v11, v10
	s_movk_i32 s0, 0x7fff
	v_fma_f32 v17, -v12, v13, 1.0
	v_fmac_f32_e32 v13, v17, v13
	v_mul_f32_e32 v17, v16, v13
	v_fma_f32 v18, -v12, v17, v16
	v_fmac_f32_e32 v17, v18, v13
	v_fma_f32 v12, -v12, v17, v16
	v_div_fmas_f32 v12, v12, v13, v17
	v_div_fixup_f32 v10, v12, v11, v10
	.loc	6 208 57 is_stmt 0              ; ./fmha_mfma_coalesced_kernel.hpp:208:57
	v_bfe_u32 v11, v10, 16, 1
	v_add3_u32 v11, v11, v10, s0
	v_or_b32_e32 v12, 0x400000, v10
	v_cmp_u_f32_e32 vcc, v10, v10
	s_nop 1
	v_cndmask_b32_e32 v10, v11, v12, vcc
	.loc	6 208 13                        ; ./fmha_mfma_coalesced_kernel.hpp:208:13
	v_lshlrev_b32_e32 v11, 1, v15
	.loc	6 208 35                        ; ./fmha_mfma_coalesced_kernel.hpp:208:35
	ds_write_b16_d16_hi v11, v10 offset:8320
.LBB1_19:                               ; %Flow303
	.loc	6 0 35                          ; ./fmha_mfma_coalesced_kernel.hpp:0:35
	s_or_b64 exec, exec, s[18:19]
	v_lshlrev_b32_e32 v10, 1, v14
.Ltmp61:
	.loc	3 691 5 is_stmt 1               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_device_functions.h:691:5 @[ /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_device_functions.h:707:63 @[ /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_device_functions.h:714:3 @[ ./fmha_mfma_coalesced_kernel.hpp:212:5 ] ] ]
	s_waitcnt lgkmcnt(0)
	.loc	3 692 5                         ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_device_functions.h:692:5 @[ /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_device_functions.h:707:63 @[ /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_device_functions.h:714:3 @[ ./fmha_mfma_coalesced_kernel.hpp:212:5 ] ] ]
	s_barrier
.Ltmp62:
	.loc	6 219 16                        ; ./fmha_mfma_coalesced_kernel.hpp:219:16
	s_and_saveexec_b64 s[0:1], s[6:7]
	s_cbranch_execz .LBB1_21
; %bb.20:
	.loc	6 0 16 is_stmt 0                ; ./fmha_mfma_coalesced_kernel.hpp:0:16
	v_mul_lo_u32 v1, v1, s16
	v_lshl_add_u32 v1, v1, 1, v10
	.loc	6 220 59 is_stmt 1              ; ./fmha_mfma_coalesced_kernel.hpp:220:59
	s_waitcnt vmcnt(0)
	ds_write_b128 v1, v[2:5]
.LBB1_21:
	.loc	6 0 59 is_stmt 0                ; ./fmha_mfma_coalesced_kernel.hpp:0:59
	s_or_b64 exec, exec, s[0:1]
	.loc	6 222 18 is_stmt 1              ; ./fmha_mfma_coalesced_kernel.hpp:222:18
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_23
; %bb.22:
	.loc	6 0 18 is_stmt 0                ; ./fmha_mfma_coalesced_kernel.hpp:0:18
	v_mul_lo_u32 v1, v23, s16
	v_lshl_add_u32 v1, v1, 1, v10
	.loc	6 223 61 is_stmt 1              ; ./fmha_mfma_coalesced_kernel.hpp:223:61
	s_waitcnt vmcnt(0)
	ds_write_b128 v1, v[6:9]
.LBB1_23:
	.loc	6 0 61 is_stmt 0                ; ./fmha_mfma_coalesced_kernel.hpp:0:61
	s_or_b64 exec, exec, s[0:1]
	s_waitcnt vmcnt(0)
	v_mov_b32_e32 v4, 0
	.loc	6 233 18 is_stmt 1              ; ./fmha_mfma_coalesced_kernel.hpp:233:18
	v_cmp_eq_u32_e32 vcc, 0, v15
	v_mov_b32_e32 v5, v4
.Ltmp63:
	.loc	3 691 5                         ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_device_functions.h:691:5 @[ /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_device_functions.h:707:63 @[ /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_device_functions.h:714:3 @[ ./fmha_mfma_coalesced_kernel.hpp:226:5 ] ] ]
	s_waitcnt lgkmcnt(0)
	.loc	3 692 5                         ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_device_functions.h:692:5 @[ /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_device_functions.h:707:63 @[ /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_device_functions.h:714:3 @[ ./fmha_mfma_coalesced_kernel.hpp:226:5 ] ] ]
	s_barrier
.Ltmp64:
	.loc	6 233 18                        ; ./fmha_mfma_coalesced_kernel.hpp:233:18
	s_and_saveexec_b64 s[0:1], vcc
; %bb.24:
	.loc	6 0 18 is_stmt 0                ; ./fmha_mfma_coalesced_kernel.hpp:0:18
	v_lshlrev_b32_e32 v1, 3, v22
	.loc	6 234 18 is_stmt 1              ; ./fmha_mfma_coalesced_kernel.hpp:234:18
	ds_read_b64 v[4:5], v1 offset:8320
; %bb.25:
	.loc	6 0 18 is_stmt 0                ; ./fmha_mfma_coalesced_kernel.hpp:0:18
	s_or_b64 exec, exec, s[0:1]
	s_add_i32 s0, s11, 63
	.loc	6 237 23 is_stmt 1              ; ./fmha_mfma_coalesced_kernel.hpp:237:23
	s_cmp_lt_u32 s0, 64
	s_cbranch_scc1 .LBB1_38
; %bb.26:                               ; %.lr.ph210
	.loc	6 92 50                         ; ./fmha_mfma_coalesced_kernel.hpp:92:50
	s_mul_i32 s6, s11, s10
	s_lshr_b32 s17, s0, 6
	s_mul_hi_i32 s2, s11, s10
	.loc	6 92 64 is_stmt 0               ; ./fmha_mfma_coalesced_kernel.hpp:92:64
	s_mul_i32 s0, s6, s5
	s_mul_hi_u32 s1, s6, s4
	s_add_i32 s0, s1, s0
	s_mul_i32 s1, s2, s4
	s_mul_i32 s3, s6, s4
	s_add_i32 s0, s0, s1
	.loc	6 92 75                         ; ./fmha_mfma_coalesced_kernel.hpp:92:75
	s_mul_i32 s1, s3, s9
	s_mul_hi_u32 s4, s3, s8
	s_add_i32 s1, s4, s1
	s_mul_i32 s0, s0, s8
	s_add_i32 s1, s1, s0
	s_mul_i32 s0, s3, s8
	.loc	6 93 60 is_stmt 1               ; ./fmha_mfma_coalesced_kernel.hpp:93:60
	s_mul_i32 s3, s6, s13
	s_mul_hi_u32 s4, s6, s12
	s_add_i32 s3, s4, s3
	s_mul_i32 s2, s2, s12
	s_add_i32 s3, s3, s2
	.loc	6 96 23                         ; ./fmha_mfma_coalesced_kernel.hpp:96:23
	s_lshl_b64 s[0:1], s[0:1], 1
	.loc	6 93 60                         ; ./fmha_mfma_coalesced_kernel.hpp:93:60
	s_mul_i32 s2, s6, s12
	.loc	6 96 23                         ; ./fmha_mfma_coalesced_kernel.hpp:96:23
	s_add_u32 s4, s14, s0
	s_addc_u32 s5, s15, s1
	.loc	6 97 23                         ; ./fmha_mfma_coalesced_kernel.hpp:97:23
	s_lshl_b64 s[0:1], s[2:3], 1
	v_lshlrev_b32_e32 v2, 2, v22
	s_add_u32 s8, s4, s0
	v_or_b32_e32 v3, 1, v2
	s_addc_u32 s9, s5, s1
	v_cmp_gt_i32_e64 s[0:1], s24, v2
	v_cmp_gt_i32_e64 s[2:3], s24, v3
	v_or_b32_e32 v3, 2, v2
	v_or_b32_e32 v2, 3, v2
	v_lshrrev_b32_e32 v1, 2, v0
	v_cmp_gt_i32_e64 s[6:7], s24, v2
	.loc	6 237 5                         ; ./fmha_mfma_coalesced_kernel.hpp:237:5
	v_mul_lo_u32 v2, s16, v22
	v_lshrrev_b32_e32 v0, 1, v0
	v_cmp_gt_i32_e64 s[4:5], s24, v3
	v_lshlrev_b32_e32 v2, 3, v2
	v_and_b32_e32 v0, 0x60, v0
	v_lshlrev_b32_e32 v3, 1, v15
	v_cmp_eq_u32_e32 vcc, 0, v22
	s_mul_i32 s12, s11, 6
	v_add3_u32 v10, v2, v0, v3
	s_lshl_b32 s13, s11, 2
	s_lshl_b32 s14, s11, 1
	v_and_or_b32 v6, v1, 48, v15
	v_mov_b32_e32 v8, 0
	s_mov_b32 s15, 0x5040100
	s_mov_b32 s16, 0xffff
	v_mov_b32_e32 v11, 0x5040100
	s_branch .LBB1_28
.LBB1_27:                               ;   in Loop: Header=BB1_28 Depth=1
	.loc	6 0 5 is_stmt 0                 ; ./fmha_mfma_coalesced_kernel.hpp:0:5
	s_or_b64 exec, exec, s[10:11]
	.loc	6 237 23 is_stmt 1              ; ./fmha_mfma_coalesced_kernel.hpp:237:23
	s_add_i32 s17, s17, -1
	v_add_u32_e32 v10, 0x80, v10
	s_cmp_eq_u32 s17, 0
	v_add_u32_e32 v6, 64, v6
	s_cbranch_scc1 .LBB1_38
.LBB1_28:                               ; =>This Inner Loop Header: Depth=1
	.loc	6 0 23 is_stmt 0                ; ./fmha_mfma_coalesced_kernel.hpp:0:23
	v_mov_b32_e32 v9, v8
	v_mov_b64_e32 v[0:1], v[8:9]
	.loc	6 245 20 is_stmt 1              ; ./fmha_mfma_coalesced_kernel.hpp:245:20
	s_and_saveexec_b64 s[10:11], s[0:1]
	s_cbranch_execnz .LBB1_35
; %bb.29:                               ;   in Loop: Header=BB1_28 Depth=1
	.loc	6 0 20 is_stmt 0                ; ./fmha_mfma_coalesced_kernel.hpp:0:20
	s_or_b64 exec, exec, s[10:11]
	.loc	6 245 20                        ; ./fmha_mfma_coalesced_kernel.hpp:245:20
	s_and_saveexec_b64 s[10:11], s[2:3]
	s_cbranch_execnz .LBB1_36
.LBB1_30:                               ;   in Loop: Header=BB1_28 Depth=1
	.loc	6 0 20                          ; ./fmha_mfma_coalesced_kernel.hpp:0:20
	s_or_b64 exec, exec, s[10:11]
	.loc	6 245 20                        ; ./fmha_mfma_coalesced_kernel.hpp:245:20
	s_and_saveexec_b64 s[10:11], s[4:5]
	s_cbranch_execnz .LBB1_37
.LBB1_31:                               ;   in Loop: Header=BB1_28 Depth=1
	.loc	6 0 20                          ; ./fmha_mfma_coalesced_kernel.hpp:0:20
	s_or_b64 exec, exec, s[10:11]
	.loc	6 245 20                        ; ./fmha_mfma_coalesced_kernel.hpp:245:20
	s_and_saveexec_b64 s[10:11], s[6:7]
	s_cbranch_execz .LBB1_33
.LBB1_32:                               ;   in Loop: Header=BB1_28 Depth=1
	.loc	6 246 24 is_stmt 1              ; ./fmha_mfma_coalesced_kernel.hpp:246:24
	v_add_u32_e32 v2, s12, v10
	ds_read_u16 v2, v2 offset:24
	.loc	6 246 22 is_stmt 0              ; ./fmha_mfma_coalesced_kernel.hpp:246:22
	s_waitcnt lgkmcnt(0)
	v_perm_b32 v1, v2, v1, s15
.LBB1_33:                               ;   in Loop: Header=BB1_28 Depth=1
	.loc	6 0 22                          ; ./fmha_mfma_coalesced_kernel.hpp:0:22
	s_or_b64 exec, exec, s[10:11]
	.loc	6 250 15 is_stmt 1              ; ./fmha_mfma_coalesced_kernel.hpp:250:15
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x16_bf16 v[0:3], v[4:5], v[0:1], 0
	.loc	6 253 22                        ; ./fmha_mfma_coalesced_kernel.hpp:253:22
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB1_27
; %bb.34:                               ;   in Loop: Header=BB1_28 Depth=1
	.loc	6 0 22 is_stmt 0                ; ./fmha_mfma_coalesced_kernel.hpp:0:22
	s_nop 4
	v_cvt_f16_f32_e32 v2, v0
	v_ashrrev_i32_e32 v7, 31, v6
	v_lshl_add_u64 v[0:1], v[6:7], 1, s[8:9]
	.loc	6 254 39 is_stmt 1              ; ./fmha_mfma_coalesced_kernel.hpp:254:39
	global_store_short v[0:1], v2, off
	s_branch .LBB1_27
.LBB1_35:                               ;   in Loop: Header=BB1_28 Depth=1
	.loc	6 246 24                        ; ./fmha_mfma_coalesced_kernel.hpp:246:24
	ds_read_u16 v0, v10
	.loc	6 246 22 is_stmt 0              ; ./fmha_mfma_coalesced_kernel.hpp:246:22
	v_mov_b32_e32 v1, v8
	s_waitcnt lgkmcnt(0)
	v_perm_b32 v0, 0, v0, v11
	s_or_b64 exec, exec, s[10:11]
	.loc	6 245 20 is_stmt 1              ; ./fmha_mfma_coalesced_kernel.hpp:245:20
	s_and_saveexec_b64 s[10:11], s[2:3]
	s_cbranch_execz .LBB1_30
.LBB1_36:                               ;   in Loop: Header=BB1_28 Depth=1
	.loc	6 246 24                        ; ./fmha_mfma_coalesced_kernel.hpp:246:24
	v_add_u32_e32 v2, s14, v10
	ds_read_u16 v2, v2 offset:8
	.loc	6 246 22 is_stmt 0              ; ./fmha_mfma_coalesced_kernel.hpp:246:22
	s_waitcnt lgkmcnt(0)
	v_perm_b32 v0, v2, v0, s15
	s_or_b64 exec, exec, s[10:11]
	.loc	6 245 20 is_stmt 1              ; ./fmha_mfma_coalesced_kernel.hpp:245:20
	s_and_saveexec_b64 s[10:11], s[4:5]
	s_cbranch_execz .LBB1_31
.LBB1_37:                               ;   in Loop: Header=BB1_28 Depth=1
	.loc	6 246 24                        ; ./fmha_mfma_coalesced_kernel.hpp:246:24
	v_add_u32_e32 v2, s13, v10
	ds_read_u16 v2, v2 offset:16
	.loc	6 246 22 is_stmt 0              ; ./fmha_mfma_coalesced_kernel.hpp:246:22
	s_waitcnt lgkmcnt(0)
	v_bfi_b32 v1, s16, v2, v1
	s_or_b64 exec, exec, s[10:11]
	.loc	6 245 20 is_stmt 1              ; ./fmha_mfma_coalesced_kernel.hpp:245:20
	s_and_saveexec_b64 s[10:11], s[6:7]
	s_cbranch_execnz .LBB1_32
	s_branch .LBB1_33
.LBB1_38:                               ; %._crit_edge211
	.loc	6 257 1                         ; ./fmha_mfma_coalesced_kernel.hpp:257:1
	s_endpgm
.Ltmp65:
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z19fmha_mfma_coalescedPKDF16bS0_S0_PDF16_PKiiiiiiif
		.amdhsa_group_segment_fixed_size 8352
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 68
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 0
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 1
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 65
		.amdhsa_next_free_sgpr 75
		.amdhsa_accum_offset 28
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_tg_split 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end1:
	.size	_Z19fmha_mfma_coalescedPKDF16bS0_S0_PDF16_PKiiiiiiif, .Lfunc_end1-_Z19fmha_mfma_coalescedPKDF16bS0_S0_PDF16_PKiiiiiiif
	.cfi_endproc
                                        ; -- End function
	.set _Z19fmha_mfma_coalescedPKDF16bS0_S0_PDF16_PKiiiiiiif.num_vgpr, 25
	.set _Z19fmha_mfma_coalescedPKDF16bS0_S0_PDF16_PKiiiiiiif.num_agpr, 0
	.set _Z19fmha_mfma_coalescedPKDF16bS0_S0_PDF16_PKiiiiiiif.numbered_sgpr, 30
	.set _Z19fmha_mfma_coalescedPKDF16bS0_S0_PDF16_PKiiiiiiif.num_named_barrier, 0
	.set _Z19fmha_mfma_coalescedPKDF16bS0_S0_PDF16_PKiiiiiiif.private_seg_size, 0
	.set _Z19fmha_mfma_coalescedPKDF16bS0_S0_PDF16_PKiiiiiiif.uses_vcc, 1
	.set _Z19fmha_mfma_coalescedPKDF16bS0_S0_PDF16_PKiiiiiiif.uses_flat_scratch, 0
	.set _Z19fmha_mfma_coalescedPKDF16bS0_S0_PDF16_PKiiiiiiif.has_dyn_sized_stack, 0
	.set _Z19fmha_mfma_coalescedPKDF16bS0_S0_PDF16_PKiiiiiiif.has_recursion, 0
	.set _Z19fmha_mfma_coalescedPKDF16bS0_S0_PDF16_PKiiiiiiif.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 2244
; TotalNumSgprs: 36
; NumVgprs: 25
; NumAgprs: 0
; TotalNumVgprs: 25
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 8352 bytes/workgroup (compile time only)
; SGPRBlocks: 10
; VGPRBlocks: 8
; NumSGPRsForWavesPerEU: 81
; NumVGPRsForWavesPerEU: 65
; AccumOffset: 28
; Occupancy: 7
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 1
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 6
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.text
	.type	__hip_cuid_124cf60105eb9515,@object ; @__hip_cuid_124cf60105eb9515
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_124cf60105eb9515
__hip_cuid_124cf60105eb9515:
	.byte	0                               ; 0x0
	.size	__hip_cuid_124cf60105eb9515, 1

	.section	.debug_abbrev,"",@progbits
	.byte	1                               ; Abbreviation Code
	.byte	17                              ; DW_TAG_compile_unit
	.byte	1                               ; DW_CHILDREN_yes
	.byte	37                              ; DW_AT_producer
	.byte	37                              ; DW_FORM_strx1
	.byte	19                              ; DW_AT_language
	.byte	5                               ; DW_FORM_data2
	.byte	3                               ; DW_AT_name
	.byte	37                              ; DW_FORM_strx1
	.byte	114                             ; DW_AT_str_offsets_base
	.byte	23                              ; DW_FORM_sec_offset
	.byte	16                              ; DW_AT_stmt_list
	.byte	23                              ; DW_FORM_sec_offset
	.byte	27                              ; DW_AT_comp_dir
	.byte	37                              ; DW_FORM_strx1
	.byte	17                              ; DW_AT_low_pc
	.byte	27                              ; DW_FORM_addrx
	.byte	18                              ; DW_AT_high_pc
	.byte	6                               ; DW_FORM_data4
	.byte	115                             ; DW_AT_addr_base
	.byte	23                              ; DW_FORM_sec_offset
	.byte	116                             ; DW_AT_rnglists_base
	.byte	23                              ; DW_FORM_sec_offset
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	2                               ; Abbreviation Code
	.byte	46                              ; DW_TAG_subprogram
	.byte	0                               ; DW_CHILDREN_no
	.byte	3                               ; DW_AT_name
	.byte	37                              ; DW_FORM_strx1
	.byte	32                              ; DW_AT_inline
	.byte	33                              ; DW_FORM_implicit_const
	.byte	1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	3                               ; Abbreviation Code
	.byte	46                              ; DW_TAG_subprogram
	.byte	1                               ; DW_CHILDREN_yes
	.byte	17                              ; DW_AT_low_pc
	.byte	27                              ; DW_FORM_addrx
	.byte	18                              ; DW_AT_high_pc
	.byte	6                               ; DW_FORM_data4
	.byte	122                             ; DW_AT_call_all_calls
	.byte	25                              ; DW_FORM_flag_present
	.byte	3                               ; DW_AT_name
	.byte	37                              ; DW_FORM_strx1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	4                               ; Abbreviation Code
	.byte	29                              ; DW_TAG_inlined_subroutine
	.byte	1                               ; DW_CHILDREN_yes
	.byte	49                              ; DW_AT_abstract_origin
	.byte	19                              ; DW_FORM_ref4
	.byte	85                              ; DW_AT_ranges
	.byte	35                              ; DW_FORM_rnglistx
	.byte	88                              ; DW_AT_call_file
	.byte	11                              ; DW_FORM_data1
	.byte	89                              ; DW_AT_call_line
	.byte	11                              ; DW_FORM_data1
	.byte	87                              ; DW_AT_call_column
	.byte	11                              ; DW_FORM_data1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	5                               ; Abbreviation Code
	.byte	29                              ; DW_TAG_inlined_subroutine
	.byte	0                               ; DW_CHILDREN_no
	.byte	49                              ; DW_AT_abstract_origin
	.byte	19                              ; DW_FORM_ref4
	.byte	85                              ; DW_AT_ranges
	.byte	35                              ; DW_FORM_rnglistx
	.byte	88                              ; DW_AT_call_file
	.byte	11                              ; DW_FORM_data1
	.byte	89                              ; DW_AT_call_line
	.byte	5                               ; DW_FORM_data2
	.byte	87                              ; DW_AT_call_column
	.byte	11                              ; DW_FORM_data1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	6                               ; Abbreviation Code
	.byte	29                              ; DW_TAG_inlined_subroutine
	.byte	1                               ; DW_CHILDREN_yes
	.byte	49                              ; DW_AT_abstract_origin
	.byte	19                              ; DW_FORM_ref4
	.byte	17                              ; DW_AT_low_pc
	.byte	27                              ; DW_FORM_addrx
	.byte	18                              ; DW_AT_high_pc
	.byte	6                               ; DW_FORM_data4
	.byte	88                              ; DW_AT_call_file
	.byte	11                              ; DW_FORM_data1
	.byte	89                              ; DW_AT_call_line
	.byte	11                              ; DW_FORM_data1
	.byte	87                              ; DW_AT_call_column
	.byte	11                              ; DW_FORM_data1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	7                               ; Abbreviation Code
	.byte	29                              ; DW_TAG_inlined_subroutine
	.byte	1                               ; DW_CHILDREN_yes
	.byte	49                              ; DW_AT_abstract_origin
	.byte	19                              ; DW_FORM_ref4
	.byte	17                              ; DW_AT_low_pc
	.byte	27                              ; DW_FORM_addrx
	.byte	18                              ; DW_AT_high_pc
	.byte	6                               ; DW_FORM_data4
	.byte	88                              ; DW_AT_call_file
	.byte	11                              ; DW_FORM_data1
	.byte	89                              ; DW_AT_call_line
	.byte	5                               ; DW_FORM_data2
	.byte	87                              ; DW_AT_call_column
	.byte	11                              ; DW_FORM_data1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	8                               ; Abbreviation Code
	.byte	29                              ; DW_TAG_inlined_subroutine
	.byte	0                               ; DW_CHILDREN_no
	.byte	49                              ; DW_AT_abstract_origin
	.byte	19                              ; DW_FORM_ref4
	.byte	17                              ; DW_AT_low_pc
	.byte	27                              ; DW_FORM_addrx
	.byte	18                              ; DW_AT_high_pc
	.byte	6                               ; DW_FORM_data4
	.byte	88                              ; DW_AT_call_file
	.byte	11                              ; DW_FORM_data1
	.byte	89                              ; DW_AT_call_line
	.byte	5                               ; DW_FORM_data2
	.byte	87                              ; DW_AT_call_column
	.byte	11                              ; DW_FORM_data1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	9                               ; Abbreviation Code
	.byte	29                              ; DW_TAG_inlined_subroutine
	.byte	0                               ; DW_CHILDREN_no
	.byte	49                              ; DW_AT_abstract_origin
	.byte	19                              ; DW_FORM_ref4
	.byte	85                              ; DW_AT_ranges
	.byte	35                              ; DW_FORM_rnglistx
	.byte	88                              ; DW_AT_call_file
	.byte	11                              ; DW_FORM_data1
	.byte	89                              ; DW_AT_call_line
	.byte	11                              ; DW_FORM_data1
	.byte	87                              ; DW_AT_call_column
	.byte	11                              ; DW_FORM_data1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	10                              ; Abbreviation Code
	.byte	29                              ; DW_TAG_inlined_subroutine
	.byte	0                               ; DW_CHILDREN_no
	.byte	49                              ; DW_AT_abstract_origin
	.byte	19                              ; DW_FORM_ref4
	.byte	17                              ; DW_AT_low_pc
	.byte	27                              ; DW_FORM_addrx
	.byte	18                              ; DW_AT_high_pc
	.byte	6                               ; DW_FORM_data4
	.byte	88                              ; DW_AT_call_file
	.byte	11                              ; DW_FORM_data1
	.byte	89                              ; DW_AT_call_line
	.byte	11                              ; DW_FORM_data1
	.byte	87                              ; DW_AT_call_column
	.byte	11                              ; DW_FORM_data1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	0                               ; EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0 ; Length of Unit
.Ldebug_info_start0:
	.short	5                               ; DWARF version number
	.byte	1                               ; DWARF Unit Type
	.byte	8                               ; Address Size (in bytes)
	.long	.debug_abbrev                   ; Offset Into Abbrev. Section
	.byte	1                               ; Abbrev [1] 0xc:0x20a DW_TAG_compile_unit
	.byte	0                               ; DW_AT_producer
	.short	33                              ; DW_AT_language
	.byte	1                               ; DW_AT_name
	.long	.Lstr_offsets_base0             ; DW_AT_str_offsets_base
	.long	.Lline_table_start0             ; DW_AT_stmt_list
	.byte	2                               ; DW_AT_comp_dir
	.byte	0                               ; DW_AT_low_pc
	.long	.Lfunc_end1-.Lfunc_begin0       ; DW_AT_high_pc
	.long	.Laddr_table_base0              ; DW_AT_addr_base
	.long	.Lrnglists_table_base0          ; DW_AT_rnglists_base
	.byte	2                               ; Abbrev [2] 0x27:0x2 DW_TAG_subprogram
	.byte	3                               ; DW_AT_name
                                        ; DW_AT_inline
	.byte	2                               ; Abbrev [2] 0x29:0x2 DW_TAG_subprogram
	.byte	4                               ; DW_AT_name
                                        ; DW_AT_inline
	.byte	2                               ; Abbrev [2] 0x2b:0x2 DW_TAG_subprogram
	.byte	5                               ; DW_AT_name
                                        ; DW_AT_inline
	.byte	2                               ; Abbrev [2] 0x2d:0x2 DW_TAG_subprogram
	.byte	6                               ; DW_AT_name
                                        ; DW_AT_inline
	.byte	2                               ; Abbrev [2] 0x2f:0x2 DW_TAG_subprogram
	.byte	7                               ; DW_AT_name
                                        ; DW_AT_inline
	.byte	2                               ; Abbrev [2] 0x31:0x2 DW_TAG_subprogram
	.byte	8                               ; DW_AT_name
                                        ; DW_AT_inline
	.byte	2                               ; Abbrev [2] 0x33:0x2 DW_TAG_subprogram
	.byte	8                               ; DW_AT_name
                                        ; DW_AT_inline
	.byte	2                               ; Abbrev [2] 0x35:0x2 DW_TAG_subprogram
	.byte	9                               ; DW_AT_name
                                        ; DW_AT_inline
	.byte	2                               ; Abbrev [2] 0x37:0x2 DW_TAG_subprogram
	.byte	10                              ; DW_AT_name
                                        ; DW_AT_inline
	.byte	3                               ; Abbrev [3] 0x39:0x106 DW_TAG_subprogram
	.byte	0                               ; DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       ; DW_AT_high_pc
                                        ; DW_AT_call_all_calls
	.byte	11                              ; DW_AT_name
	.byte	4                               ; Abbrev [4] 0x40:0x14 DW_TAG_inlined_subroutine
	.long	41                              ; DW_AT_abstract_origin
	.byte	0                               ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.byte	38                              ; DW_AT_call_line
	.byte	27                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x49:0xa DW_TAG_inlined_subroutine
	.long	39                              ; DW_AT_abstract_origin
	.byte	0                               ; DW_AT_ranges
	.byte	2                               ; DW_AT_call_file
	.short	289                             ; DW_AT_call_line
	.byte	160                             ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	6                               ; Abbrev [6] 0x54:0x2b DW_TAG_inlined_subroutine
	.long	47                              ; DW_AT_abstract_origin
	.byte	1                               ; DW_AT_low_pc
	.long	.Ltmp5-.Ltmp4                   ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.byte	101                             ; DW_AT_call_line
	.byte	5                               ; DW_AT_call_column
	.byte	7                               ; Abbrev [7] 0x61:0x1d DW_TAG_inlined_subroutine
	.long	45                              ; DW_AT_abstract_origin
	.byte	1                               ; DW_AT_low_pc
	.long	.Ltmp5-.Ltmp4                   ; DW_AT_high_pc
	.byte	3                               ; DW_AT_call_file
	.short	714                             ; DW_AT_call_line
	.byte	3                               ; DW_AT_call_column
	.byte	8                               ; Abbrev [8] 0x6f:0xe DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	1                               ; DW_AT_low_pc
	.long	.Ltmp5-.Ltmp4                   ; DW_AT_high_pc
	.byte	3                               ; DW_AT_call_file
	.short	707                             ; DW_AT_call_line
	.byte	63                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
	.byte	6                               ; Abbrev [6] 0x7f:0x2b DW_TAG_inlined_subroutine
	.long	47                              ; DW_AT_abstract_origin
	.byte	2                               ; DW_AT_low_pc
	.long	.Ltmp7-.Ltmp6                   ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.byte	130                             ; DW_AT_call_line
	.byte	5                               ; DW_AT_call_column
	.byte	7                               ; Abbrev [7] 0x8c:0x1d DW_TAG_inlined_subroutine
	.long	45                              ; DW_AT_abstract_origin
	.byte	2                               ; DW_AT_low_pc
	.long	.Ltmp7-.Ltmp6                   ; DW_AT_high_pc
	.byte	3                               ; DW_AT_call_file
	.short	714                             ; DW_AT_call_line
	.byte	3                               ; DW_AT_call_column
	.byte	8                               ; Abbrev [8] 0x9a:0xe DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	2                               ; DW_AT_low_pc
	.long	.Ltmp7-.Ltmp6                   ; DW_AT_high_pc
	.byte	3                               ; DW_AT_call_file
	.short	707                             ; DW_AT_call_line
	.byte	63                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
	.byte	4                               ; Abbrev [4] 0xaa:0x14 DW_TAG_inlined_subroutine
	.long	51                              ; DW_AT_abstract_origin
	.byte	1                               ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.byte	140                             ; DW_AT_call_line
	.byte	32                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0xb3:0xa DW_TAG_inlined_subroutine
	.long	49                              ; DW_AT_abstract_origin
	.byte	1                               ; DW_AT_ranges
	.byte	4                               ; DW_AT_call_file
	.short	515                             ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	9                               ; Abbrev [9] 0xbe:0x9 DW_TAG_inlined_subroutine
	.long	53                              ; DW_AT_abstract_origin
	.byte	2                               ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.byte	140                             ; DW_AT_call_line
	.byte	18                              ; DW_AT_call_column
	.byte	10                              ; Abbrev [10] 0xc7:0xd DW_TAG_inlined_subroutine
	.long	55                              ; DW_AT_abstract_origin
	.byte	3                               ; DW_AT_low_pc
	.long	.Ltmp23-.Ltmp22                 ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.byte	146                             ; DW_AT_call_line
	.byte	23                              ; DW_AT_call_column
	.byte	4                               ; Abbrev [4] 0xd4:0x14 DW_TAG_inlined_subroutine
	.long	51                              ; DW_AT_abstract_origin
	.byte	3                               ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.byte	152                             ; DW_AT_call_line
	.byte	19                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0xdd:0xa DW_TAG_inlined_subroutine
	.long	49                              ; DW_AT_abstract_origin
	.byte	3                               ; DW_AT_ranges
	.byte	4                               ; DW_AT_call_file
	.short	515                             ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	6                               ; Abbrev [6] 0xe8:0x2b DW_TAG_inlined_subroutine
	.long	47                              ; DW_AT_abstract_origin
	.byte	4                               ; DW_AT_low_pc
	.long	.Ltmp33-.Ltmp32                 ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.byte	162                             ; DW_AT_call_line
	.byte	5                               ; DW_AT_call_column
	.byte	7                               ; Abbrev [7] 0xf5:0x1d DW_TAG_inlined_subroutine
	.long	45                              ; DW_AT_abstract_origin
	.byte	4                               ; DW_AT_low_pc
	.long	.Ltmp33-.Ltmp32                 ; DW_AT_high_pc
	.byte	3                               ; DW_AT_call_file
	.short	714                             ; DW_AT_call_line
	.byte	3                               ; DW_AT_call_column
	.byte	8                               ; Abbrev [8] 0x103:0xe DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	4                               ; DW_AT_low_pc
	.long	.Ltmp33-.Ltmp32                 ; DW_AT_high_pc
	.byte	3                               ; DW_AT_call_file
	.short	707                             ; DW_AT_call_line
	.byte	63                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
	.byte	6                               ; Abbrev [6] 0x113:0x2b DW_TAG_inlined_subroutine
	.long	47                              ; DW_AT_abstract_origin
	.byte	5                               ; DW_AT_low_pc
	.long	.Ltmp35-.Ltmp34                 ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.byte	170                             ; DW_AT_call_line
	.byte	5                               ; DW_AT_call_column
	.byte	7                               ; Abbrev [7] 0x120:0x1d DW_TAG_inlined_subroutine
	.long	45                              ; DW_AT_abstract_origin
	.byte	5                               ; DW_AT_low_pc
	.long	.Ltmp35-.Ltmp34                 ; DW_AT_high_pc
	.byte	3                               ; DW_AT_call_file
	.short	714                             ; DW_AT_call_line
	.byte	3                               ; DW_AT_call_column
	.byte	8                               ; Abbrev [8] 0x12e:0xe DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	5                               ; DW_AT_low_pc
	.long	.Ltmp35-.Ltmp34                 ; DW_AT_high_pc
	.byte	3                               ; DW_AT_call_file
	.short	707                             ; DW_AT_call_line
	.byte	63                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
	.byte	3                               ; Abbrev [3] 0x13f:0xd6 DW_TAG_subprogram
	.byte	6                               ; DW_AT_low_pc
	.long	.Lfunc_end1-.Lfunc_begin1       ; DW_AT_high_pc
                                        ; DW_AT_call_all_calls
	.byte	12                              ; DW_AT_name
	.byte	6                               ; Abbrev [6] 0x146:0x1c DW_TAG_inlined_subroutine
	.long	41                              ; DW_AT_abstract_origin
	.byte	7                               ; DW_AT_low_pc
	.long	.Ltmp39-.Ltmp38                 ; DW_AT_high_pc
	.byte	6                               ; DW_AT_call_file
	.byte	68                              ; DW_AT_call_line
	.byte	27                              ; DW_AT_call_column
	.byte	8                               ; Abbrev [8] 0x153:0xe DW_TAG_inlined_subroutine
	.long	39                              ; DW_AT_abstract_origin
	.byte	7                               ; DW_AT_low_pc
	.long	.Ltmp39-.Ltmp38                 ; DW_AT_high_pc
	.byte	2                               ; DW_AT_call_file
	.short	289                             ; DW_AT_call_line
	.byte	160                             ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	6                               ; Abbrev [6] 0x162:0x2b DW_TAG_inlined_subroutine
	.long	47                              ; DW_AT_abstract_origin
	.byte	8                               ; DW_AT_low_pc
	.long	.Ltmp41-.Ltmp40                 ; DW_AT_high_pc
	.byte	6                               ; DW_AT_call_file
	.byte	153                             ; DW_AT_call_line
	.byte	5                               ; DW_AT_call_column
	.byte	7                               ; Abbrev [7] 0x16f:0x1d DW_TAG_inlined_subroutine
	.long	45                              ; DW_AT_abstract_origin
	.byte	8                               ; DW_AT_low_pc
	.long	.Ltmp41-.Ltmp40                 ; DW_AT_high_pc
	.byte	3                               ; DW_AT_call_file
	.short	714                             ; DW_AT_call_line
	.byte	3                               ; DW_AT_call_column
	.byte	8                               ; Abbrev [8] 0x17d:0xe DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	8                               ; DW_AT_low_pc
	.long	.Ltmp41-.Ltmp40                 ; DW_AT_high_pc
	.byte	3                               ; DW_AT_call_file
	.short	707                             ; DW_AT_call_line
	.byte	63                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
	.byte	4                               ; Abbrev [4] 0x18d:0x14 DW_TAG_inlined_subroutine
	.long	51                              ; DW_AT_abstract_origin
	.byte	4                               ; DW_AT_ranges
	.byte	6                               ; DW_AT_call_file
	.byte	193                             ; DW_AT_call_line
	.byte	36                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x196:0xa DW_TAG_inlined_subroutine
	.long	49                              ; DW_AT_abstract_origin
	.byte	4                               ; DW_AT_ranges
	.byte	4                               ; DW_AT_call_file
	.short	515                             ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	9                               ; Abbrev [9] 0x1a1:0x9 DW_TAG_inlined_subroutine
	.long	53                              ; DW_AT_abstract_origin
	.byte	5                               ; DW_AT_ranges
	.byte	6                               ; DW_AT_call_file
	.byte	193                             ; DW_AT_call_line
	.byte	22                              ; DW_AT_call_column
	.byte	4                               ; Abbrev [4] 0x1aa:0x14 DW_TAG_inlined_subroutine
	.long	51                              ; DW_AT_abstract_origin
	.byte	6                               ; DW_AT_ranges
	.byte	6                               ; DW_AT_call_file
	.byte	203                             ; DW_AT_call_line
	.byte	23                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x1b3:0xa DW_TAG_inlined_subroutine
	.long	49                              ; DW_AT_abstract_origin
	.byte	6                               ; DW_AT_ranges
	.byte	4                               ; DW_AT_call_file
	.short	515                             ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	6                               ; Abbrev [6] 0x1be:0x2b DW_TAG_inlined_subroutine
	.long	47                              ; DW_AT_abstract_origin
	.byte	9                               ; DW_AT_low_pc
	.long	.Ltmp62-.Ltmp61                 ; DW_AT_high_pc
	.byte	6                               ; DW_AT_call_file
	.byte	212                             ; DW_AT_call_line
	.byte	5                               ; DW_AT_call_column
	.byte	7                               ; Abbrev [7] 0x1cb:0x1d DW_TAG_inlined_subroutine
	.long	45                              ; DW_AT_abstract_origin
	.byte	9                               ; DW_AT_low_pc
	.long	.Ltmp62-.Ltmp61                 ; DW_AT_high_pc
	.byte	3                               ; DW_AT_call_file
	.short	714                             ; DW_AT_call_line
	.byte	3                               ; DW_AT_call_column
	.byte	8                               ; Abbrev [8] 0x1d9:0xe DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	9                               ; DW_AT_low_pc
	.long	.Ltmp62-.Ltmp61                 ; DW_AT_high_pc
	.byte	3                               ; DW_AT_call_file
	.short	707                             ; DW_AT_call_line
	.byte	63                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
	.byte	6                               ; Abbrev [6] 0x1e9:0x2b DW_TAG_inlined_subroutine
	.long	47                              ; DW_AT_abstract_origin
	.byte	10                              ; DW_AT_low_pc
	.long	.Ltmp64-.Ltmp63                 ; DW_AT_high_pc
	.byte	6                               ; DW_AT_call_file
	.byte	226                             ; DW_AT_call_line
	.byte	5                               ; DW_AT_call_column
	.byte	7                               ; Abbrev [7] 0x1f6:0x1d DW_TAG_inlined_subroutine
	.long	45                              ; DW_AT_abstract_origin
	.byte	10                              ; DW_AT_low_pc
	.long	.Ltmp64-.Ltmp63                 ; DW_AT_high_pc
	.byte	3                               ; DW_AT_call_file
	.short	714                             ; DW_AT_call_line
	.byte	3                               ; DW_AT_call_column
	.byte	8                               ; Abbrev [8] 0x204:0xe DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	10                              ; DW_AT_low_pc
	.long	.Ltmp64-.Ltmp63                 ; DW_AT_high_pc
	.byte	3                               ; DW_AT_call_file
	.short	707                             ; DW_AT_call_line
	.byte	63                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_rnglists,"",@progbits
	.long	.Ldebug_list_header_end0-.Ldebug_list_header_start0 ; Length
.Ldebug_list_header_start0:
	.short	5                               ; Version
	.byte	8                               ; Address size
	.byte	0                               ; Segment selector size
	.long	7                               ; Offset entry count
.Lrnglists_table_base0:
	.long	.Ldebug_ranges0-.Lrnglists_table_base0
	.long	.Ldebug_ranges1-.Lrnglists_table_base0
	.long	.Ldebug_ranges2-.Lrnglists_table_base0
	.long	.Ldebug_ranges3-.Lrnglists_table_base0
	.long	.Ldebug_ranges4-.Lrnglists_table_base0
	.long	.Ldebug_ranges5-.Lrnglists_table_base0
	.long	.Ldebug_ranges6-.Lrnglists_table_base0
.Ldebug_ranges0:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp0-.Lfunc_begin0           ;   starting offset
	.uleb128 .Ltmp1-.Lfunc_begin0           ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp2-.Lfunc_begin0           ;   starting offset
	.uleb128 .Ltmp3-.Lfunc_begin0           ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges1:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp8-.Lfunc_begin0           ;   starting offset
	.uleb128 .Ltmp9-.Lfunc_begin0           ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp10-.Lfunc_begin0          ;   starting offset
	.uleb128 .Ltmp11-.Lfunc_begin0          ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp12-.Lfunc_begin0          ;   starting offset
	.uleb128 .Ltmp13-.Lfunc_begin0          ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp14-.Lfunc_begin0          ;   starting offset
	.uleb128 .Ltmp15-.Lfunc_begin0          ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp16-.Lfunc_begin0          ;   starting offset
	.uleb128 .Ltmp17-.Lfunc_begin0          ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp18-.Lfunc_begin0          ;   starting offset
	.uleb128 .Ltmp19-.Lfunc_begin0          ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges2:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp9-.Lfunc_begin0           ;   starting offset
	.uleb128 .Ltmp10-.Lfunc_begin0          ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp11-.Lfunc_begin0          ;   starting offset
	.uleb128 .Ltmp12-.Lfunc_begin0          ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp13-.Lfunc_begin0          ;   starting offset
	.uleb128 .Ltmp14-.Lfunc_begin0          ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp15-.Lfunc_begin0          ;   starting offset
	.uleb128 .Ltmp16-.Lfunc_begin0          ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp17-.Lfunc_begin0          ;   starting offset
	.uleb128 .Ltmp18-.Lfunc_begin0          ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp20-.Lfunc_begin0          ;   starting offset
	.uleb128 .Ltmp21-.Lfunc_begin0          ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges3:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp24-.Lfunc_begin0          ;   starting offset
	.uleb128 .Ltmp25-.Lfunc_begin0          ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp26-.Lfunc_begin0          ;   starting offset
	.uleb128 .Ltmp27-.Lfunc_begin0          ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp28-.Lfunc_begin0          ;   starting offset
	.uleb128 .Ltmp29-.Lfunc_begin0          ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp30-.Lfunc_begin0          ;   starting offset
	.uleb128 .Ltmp31-.Lfunc_begin0          ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges4:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp42-.Lfunc_begin0          ;   starting offset
	.uleb128 .Ltmp43-.Lfunc_begin0          ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp44-.Lfunc_begin0          ;   starting offset
	.uleb128 .Ltmp45-.Lfunc_begin0          ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp46-.Lfunc_begin0          ;   starting offset
	.uleb128 .Ltmp47-.Lfunc_begin0          ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp48-.Lfunc_begin0          ;   starting offset
	.uleb128 .Ltmp49-.Lfunc_begin0          ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp50-.Lfunc_begin0          ;   starting offset
	.uleb128 .Ltmp51-.Lfunc_begin0          ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges5:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp43-.Lfunc_begin0          ;   starting offset
	.uleb128 .Ltmp44-.Lfunc_begin0          ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp45-.Lfunc_begin0          ;   starting offset
	.uleb128 .Ltmp46-.Lfunc_begin0          ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp47-.Lfunc_begin0          ;   starting offset
	.uleb128 .Ltmp48-.Lfunc_begin0          ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp49-.Lfunc_begin0          ;   starting offset
	.uleb128 .Ltmp50-.Lfunc_begin0          ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp51-.Lfunc_begin0          ;   starting offset
	.uleb128 .Ltmp52-.Lfunc_begin0          ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges6:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp53-.Lfunc_begin0          ;   starting offset
	.uleb128 .Ltmp54-.Lfunc_begin0          ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp55-.Lfunc_begin0          ;   starting offset
	.uleb128 .Ltmp56-.Lfunc_begin0          ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp57-.Lfunc_begin0          ;   starting offset
	.uleb128 .Ltmp58-.Lfunc_begin0          ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp59-.Lfunc_begin0          ;   starting offset
	.uleb128 .Ltmp60-.Lfunc_begin0          ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_list_header_end0:
	.section	.debug_str_offsets,"",@progbits
	.long	56                              ; Length of String Offsets Set
	.short	5
	.short	0
.Lstr_offsets_base0:
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"AMD clang version 22.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.2.0 26014 7b800a19466229b8479a78de19143dc33c3ab9b5)" ; string offset=0
.Linfo_string1:
	.asciz	"fmha_host.cpp"                 ; string offset=137
.Linfo_string2:
	.asciz	"/workspace/fmha"               ; string offset=151
.Linfo_string3:
	.asciz	"__hip_get_block_idx_z"         ; string offset=167
.Linfo_string4:
	.asciz	"__get_z"                       ; string offset=189
.Linfo_string5:
	.asciz	"__work_group_barrier"          ; string offset=197
.Linfo_string6:
	.asciz	"__barrier"                     ; string offset=218
.Linfo_string7:
	.asciz	"__syncthreads"                 ; string offset=228
.Linfo_string8:
	.asciz	"__shfl_xor"                    ; string offset=242
.Linfo_string9:
	.asciz	"fmaxf"                         ; string offset=253
.Linfo_string10:
	.asciz	"expf"                          ; string offset=259
.Linfo_string11:
	.asciz	"fmha_mfma"                     ; string offset=264
.Linfo_string12:
	.asciz	"fmha_mfma_coalesced"           ; string offset=274
	.section	.debug_str_offsets,"",@progbits
	.long	.Linfo_string0
	.long	.Linfo_string1
	.long	.Linfo_string2
	.long	.Linfo_string3
	.long	.Linfo_string4
	.long	.Linfo_string5
	.long	.Linfo_string6
	.long	.Linfo_string7
	.long	.Linfo_string8
	.long	.Linfo_string9
	.long	.Linfo_string10
	.long	.Linfo_string11
	.long	.Linfo_string12
	.section	.debug_addr,"",@progbits
	.long	.Ldebug_addr_end0-.Ldebug_addr_start0 ; Length of contribution
.Ldebug_addr_start0:
	.short	5                               ; DWARF version number
	.byte	8                               ; Address size
	.byte	0                               ; Segment selector size
.Laddr_table_base0:
	.quad	.Lfunc_begin0
	.quad	.Ltmp4
	.quad	.Ltmp6
	.quad	.Ltmp22
	.quad	.Ltmp32
	.quad	.Ltmp34
	.quad	.Lfunc_begin1
	.quad	.Ltmp38
	.quad	.Ltmp40
	.quad	.Ltmp61
	.quad	.Ltmp63
.Ldebug_addr_end0:
	.ident	"AMD clang version 22.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.2.0 26014 7b800a19466229b8479a78de19143dc33c3ab9b5)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_124cf60105eb9515
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     0
    .args:
      - .actual_access:  read_only
        .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  write_only
        .address_space:  global
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         32
        .size:           8
        .value_kind:     global_buffer
      - .offset:         40
        .size:           4
        .value_kind:     by_value
      - .offset:         44
        .size:           4
        .value_kind:     by_value
      - .offset:         48
        .size:           4
        .value_kind:     by_value
      - .offset:         52
        .size:           4
        .value_kind:     by_value
      - .offset:         56
        .size:           4
        .value_kind:     by_value
      - .offset:         60
        .size:           4
        .value_kind:     by_value
      - .offset:         64
        .size:           4
        .value_kind:     by_value
    .group_segment_fixed_size: 9856
    .kernarg_segment_align: 8
    .kernarg_segment_size: 68
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           _Z9fmha_mfmaPKDF16bS0_S0_PDF16_PKiiiiiiif
    .private_segment_fixed_size: 0
    .sgpr_count:     35
    .sgpr_spill_count: 0
    .symbol:         _Z9fmha_mfmaPKDF16bS0_S0_PDF16_PKiiiiiiif.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     28
    .vgpr_spill_count: 0
    .wavefront_size: 64
  - .agpr_count:     0
    .args:
      - .actual_access:  read_only
        .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  write_only
        .address_space:  global
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         32
        .size:           8
        .value_kind:     global_buffer
      - .offset:         40
        .size:           4
        .value_kind:     by_value
      - .offset:         44
        .size:           4
        .value_kind:     by_value
      - .offset:         48
        .size:           4
        .value_kind:     by_value
      - .offset:         52
        .size:           4
        .value_kind:     by_value
      - .offset:         56
        .size:           4
        .value_kind:     by_value
      - .offset:         60
        .size:           4
        .value_kind:     by_value
      - .offset:         64
        .size:           4
        .value_kind:     by_value
    .group_segment_fixed_size: 8352
    .kernarg_segment_align: 8
    .kernarg_segment_size: 68
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           _Z19fmha_mfma_coalescedPKDF16bS0_S0_PDF16_PKiiiiiiif
    .private_segment_fixed_size: 0
    .sgpr_count:     36
    .sgpr_spill_count: 0
    .symbol:         _Z19fmha_mfma_coalescedPKDF16bS0_S0_PDF16_PKiiiiiiif.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     25
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx942
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
	.section	.debug_line,"",@progbits
.Lline_table_start0:
