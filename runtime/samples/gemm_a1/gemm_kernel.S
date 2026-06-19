    # Regalloc stats: {"preallocated_float": ["ft0", "ft1"], "preallocated_int": ["a0", "a1", "a2", "zero"], "excluded_float": ["ft0", "ft1", "ft2"], "excluded_int": [], "allocated_float": ["ft0", "ft1", "ft3", "ft4", "ft5", "ft6"], "allocated_int": ["a0", "a1", "a2", "a3", "a4", "a5", "a6", "t0", "t1", "t2", "t3", "t4", "t5", "t6", "zero"]}
.text
.globl gemm_kernel
.p2align 2
gemm_kernel:
    mv t3, a0
    mv t4, a1
    mv t1, a2
    li t0, 15
    scfgwi t0, 64                                # dm 0 dim 0 bound
    li t6, 3
    scfgwi t6, 96                                # dm 0 dim 1 bound
    scfgwi t0, 128                               # dm 0 dim 2 bound
    li t2, 8
    scfgwi t2, 192                               # dm 0 dim 0 stride
    li t5, -120
    scfgwi t5, 224                               # dm 0 dim 1 stride
    scfgwi t2, 256                               # dm 0 dim 2 stride
    scfgwi t6, 32                                # dm 0 repeat
    scfgwi t6, 65                                # dm 1 dim 0 bound
    scfgwi t0, 97                                # dm 1 dim 1 bound
    scfgwi t6, 129                               # dm 1 dim 2 bound
    scfgwi t0, 161                               # dm 1 dim 3 bound
    scfgwi t2, 193                               # dm 1 dim 0 stride
    li t6, 104
    scfgwi t6, 225                               # dm 1 dim 1 stride
    li t6, -1912
    scfgwi t6, 257                               # dm 1 dim 2 stride
    li t6, -2040
    scfgwi t6, 289                               # dm 1 dim 3 stride
    scfgwi zero, 33                              # dm 1 repeat
    scfgwi t3, 832                               # dm 0 dim 2 source
    scfgwi t4, 865                               # dm 1 dim 3 source
    csrrsi zero, 1984, 1                         # SSR enable
    li t4, 256
    mv t3, zero
    # Constant folded riscv_cf.bge
scf_body_1_for:
    li t6, 16
    mv t5, zero
    # Constant folded riscv_cf.bge
scf_body_0_for:
    mv a6, t5
    add a6, t3, a6
    mul a6, a6, t2                               # multiply by element size
    add a6, t1, a6
    fld ft6, 0(a6)                               # load double from memref of shape (16, 16)
    addi a5, t5, 1
    add a5, t3, a5
    mul a5, a5, t2                               # multiply by element size
    add a5, t1, a5
    fld ft5, 0(a5)                               # load double from memref of shape (16, 16)
    addi a4, t5, 2
    add a4, t3, a4
    mul a4, a4, t2                               # multiply by element size
    add a4, t1, a4
    fld ft4, 0(a4)                               # load double from memref of shape (16, 16)
    addi a3, t5, 3
    add a3, t3, a3
    mul a3, a3, t2                               # multiply by element size
    add a3, t1, a3
    fld ft3, 0(a3)                               # load double from memref of shape (16, 16)
    frep.o t0, 4, 0, 0
    fmadd.d ft6, ft0, ft1, ft6
    fmadd.d ft5, ft0, ft1, ft5
    fmadd.d ft4, ft0, ft1, ft4
    fmadd.d ft3, ft0, ft1, ft3
    fsd ft6, 0(a6)                               # store double value to memref of shape (16, 16)
    fsd ft5, 0(a5)                               # store double value to memref of shape (16, 16)
    fsd ft4, 0(a4)                               # store double value to memref of shape (16, 16)
    fsd ft3, 0(a3)                               # store double value to memref of shape (16, 16)
    addi t5, t5, 4
    blt t5, t6, scf_body_0_for
scf_body_end_0_for:
    addi t3, t3, 16
    blt t3, t4, scf_body_1_for
scf_body_end_1_for:
    csrrci zero, 1984, 1                         # SSR disable
    ret
