project(snRuntime C CXX ASM)

find_package(Python3 REQUIRED)
find_program(PEAKRDL_PATH NAMES peakrdl REQUIRED)

set(snRuntimeSrc ${SNITCH_CLUSTER_SOURCE_DIR})
set(header_dir ${CMAKE_CURRENT_BINARY_DIR}/cluster_gen)
set(config_file ${snRuntimeSrc}/cfg/default.json)
set(clustergen ${snRuntimeSrc}/util/clustergen/clustergen.py)
set(clustergen_src ${snRuntimeSrc}/util/clustergen/cluster.py)
set(periph_dir ${snRuntimeSrc}/hw/snitch_cluster/src/snitch_cluster_peripheral)
set(periph_rdl ${periph_dir}/snitch_cluster_peripheral_reg.rdl)
set(runtime_impl ${snRuntimeSrc}/sw/runtime/impl)

file(MAKE_DIRECTORY ${header_dir})

function(clustergen_render out template)
  add_custom_command(OUTPUT ${out}
      COMMAND ${Python3_EXECUTABLE} ${clustergen} -c ${config_file} -o ${out} --template ${template}
      DEPENDS ${config_file} ${clustergen} ${clustergen_src} ${template}
      VERBATIM)
endfunction()

clustergen_render(${header_dir}/snitch_cluster.rdl
    ${snRuntimeSrc}/hw/snitch_cluster/src/snitch_cluster.rdl.tpl)
clustergen_render(${header_dir}/snitch_cluster_addrmap.rdl
    ${runtime_impl}/snitch_cluster_addrmap.rdl.tpl)
clustergen_render(${header_dir}/snitch_cluster_cfg.h
    ${runtime_impl}/snitch_cluster_cfg.h.tpl)

# Expose the cfg-generated cluster header so the codegen target (iree-compile, via
# quidditch_module) can derive its HW params from the SAME cfg the runtime is built
# from, instead of baked literals (--iree-quidditch-cluster-cfg-header).
set(QUIDDITCH_CLUSTER_CFG_HEADER ${header_dir}/snitch_cluster_cfg.h
    CACHE INTERNAL "cfg-generated snitch_cluster_cfg.h for the Quidditch codegen target")

set(peakrdl_inc -I ${periph_dir} -I ${header_dir})

add_custom_command(OUTPUT ${header_dir}/snitch_cluster_peripheral.h
    COMMAND ${PEAKRDL_PATH} c-header ${periph_rdl}
            -o ${header_dir}/snitch_cluster_peripheral.h -i -b ltoh
    DEPENDS ${periph_rdl}
    VERBATIM)

add_custom_command(OUTPUT ${header_dir}/snitch_cluster_peripheral_addrmap.h
    COMMAND ${PEAKRDL_PATH} raw-header ${periph_rdl}
            -o ${header_dir}/snitch_cluster_peripheral_addrmap.h --format c ${peakrdl_inc}
    DEPENDS ${periph_rdl} ${header_dir}/snitch_cluster.rdl
    VERBATIM)

add_custom_command(OUTPUT ${header_dir}/snitch_cluster_addrmap.h
    COMMAND ${PEAKRDL_PATH} c-header ${header_dir}/snitch_cluster_addrmap.rdl
            -o ${header_dir}/snitch_cluster_addrmap.h -i -b ltoh ${peakrdl_inc}
    DEPENDS ${header_dir}/snitch_cluster_addrmap.rdl ${header_dir}/snitch_cluster.rdl
    VERBATIM)

add_custom_command(OUTPUT ${header_dir}/snitch_cluster_raw_addrmap.h
    COMMAND ${PEAKRDL_PATH} raw-header ${header_dir}/snitch_cluster_addrmap.rdl
            -o ${header_dir}/snitch_cluster_raw_addrmap.h
            --base_name snitch_cluster_raw_addrmap --format c ${peakrdl_inc}
    DEPENDS ${header_dir}/snitch_cluster_addrmap.rdl ${header_dir}/snitch_cluster.rdl
    VERBATIM)

add_custom_target(snRuntimeConfig DEPENDS
    ${header_dir}/snitch_cluster_cfg.h
    ${header_dir}/snitch_cluster_addrmap.h
    ${header_dir}/snitch_cluster_raw_addrmap.h
    ${header_dir}/snitch_cluster_peripheral.h
    ${header_dir}/snitch_cluster_peripheral_addrmap.h)

add_library(snRuntimeInterface INTERFACE)
target_include_directories(snRuntimeInterface
    INTERFACE
    ${snRuntimeSrc}/sw/runtime/api
    ${snRuntimeSrc}/sw/deps/riscv-opcodes
    ${CMAKE_CURRENT_LIST_DIR}/api
)
target_compile_options(snRuntimeInterface INTERFACE -Wno-undefined-inline)

add_library(snRuntime
    rtl/src/snitch_cluster_start.S
    rtl/src/snrt.cc
    rtl/src/quidditch_snrt_exports.cc
    rtl/src/syscalls.c
)
target_link_libraries(snRuntime PUBLIC snRuntimeInterface)
target_include_directories(snRuntime
    PRIVATE
    ${snRuntimeSrc}/sw/runtime/src
    ${snRuntimeSrc}/sw/runtime/impl
    ${header_dir}
)
add_dependencies(snRuntime snRuntimeConfig)
target_link_options(snRuntime INTERFACE -Tbase.ld)
target_link_directories(snRuntime
    INTERFACE
    ${snRuntimeSrc}/sw/runtime
    rtl
)
