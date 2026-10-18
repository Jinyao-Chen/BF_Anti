cmake_minimum_required(VERSION 3.13)
include($ENV{PICO_SDK_PATH}/external/pico_sdk_import.cmake)

project(pico_pal C CXX ASM)
pico_sdk_init()

add_executable(pico_pal main.c)

pico_generate_pio_header(pico_pal ${CMAKE_CURRENT_LIST_DIR}/pal_video.pio)

target_link_libraries(pico_pal
    pico_stdlib
    hardware_pio
    hardware_spi
    hardware_dma
    hardware_clocks
)

pico_add_extra_outputs(pico_pal)
