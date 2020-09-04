cmake_minimum_required (VERSION 3.13)

project (promort_pipeline_ecvl_eddl CXX)

include(CheckCXXCompilerFlag)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER "")

#--------------------------------------------------------------------------------------------------------
# BEGIN: Check for c++17 support
#--------------------------------------------------------------------------------------------------------
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    check_cxx_compiler_flag("-std=c++17" COMPILER_SUPPORTS_CXX17)
elseif(MSVC)
    check_cxx_compiler_flag("/std:c++17" COMPILER_SUPPORTS_CXX17)
else()
    # Unrecognized compiler
    set(COMPILER_SUPPORTS_CXX17 ON)
endif()

if(NOT COMPILER_SUPPORTS_CXX17)
    message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++17 support. Please use a different C++ compiler.")
endif()

set(CMAKE_CXX_EXTENSIONS OFF)
#--------------------------------------------------------------------------------------------------------
# END: Check for c++17 support
#--------------------------------------------------------------------------------------------------------

# SET BUILD TYPE (RELEASE/DEBUG)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(MSVC)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()

#--------------------------------------------------------------------------------------------------------
# BEGIN: static/dynamic runtime
#--------------------------------------------------------------------------------------------------------
set(pipeline_ecvl_eddl_STATIC ON CACHE BOOL "Whether to build static or dynamic libraries")

if(MSVC)
    if(pipeline_ecvl_eddl_STATIC)
        message(STATUS "Setting static runtime for MSVC")
        foreach(flag_var CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
            string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
        endforeach(flag_var)
    else()
        message(STATUS "Setting dynamic runtime for MSVC")
        foreach(flag_var CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
            string(REGEX REPLACE "/MT" "/MD" ${flag_var} "${${flag_var}}")
        endforeach(flag_var)
    endif()
endif()
#--------------------------------------------------------------------------------------------------------
# END: static/dynamic runtime
#--------------------------------------------------------------------------------------------------------

add_library(METRICS STATIC "")
add_library(MODELS STATIC "")
add_library(DATA_GENERATOR STATIC "")
add_executable(PROMORT_CLASSIFICATION_TRAINING)
add_subdirectory(cpp)
add_subdirectory("cpp/metrics")
add_subdirectory("cpp/models")
add_subdirectory("cpp/data_generator")

# Set SKIN_LESION_CLASSIFICATION_TRAINING as default
set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT PROMORT_CLASSIFICATION_TRAINING)

set(ECVL_BUILD ON CACHE BOOL "Whether to build ECVL or not (requires ECVL)")

find_package(ecvl)
if(ecvl_FOUND)
    # Enable c++17 features
    target_compile_features(METRICS PUBLIC cxx_std_17)
    target_compile_features(PROMORT_CLASSIFICATION_TRAINING PUBLIC cxx_std_17)

    target_link_libraries(METRICS ECVL_CORE ECVL_EDDL DATASET)
    target_link_libraries(DATA_GENERATOR METRICS)
    target_link_libraries(MODELS DATA_GENERATOR)
    target_link_libraries(PROMORT_CLASSIFICATION_TRAINING MODELS)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
	    target_link_libraries(PROMORT_CLASSIFICATION_TRAINING stdc++fs)
    endif()
    set_target_properties(METRICS PROPERTIES FOLDER "Utils")
    set_target_properties(MODELS PROPERTIES FOLDER "Utils")
    set_target_properties(DATA_GENERATOR PROPERTIES FOLDER "Utils")

    include_directories("${ecvl_DIR}/include")
    message(STATUS "ECVL found")
endif()

#--------------------------------------------------------------------------------------------------------
# BEGIN: Summary
#--------------------------------------------------------------------------------------------------------
message(STATUS "--------------------------------------")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ flags: ${CMAKE_CXX_FLAGS}")
message(STATUS "C++ Release flags: ${CMAKE_CXX_FLAGS_RELEASE}")
message(STATUS "C++ Debug flags: ${CMAKE_CXX_FLAGS_DEBUG}")
#--------------------------------------------------------------------------------------------------------
# END: Summary
#--------------------------------------------------------------------------------------------------------
