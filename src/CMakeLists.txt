cmake_minimum_required(VERSION 3.7)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

add_library(mcrl SHARED ${HEADER_FILES} mcrl.cpp)
target_include_directories (mcrl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})


install(TARGETS mcrl
	RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

