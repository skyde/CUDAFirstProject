################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../inc/cuda_drvapi_dynlink.cpp 

OBJS += \
./inc/cuda_drvapi_dynlink.o 

CPP_DEPS += \
./inc/cuda_drvapi_dynlink.d 


# Each subdirectory must supply rules for building sources it contributes
inc/%.o: ../inc/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/Developer/NVIDIA/CUDA-7.5/bin/nvcc -G -g -O0 -gencode arch=compute_30,code=sm_30  -odir "inc" -M -o "$(@:%.o=%.d)" "$<"
	/Developer/NVIDIA/CUDA-7.5/bin/nvcc -G -g -O0 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


