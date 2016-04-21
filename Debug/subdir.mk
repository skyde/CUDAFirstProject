################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../Element.cu \
../Layer.cu \
../Main.cu \
../NeuralNetwork.cu \
../Node.cu \
../SharedData.cu 

OBJS += \
./Element.o \
./Layer.o \
./Main.o \
./NeuralNetwork.o \
./Node.o \
./SharedData.o 

CU_DEPS += \
./Element.d \
./Layer.d \
./Main.d \
./NeuralNetwork.d \
./Node.d \
./SharedData.d 


# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/Developer/NVIDIA/CUDA-7.5/bin/nvcc -G -g -O0 -gencode arch=compute_30,code=sm_30  -odir "." -M -o "$(@:%.o=%.d)" "$<"
	/Developer/NVIDIA/CUDA-7.5/bin/nvcc -G -g -O0 --compile --relocatable-device-code=false -gencode arch=compute_30,code=compute_30 -gencode arch=compute_30,code=sm_30  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


