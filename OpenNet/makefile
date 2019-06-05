
# Author     KMS - Martin Dubois, ing.
# Copyright  (C) 2018-2019 KMS. All rights reserved.
# Product    OpenNet
# File       OpenNet/makefile

include ../User.mk

INCLUDES = -I ../Includes $(INCLUDE_IMPORT) -I $(CUDA_FOLDER)/include

LIBRARIES = $(KMS_LIB_A)

OUTPUT = ../Binaries/libOpenNet.so

SOURCES =	Adapter.cpp                  \
			Buffer.cpp                   \
			CUDA/Buffer_CUDA.cpp              \
			CUDA/CUDAW.cpp                    \
			CUDA/CUW.cpp                      \
			CUDA/Event_CUDA.cpp               \
			CUDA/NVRTCW.cpp                   \
			CUDA/Processor_CUDA.cpp           \
			CUDA/System_CUDA.cpp              \
			CUDA/Thread_CUDA.cpp              \
			CUDA/Thread_Functions_CUDA.cpp    \
			CUDA/Thread_Kernel_CUDA.cpp       \
			CUDA/UserBuffer_CUDA.cpp          \
			EthernetAddress.cpp          \
			Event.cpp                    \
			FolderFinder.cpp             \
			Function.cpp                 \
			Function_Forward.cpp         \
			Internal/Adapter_Internal.cpp \
			Internal/Buffer_Internal.cpp  \
			Internal/PacketGenerator_Internal.cpp \
			Internal/Processor_Internal.cpp       \
			Internal/SetupTool_Internal.cpp       \
			Internal/System_Internal.cpp          \
			Internal/UserBuffer_Internal.cpp      \
			Kernel.cpp                   \
			Kernel_Forward.cpp           \
			Kernel_Functions.cpp		 \
			Linux/Adapter_Linux.cpp      \
			Linux/FolderFinder_Linux.cpp \
			Linux/SetupTool_Linux.cpp    \
			PacketGenerator.cpp          \
			Processor.cpp                \
			SetupTool.cpp                \
			SourceCode.cpp			     \
			SourceCode_Forward.cpp       \
			StatisticsProvider.cpp       \
			Status.cpp                   \
			System.cpp                   \
			Thread.cpp                   \
			Thread_Functions.cpp		 \
			Thread_Kernel.cpp            \
			UserBuffer.cpp               \
			Utils.cpp                    \
			VersionInfo.cpp

# ===== Rules / Regles ======================================================

.cpp.o:
	g++ -c $(COMPILE_FLAGS) -o $@ $(INCLUDES) $<

# ===== Macros ==============================================================

OBJECTS = $(SOURCES:.cpp=.o)

# ===== Targets ==============================================================

$(OUTPUT) : $(OBJECTS) $(LIBRARIES)
	g++ -L $(CUDA_FOLDER)/lib64 -l cudart -shared -o $@ $^

depend:
	makedepend -Y $(INCLUDES) $(SOURCES)
	
# DO NOT DELETE

Adapter.o: Component.h ../Import/Includes/KmsBase.h
Adapter.o: ../Import/Includes/SafeAPI.h ../Import/Includes/WindowsToLinux.h
Adapter.o: ../Import/Includes/KmsLib/ValueVector.h
Adapter.o: ../Includes/OpenNet/Adapter.h ../Includes/OpenNet/Processor.h
Adapter.o: ../Includes/OpenNet/OpenNet.h ../Includes/OpenNet/Status.h
Adapter.o: ../Includes/OpenNet/StatisticsProvider.h
Adapter.o: ../Includes/OpenNetK/Adapter_Types.h
Adapter.o: ../Includes/OpenNetK/Constants.h ../Common/Constants.h
Adapter.o: ../Common/IoCtl.h EthernetAddress.h
Adapter.o: ../Common/OpenNet/EthernetAddress.h VersionInfo.h
Buffer.o: Component.h ../Import/Includes/KmsBase.h
Buffer.o: ../Import/Includes/SafeAPI.h ../Import/Includes/WindowsToLinux.h
Buffer.o: ../Includes/OpenNet/Buffer.h ../Includes/OpenNet/Status.h
Buffer.o: ../Includes/OpenNet/OpenNet.h
CUDA/Buffer_CUDA.o: Component.h ../Import/Includes/KmsBase.h
CUDA/Buffer_CUDA.o: ../Import/Includes/SafeAPI.h
CUDA/Buffer_CUDA.o: ../Import/Includes/WindowsToLinux.h CUDA/CUW.h
CUDA/Buffer_CUDA.o: /usr/local/cuda-10.0/include/cuda.h CUDA/Buffer_CUDA.h
CUDA/Buffer_CUDA.o: Internal/Buffer_Internal.h ../Includes/OpenNet/Buffer.h
CUDA/Buffer_CUDA.o: ../Includes/OpenNet/Status.h
CUDA/Buffer_CUDA.o: ../Includes/OpenNet/OpenNet.h
CUDA/Buffer_CUDA.o: ../Includes/OpenNetK/Types.h CUDA/Event_CUDA.h Event.h
CUDA/CUDAW.o: ../Import/Includes/KmsLib/Exception.h CUDA/CUDAW.h
CUDA/CUDAW.o: /usr/local/cuda-10.0/include/cuda_runtime.h
CUDA/CUDAW.o: /usr/local/cuda-10.0/include/crt/host_config.h
CUDA/CUDAW.o: /usr/local/cuda-10.0/include/builtin_types.h
CUDA/CUDAW.o: /usr/local/cuda-10.0/include/device_types.h
CUDA/CUDAW.o: /usr/local/cuda-10.0/include/crt/host_defines.h
CUDA/CUDAW.o: /usr/local/cuda-10.0/include/driver_types.h
CUDA/CUDAW.o: /usr/local/cuda-10.0/include/vector_types.h
CUDA/CUDAW.o: /usr/local/cuda-10.0/include/surface_types.h
CUDA/CUDAW.o: /usr/local/cuda-10.0/include/texture_types.h
CUDA/CUDAW.o: /usr/local/cuda-10.0/include/library_types.h
CUDA/CUDAW.o: /usr/local/cuda-10.0/include/channel_descriptor.h
CUDA/CUDAW.o: /usr/local/cuda-10.0/include/cuda_runtime_api.h
CUDA/CUDAW.o: /usr/local/cuda-10.0/include/cuda_device_runtime_api.h
CUDA/CUDAW.o: /usr/local/cuda-10.0/include/driver_functions.h
CUDA/CUDAW.o: /usr/local/cuda-10.0/include/vector_functions.h
CUDA/CUDAW.o: /usr/local/cuda-10.0/include/vector_functions.hpp
CUDA/CUW.o: Component.h ../Import/Includes/KmsBase.h
CUDA/CUW.o: ../Import/Includes/SafeAPI.h ../Import/Includes/WindowsToLinux.h
CUDA/CUW.o: ../Import/Includes/KmsLib/Exception.h CUDA/CUW.h
CUDA/CUW.o: /usr/local/cuda-10.0/include/cuda.h
CUDA/Event_CUDA.o: Component.h ../Import/Includes/KmsBase.h
CUDA/Event_CUDA.o: ../Import/Includes/SafeAPI.h
CUDA/Event_CUDA.o: ../Import/Includes/WindowsToLinux.h CUDA/CUW.h
CUDA/Event_CUDA.o: /usr/local/cuda-10.0/include/cuda.h CUDA/Event_CUDA.h
CUDA/Event_CUDA.o: Event.h
CUDA/NVRTCW.o: ../Import/Includes/KmsLib/Exception.h CUDA/NVRTCW.h
CUDA/NVRTCW.o: /usr/local/cuda-10.0/include/nvrtc.h
CUDA/Processor_CUDA.o: Component.h ../Import/Includes/KmsBase.h
CUDA/Processor_CUDA.o: ../Import/Includes/SafeAPI.h
CUDA/Processor_CUDA.o: ../Import/Includes/WindowsToLinux.h
CUDA/Processor_CUDA.o: ../Includes/OpenNetK/Types.h ../Common/Constants.h
CUDA/Processor_CUDA.o: FolderFinder.h CUDA/Buffer_CUDA.h
CUDA/Processor_CUDA.o: /usr/local/cuda-10.0/include/cuda.h
CUDA/Processor_CUDA.o: Internal/Buffer_Internal.h
CUDA/Processor_CUDA.o: ../Includes/OpenNet/Buffer.h
CUDA/Processor_CUDA.o: ../Includes/OpenNet/Status.h
CUDA/Processor_CUDA.o: ../Includes/OpenNet/OpenNet.h CUDA/Event_CUDA.h
CUDA/Processor_CUDA.o: Event.h CUDA/CUW.h CUDA/NVRTCW.h
CUDA/Processor_CUDA.o: /usr/local/cuda-10.0/include/nvrtc.h
CUDA/Processor_CUDA.o: CUDA/Thread_Functions_CUDA.h Thread_Functions.h
CUDA/Processor_CUDA.o: ../Includes/OpenNet/Function.h
CUDA/Processor_CUDA.o: ../Includes/OpenNet/SourceCode.h Kernel_Functions.h
CUDA/Processor_CUDA.o: ../Includes/OpenNet/Kernel.h
CUDA/Processor_CUDA.o: ../Includes/OpenNet/StatisticsProvider.h Thread.h
CUDA/Processor_CUDA.o: ../Import/Includes/KmsLib/ThreadBase.h
CUDA/Processor_CUDA.o: Internal/Adapter_Internal.h
CUDA/Processor_CUDA.o: ../Import/Includes/KmsLib/DebugLog.h
CUDA/Processor_CUDA.o: ../Import/Includes/KmsLib/Exception.h
CUDA/Processor_CUDA.o: ../Import/Includes/KmsLib/TextFile.h
CUDA/Processor_CUDA.o: ../Import/Includes/KmsLib/DriverHandle.h
CUDA/Processor_CUDA.o: ../Import/Includes/KmsLib/FileHandle.h
CUDA/Processor_CUDA.o: ../Includes/OpenNet/Adapter.h
CUDA/Processor_CUDA.o: ../Includes/OpenNet/Processor.h
CUDA/Processor_CUDA.o: ../Includes/OpenNetK/Adapter_Types.h
CUDA/Processor_CUDA.o: ../Includes/OpenNet/System.h
CUDA/Processor_CUDA.o: ../Includes/OpenNetK/Constants.h
CUDA/Processor_CUDA.o: ../Includes/OpenNetK/PacketGenerator_Types.h
CUDA/Processor_CUDA.o: ../Common/IoCtl.h
CUDA/Processor_CUDA.o: ../Common/OpenNet/Adapter_Statistics.h
CUDA/Processor_CUDA.o: Internal/Buffer_Internal.h
CUDA/Processor_CUDA.o: Internal/Processor_Internal.h
CUDA/Processor_CUDA.o: ../Includes/OpenNet/UserBuffer.h
CUDA/Processor_CUDA.o: Internal/Processor_Internal.h CUDA/Thread_CUDA.h
CUDA/Processor_CUDA.o: Internal/Adapter_Internal.h CUDA/Processor_CUDA.h
CUDA/Processor_CUDA.o: Internal/Processor_Internal.h CUDA/UserBuffer_CUDA.h
CUDA/Processor_CUDA.o: Internal/UserBuffer_Internal.h
CUDA/System_CUDA.o: Component.h ../Import/Includes/KmsBase.h
CUDA/System_CUDA.o: ../Import/Includes/SafeAPI.h
CUDA/System_CUDA.o: ../Import/Includes/WindowsToLinux.h ../Common/Constants.h
CUDA/System_CUDA.o: Linux/Adapter_Linux.h /usr/local/cuda-10.0/include/cuda.h
CUDA/System_CUDA.o: Internal/Adapter_Internal.h
CUDA/System_CUDA.o: ../Import/Includes/KmsLib/DebugLog.h
CUDA/System_CUDA.o: ../Import/Includes/KmsLib/Exception.h
CUDA/System_CUDA.o: ../Import/Includes/KmsLib/TextFile.h
CUDA/System_CUDA.o: ../Import/Includes/KmsLib/DriverHandle.h
CUDA/System_CUDA.o: ../Import/Includes/KmsLib/FileHandle.h
CUDA/System_CUDA.o: ../Import/Includes/KmsLib/ThreadBase.h
CUDA/System_CUDA.o: ../Includes/OpenNet/Adapter.h
CUDA/System_CUDA.o: ../Includes/OpenNet/Processor.h
CUDA/System_CUDA.o: ../Includes/OpenNet/OpenNet.h
CUDA/System_CUDA.o: ../Includes/OpenNet/Status.h
CUDA/System_CUDA.o: ../Includes/OpenNet/StatisticsProvider.h
CUDA/System_CUDA.o: ../Includes/OpenNetK/Adapter_Types.h
CUDA/System_CUDA.o: ../Includes/OpenNet/System.h
CUDA/System_CUDA.o: ../Includes/OpenNetK/Constants.h
CUDA/System_CUDA.o: ../Includes/OpenNetK/PacketGenerator_Types.h
CUDA/System_CUDA.o: ../Common/IoCtl.h ../Common/OpenNet/Adapter_Statistics.h
CUDA/System_CUDA.o: Internal/Buffer_Internal.h ../Includes/OpenNet/Buffer.h
CUDA/System_CUDA.o: ../Includes/OpenNetK/Types.h
CUDA/System_CUDA.o: Internal/Processor_Internal.h
CUDA/System_CUDA.o: ../Includes/OpenNet/Kernel.h
CUDA/System_CUDA.o: ../Includes/OpenNet/SourceCode.h
CUDA/System_CUDA.o: ../Includes/OpenNet/UserBuffer.h CUDA/CUW.h
CUDA/System_CUDA.o: CUDA/Processor_CUDA.h
CUDA/System_CUDA.o: /usr/local/cuda-10.0/include/nvrtc.h
CUDA/System_CUDA.o: Internal/Processor_Internal.h CUDA/System_CUDA.h
CUDA/System_CUDA.o: Internal/System_Internal.h
CUDA/Thread_CUDA.o: Component.h ../Import/Includes/KmsBase.h
CUDA/Thread_CUDA.o: ../Import/Includes/SafeAPI.h
CUDA/Thread_CUDA.o: ../Import/Includes/WindowsToLinux.h Linux/Adapter_Linux.h
CUDA/Thread_CUDA.o: /usr/local/cuda-10.0/include/cuda.h
CUDA/Thread_CUDA.o: Internal/Adapter_Internal.h
CUDA/Thread_CUDA.o: ../Import/Includes/KmsLib/DebugLog.h
CUDA/Thread_CUDA.o: ../Import/Includes/KmsLib/Exception.h
CUDA/Thread_CUDA.o: ../Import/Includes/KmsLib/TextFile.h
CUDA/Thread_CUDA.o: ../Import/Includes/KmsLib/DriverHandle.h
CUDA/Thread_CUDA.o: ../Import/Includes/KmsLib/FileHandle.h
CUDA/Thread_CUDA.o: ../Import/Includes/KmsLib/ThreadBase.h
CUDA/Thread_CUDA.o: ../Includes/OpenNet/Adapter.h
CUDA/Thread_CUDA.o: ../Includes/OpenNet/Processor.h
CUDA/Thread_CUDA.o: ../Includes/OpenNet/OpenNet.h
CUDA/Thread_CUDA.o: ../Includes/OpenNet/Status.h
CUDA/Thread_CUDA.o: ../Includes/OpenNet/StatisticsProvider.h
CUDA/Thread_CUDA.o: ../Includes/OpenNetK/Adapter_Types.h
CUDA/Thread_CUDA.o: ../Includes/OpenNet/System.h
CUDA/Thread_CUDA.o: ../Includes/OpenNetK/Constants.h
CUDA/Thread_CUDA.o: ../Includes/OpenNetK/PacketGenerator_Types.h
CUDA/Thread_CUDA.o: ../Common/IoCtl.h ../Common/OpenNet/Adapter_Statistics.h
CUDA/Thread_CUDA.o: Internal/Buffer_Internal.h ../Includes/OpenNet/Buffer.h
CUDA/Thread_CUDA.o: ../Includes/OpenNetK/Types.h
CUDA/Thread_CUDA.o: Internal/Processor_Internal.h
CUDA/Thread_CUDA.o: ../Includes/OpenNet/Kernel.h
CUDA/Thread_CUDA.o: ../Includes/OpenNet/SourceCode.h
CUDA/Thread_CUDA.o: ../Includes/OpenNet/UserBuffer.h CUDA/CUW.h
CUDA/Thread_CUDA.o: CUDA/Event_CUDA.h Event.h CUDA/Thread_CUDA.h
CUDA/Thread_CUDA.o: CUDA/Processor_CUDA.h
CUDA/Thread_CUDA.o: /usr/local/cuda-10.0/include/nvrtc.h
CUDA/Thread_CUDA.o: Internal/Processor_Internal.h
CUDA/Thread_Functions_CUDA.o: Component.h ../Import/Includes/KmsBase.h
CUDA/Thread_Functions_CUDA.o: ../Import/Includes/SafeAPI.h
CUDA/Thread_Functions_CUDA.o: ../Import/Includes/WindowsToLinux.h
CUDA/Thread_Functions_CUDA.o: ../Common/Constants.h CUDA/Buffer_CUDA.h
CUDA/Thread_Functions_CUDA.o: /usr/local/cuda-10.0/include/cuda.h
CUDA/Thread_Functions_CUDA.o: Internal/Buffer_Internal.h
CUDA/Thread_Functions_CUDA.o: ../Includes/OpenNet/Buffer.h
CUDA/Thread_Functions_CUDA.o: ../Includes/OpenNet/Status.h
CUDA/Thread_Functions_CUDA.o: ../Includes/OpenNet/OpenNet.h
CUDA/Thread_Functions_CUDA.o: ../Includes/OpenNetK/Types.h CUDA/Event_CUDA.h
CUDA/Thread_Functions_CUDA.o: Event.h CUDA/CUW.h CUDA/Thread_Functions_CUDA.h
CUDA/Thread_Functions_CUDA.o: Thread_Functions.h
CUDA/Thread_Functions_CUDA.o: ../Includes/OpenNet/Function.h
CUDA/Thread_Functions_CUDA.o: ../Includes/OpenNet/SourceCode.h
CUDA/Thread_Functions_CUDA.o: Kernel_Functions.h ../Includes/OpenNet/Kernel.h
CUDA/Thread_Functions_CUDA.o: ../Includes/OpenNet/StatisticsProvider.h
CUDA/Thread_Functions_CUDA.o: Thread.h ../Import/Includes/KmsLib/ThreadBase.h
CUDA/Thread_Functions_CUDA.o: Internal/Adapter_Internal.h
CUDA/Thread_Functions_CUDA.o: ../Import/Includes/KmsLib/DebugLog.h
CUDA/Thread_Functions_CUDA.o: ../Import/Includes/KmsLib/Exception.h
CUDA/Thread_Functions_CUDA.o: ../Import/Includes/KmsLib/TextFile.h
CUDA/Thread_Functions_CUDA.o: ../Import/Includes/KmsLib/DriverHandle.h
CUDA/Thread_Functions_CUDA.o: ../Import/Includes/KmsLib/FileHandle.h
CUDA/Thread_Functions_CUDA.o: ../Includes/OpenNet/Adapter.h
CUDA/Thread_Functions_CUDA.o: ../Includes/OpenNet/Processor.h
CUDA/Thread_Functions_CUDA.o: ../Includes/OpenNetK/Adapter_Types.h
CUDA/Thread_Functions_CUDA.o: ../Includes/OpenNet/System.h
CUDA/Thread_Functions_CUDA.o: ../Includes/OpenNetK/Constants.h
CUDA/Thread_Functions_CUDA.o: ../Includes/OpenNetK/PacketGenerator_Types.h
CUDA/Thread_Functions_CUDA.o: ../Common/IoCtl.h
CUDA/Thread_Functions_CUDA.o: ../Common/OpenNet/Adapter_Statistics.h
CUDA/Thread_Functions_CUDA.o: Internal/Buffer_Internal.h
CUDA/Thread_Functions_CUDA.o: Internal/Processor_Internal.h
CUDA/Thread_Functions_CUDA.o: ../Includes/OpenNet/UserBuffer.h
CUDA/Thread_Functions_CUDA.o: Internal/Processor_Internal.h
CUDA/Thread_Functions_CUDA.o: CUDA/Thread_CUDA.h Internal/Adapter_Internal.h
CUDA/Thread_Functions_CUDA.o: CUDA/Processor_CUDA.h
CUDA/Thread_Functions_CUDA.o: /usr/local/cuda-10.0/include/nvrtc.h
CUDA/Thread_Functions_CUDA.o: Internal/Processor_Internal.h
CUDA/Thread_Kernel_CUDA.o: Component.h ../Import/Includes/KmsBase.h
CUDA/Thread_Kernel_CUDA.o: ../Import/Includes/SafeAPI.h
CUDA/Thread_Kernel_CUDA.o: ../Import/Includes/WindowsToLinux.h
CUDA/Thread_Kernel_CUDA.o: CUDA/Buffer_CUDA.h
CUDA/Thread_Kernel_CUDA.o: /usr/local/cuda-10.0/include/cuda.h
CUDA/Thread_Kernel_CUDA.o: Internal/Buffer_Internal.h
CUDA/Thread_Kernel_CUDA.o: ../Includes/OpenNet/Buffer.h
CUDA/Thread_Kernel_CUDA.o: ../Includes/OpenNet/Status.h
CUDA/Thread_Kernel_CUDA.o: ../Includes/OpenNet/OpenNet.h
CUDA/Thread_Kernel_CUDA.o: ../Includes/OpenNetK/Types.h CUDA/Event_CUDA.h
CUDA/Thread_Kernel_CUDA.o: Event.h CUDA/CUDAW.h
CUDA/Thread_Kernel_CUDA.o: /usr/local/cuda-10.0/include/cuda_runtime.h
CUDA/Thread_Kernel_CUDA.o: /usr/local/cuda-10.0/include/crt/host_config.h
CUDA/Thread_Kernel_CUDA.o: /usr/local/cuda-10.0/include/builtin_types.h
CUDA/Thread_Kernel_CUDA.o: /usr/local/cuda-10.0/include/device_types.h
CUDA/Thread_Kernel_CUDA.o: /usr/local/cuda-10.0/include/crt/host_defines.h
CUDA/Thread_Kernel_CUDA.o: /usr/local/cuda-10.0/include/driver_types.h
CUDA/Thread_Kernel_CUDA.o: /usr/local/cuda-10.0/include/vector_types.h
CUDA/Thread_Kernel_CUDA.o: /usr/local/cuda-10.0/include/surface_types.h
CUDA/Thread_Kernel_CUDA.o: /usr/local/cuda-10.0/include/texture_types.h
CUDA/Thread_Kernel_CUDA.o: /usr/local/cuda-10.0/include/library_types.h
CUDA/Thread_Kernel_CUDA.o: /usr/local/cuda-10.0/include/channel_descriptor.h
CUDA/Thread_Kernel_CUDA.o: /usr/local/cuda-10.0/include/cuda_runtime_api.h
CUDA/Thread_Kernel_CUDA.o: /usr/local/cuda-10.0/include/cuda_device_runtime_api.h
CUDA/Thread_Kernel_CUDA.o: /usr/local/cuda-10.0/include/driver_functions.h
CUDA/Thread_Kernel_CUDA.o: /usr/local/cuda-10.0/include/vector_functions.h
CUDA/Thread_Kernel_CUDA.o: /usr/local/cuda-10.0/include/vector_functions.hpp
CUDA/Thread_Kernel_CUDA.o: CUDA/Processor_CUDA.h
CUDA/Thread_Kernel_CUDA.o: /usr/local/cuda-10.0/include/nvrtc.h
CUDA/Thread_Kernel_CUDA.o: Internal/Processor_Internal.h
CUDA/Thread_Kernel_CUDA.o: ../Import/Includes/KmsLib/DebugLog.h
CUDA/Thread_Kernel_CUDA.o: ../Import/Includes/KmsLib/Exception.h
CUDA/Thread_Kernel_CUDA.o: ../Import/Includes/KmsLib/TextFile.h
CUDA/Thread_Kernel_CUDA.o: ../Includes/OpenNet/Kernel.h
CUDA/Thread_Kernel_CUDA.o: ../Includes/OpenNet/SourceCode.h
CUDA/Thread_Kernel_CUDA.o: ../Includes/OpenNet/StatisticsProvider.h
CUDA/Thread_Kernel_CUDA.o: ../Includes/OpenNet/Processor.h
CUDA/Thread_Kernel_CUDA.o: ../Includes/OpenNet/UserBuffer.h
CUDA/Thread_Kernel_CUDA.o: ../Includes/OpenNetK/Adapter_Types.h
CUDA/Thread_Kernel_CUDA.o: CUDA/Thread_Kernel_CUDA.h Thread_Kernel.h Thread.h
CUDA/Thread_Kernel_CUDA.o: ../Import/Includes/KmsLib/ThreadBase.h
CUDA/Thread_Kernel_CUDA.o: Internal/Adapter_Internal.h
CUDA/Thread_Kernel_CUDA.o: ../Import/Includes/KmsLib/DriverHandle.h
CUDA/Thread_Kernel_CUDA.o: ../Import/Includes/KmsLib/FileHandle.h
CUDA/Thread_Kernel_CUDA.o: ../Includes/OpenNet/Adapter.h
CUDA/Thread_Kernel_CUDA.o: ../Includes/OpenNet/System.h
CUDA/Thread_Kernel_CUDA.o: ../Includes/OpenNetK/Constants.h
CUDA/Thread_Kernel_CUDA.o: ../Includes/OpenNetK/PacketGenerator_Types.h
CUDA/Thread_Kernel_CUDA.o: ../Common/IoCtl.h
CUDA/Thread_Kernel_CUDA.o: ../Common/OpenNet/Adapter_Statistics.h
CUDA/Thread_Kernel_CUDA.o: Internal/Buffer_Internal.h
CUDA/Thread_Kernel_CUDA.o: Internal/Processor_Internal.h
CUDA/Thread_Kernel_CUDA.o: Internal/Processor_Internal.h CUDA/Thread_CUDA.h
CUDA/Thread_Kernel_CUDA.o: Internal/Adapter_Internal.h
CUDA/UserBuffer_CUDA.o: CUDA/CUW.h /usr/local/cuda-10.0/include/cuda.h
CUDA/UserBuffer_CUDA.o: CUDA/UserBuffer_CUDA.h Internal/UserBuffer_Internal.h
CUDA/UserBuffer_CUDA.o: ../Includes/OpenNet/UserBuffer.h
CUDA/UserBuffer_CUDA.o: ../Includes/OpenNet/Status.h
CUDA/UserBuffer_CUDA.o: ../Includes/OpenNet/OpenNet.h
EthernetAddress.o: Component.h ../Import/Includes/KmsBase.h
EthernetAddress.o: ../Import/Includes/SafeAPI.h
EthernetAddress.o: ../Import/Includes/WindowsToLinux.h
EthernetAddress.o: ../Common/OpenNet/EthernetAddress.h
EthernetAddress.o: ../Includes/OpenNet/OpenNet.h ../Includes/OpenNet/Status.h
EthernetAddress.o: ../Includes/OpenNetK/Adapter_Types.h
Event.o: Component.h ../Import/Includes/KmsBase.h
Event.o: ../Import/Includes/SafeAPI.h ../Import/Includes/WindowsToLinux.h
Event.o: Event.h
FolderFinder.o: Component.h ../Import/Includes/KmsBase.h
FolderFinder.o: ../Import/Includes/SafeAPI.h
FolderFinder.o: ../Import/Includes/WindowsToLinux.h FolderFinder.h
Function.o: Component.h ../Import/Includes/KmsBase.h
Function.o: ../Import/Includes/SafeAPI.h ../Import/Includes/WindowsToLinux.h
Function.o: ../Includes/OpenNet/Function.h ../Includes/OpenNet/SourceCode.h
Function.o: ../Includes/OpenNet/OpenNet.h ../Includes/OpenNet/Status.h
Function_Forward.o: Component.h ../Import/Includes/KmsBase.h
Function_Forward.o: ../Import/Includes/SafeAPI.h
Function_Forward.o: ../Import/Includes/WindowsToLinux.h
Function_Forward.o: ../Includes/OpenNet/Adapter.h
Function_Forward.o: ../Includes/OpenNet/Processor.h
Function_Forward.o: ../Includes/OpenNet/OpenNet.h
Function_Forward.o: ../Includes/OpenNet/Status.h
Function_Forward.o: ../Includes/OpenNet/StatisticsProvider.h
Function_Forward.o: ../Includes/OpenNetK/Adapter_Types.h
Function_Forward.o: ../Includes/OpenNet/Function_Forward.h
Function_Forward.o: ../Includes/OpenNet/Function.h
Function_Forward.o: ../Includes/OpenNet/SourceCode.h SourceCode_Forward.h
Internal/Adapter_Internal.o: Component.h ../Import/Includes/KmsBase.h
Internal/Adapter_Internal.o: ../Import/Includes/SafeAPI.h
Internal/Adapter_Internal.o: ../Import/Includes/WindowsToLinux.h
Internal/Adapter_Internal.o: ../Common/Constants.h EthernetAddress.h
Internal/Adapter_Internal.o: ../Includes/OpenNet/Status.h
Internal/Adapter_Internal.o: ../Includes/OpenNet/OpenNet.h
Internal/Adapter_Internal.o: ../Common/OpenNet/EthernetAddress.h
Internal/Adapter_Internal.o: ../Includes/OpenNetK/Adapter_Types.h
Internal/Adapter_Internal.o: FolderFinder.h Thread_Functions.h
Internal/Adapter_Internal.o: ../Includes/OpenNet/Function.h
Internal/Adapter_Internal.o: ../Includes/OpenNet/SourceCode.h
Internal/Adapter_Internal.o: Kernel_Functions.h ../Includes/OpenNet/Kernel.h
Internal/Adapter_Internal.o: ../Includes/OpenNet/StatisticsProvider.h
Internal/Adapter_Internal.o: Thread.h ../Import/Includes/KmsLib/ThreadBase.h
Internal/Adapter_Internal.o: Internal/Adapter_Internal.h
Internal/Adapter_Internal.o: ../Import/Includes/KmsLib/DebugLog.h
Internal/Adapter_Internal.o: ../Import/Includes/KmsLib/Exception.h
Internal/Adapter_Internal.o: ../Import/Includes/KmsLib/TextFile.h
Internal/Adapter_Internal.o: ../Import/Includes/KmsLib/DriverHandle.h
Internal/Adapter_Internal.o: ../Import/Includes/KmsLib/FileHandle.h
Internal/Adapter_Internal.o: ../Includes/OpenNet/Adapter.h
Internal/Adapter_Internal.o: ../Includes/OpenNet/Processor.h
Internal/Adapter_Internal.o: ../Includes/OpenNet/System.h
Internal/Adapter_Internal.o: ../Includes/OpenNetK/Constants.h
Internal/Adapter_Internal.o: ../Includes/OpenNetK/PacketGenerator_Types.h
Internal/Adapter_Internal.o: ../Common/IoCtl.h
Internal/Adapter_Internal.o: ../Common/OpenNet/Adapter_Statistics.h
Internal/Adapter_Internal.o: Internal/Buffer_Internal.h
Internal/Adapter_Internal.o: ../Includes/OpenNet/Buffer.h
Internal/Adapter_Internal.o: ../Includes/OpenNetK/Types.h
Internal/Adapter_Internal.o: Internal/Processor_Internal.h
Internal/Adapter_Internal.o: ../Includes/OpenNet/UserBuffer.h
Internal/Adapter_Internal.o: Internal/Processor_Internal.h Utils.h
Internal/Adapter_Internal.o: Internal/Adapter_Internal.h
Internal/Buffer_Internal.o: Component.h ../Import/Includes/KmsBase.h
Internal/Buffer_Internal.o: ../Import/Includes/SafeAPI.h
Internal/Buffer_Internal.o: ../Import/Includes/WindowsToLinux.h Utils.h
Internal/Buffer_Internal.o: ../Import/Includes/KmsLib/Exception.h
Internal/Buffer_Internal.o: ../Includes/OpenNet/Status.h
Internal/Buffer_Internal.o: ../Includes/OpenNet/OpenNet.h
Internal/Buffer_Internal.o: Internal/Buffer_Internal.h
Internal/Buffer_Internal.o: ../Includes/OpenNet/Buffer.h
Internal/Buffer_Internal.o: ../Includes/OpenNetK/Types.h
Internal/PacketGenerator_Internal.o: Component.h ../Import/Includes/KmsBase.h
Internal/PacketGenerator_Internal.o: ../Import/Includes/SafeAPI.h
Internal/PacketGenerator_Internal.o: ../Import/Includes/WindowsToLinux.h
Internal/PacketGenerator_Internal.o: ../Includes/OpenNet/Adapter.h
Internal/PacketGenerator_Internal.o: ../Includes/OpenNet/Processor.h
Internal/PacketGenerator_Internal.o: ../Includes/OpenNet/OpenNet.h
Internal/PacketGenerator_Internal.o: ../Includes/OpenNet/Status.h
Internal/PacketGenerator_Internal.o: ../Includes/OpenNet/StatisticsProvider.h
Internal/PacketGenerator_Internal.o: ../Includes/OpenNetK/Adapter_Types.h
Internal/PacketGenerator_Internal.o: ../Common/Constants.h Constants.h
Internal/PacketGenerator_Internal.o: Internal/Adapter_Internal.h
Internal/PacketGenerator_Internal.o: ../Import/Includes/KmsLib/DebugLog.h
Internal/PacketGenerator_Internal.o: ../Import/Includes/KmsLib/Exception.h
Internal/PacketGenerator_Internal.o: ../Import/Includes/KmsLib/TextFile.h
Internal/PacketGenerator_Internal.o: ../Import/Includes/KmsLib/DriverHandle.h
Internal/PacketGenerator_Internal.o: ../Import/Includes/KmsLib/FileHandle.h
Internal/PacketGenerator_Internal.o: ../Import/Includes/KmsLib/ThreadBase.h
Internal/PacketGenerator_Internal.o: ../Includes/OpenNet/System.h
Internal/PacketGenerator_Internal.o: ../Includes/OpenNetK/Constants.h
Internal/PacketGenerator_Internal.o: ../Includes/OpenNetK/PacketGenerator_Types.h
Internal/PacketGenerator_Internal.o: ../Common/IoCtl.h
Internal/PacketGenerator_Internal.o: ../Common/OpenNet/Adapter_Statistics.h
Internal/PacketGenerator_Internal.o: Internal/Buffer_Internal.h
Internal/PacketGenerator_Internal.o: ../Includes/OpenNet/Buffer.h
Internal/PacketGenerator_Internal.o: ../Includes/OpenNetK/Types.h
Internal/PacketGenerator_Internal.o: Internal/Processor_Internal.h
Internal/PacketGenerator_Internal.o: ../Includes/OpenNet/Kernel.h
Internal/PacketGenerator_Internal.o: ../Includes/OpenNet/SourceCode.h
Internal/PacketGenerator_Internal.o: ../Includes/OpenNet/UserBuffer.h
Internal/PacketGenerator_Internal.o: Internal/PacketGenerator_Internal.h
Internal/PacketGenerator_Internal.o: ../Includes/OpenNet/PacketGenerator.h
Internal/Processor_Internal.o: Component.h ../Import/Includes/KmsBase.h
Internal/Processor_Internal.o: ../Import/Includes/SafeAPI.h
Internal/Processor_Internal.o: ../Import/Includes/WindowsToLinux.h
Internal/Processor_Internal.o: ../Import/Includes/KmsLib/Exception.h
Internal/Processor_Internal.o: Thread_Functions.h
Internal/Processor_Internal.o: ../Includes/OpenNet/Function.h
Internal/Processor_Internal.o: ../Includes/OpenNet/SourceCode.h
Internal/Processor_Internal.o: ../Includes/OpenNet/OpenNet.h
Internal/Processor_Internal.o: ../Includes/OpenNet/Status.h
Internal/Processor_Internal.o: Kernel_Functions.h
Internal/Processor_Internal.o: ../Includes/OpenNet/Kernel.h
Internal/Processor_Internal.o: ../Includes/OpenNet/StatisticsProvider.h
Internal/Processor_Internal.o: Thread.h
Internal/Processor_Internal.o: ../Import/Includes/KmsLib/ThreadBase.h
Internal/Processor_Internal.o: Internal/Adapter_Internal.h
Internal/Processor_Internal.o: ../Import/Includes/KmsLib/DebugLog.h
Internal/Processor_Internal.o: ../Import/Includes/KmsLib/TextFile.h
Internal/Processor_Internal.o: ../Import/Includes/KmsLib/DriverHandle.h
Internal/Processor_Internal.o: ../Import/Includes/KmsLib/FileHandle.h
Internal/Processor_Internal.o: ../Includes/OpenNet/Adapter.h
Internal/Processor_Internal.o: ../Includes/OpenNet/Processor.h
Internal/Processor_Internal.o: ../Includes/OpenNetK/Adapter_Types.h
Internal/Processor_Internal.o: ../Includes/OpenNet/System.h
Internal/Processor_Internal.o: ../Includes/OpenNetK/Constants.h
Internal/Processor_Internal.o: ../Includes/OpenNetK/PacketGenerator_Types.h
Internal/Processor_Internal.o: ../Common/IoCtl.h
Internal/Processor_Internal.o: ../Common/OpenNet/Adapter_Statistics.h
Internal/Processor_Internal.o: Internal/Buffer_Internal.h
Internal/Processor_Internal.o: ../Includes/OpenNet/Buffer.h
Internal/Processor_Internal.o: ../Includes/OpenNetK/Types.h
Internal/Processor_Internal.o: Internal/Processor_Internal.h
Internal/Processor_Internal.o: ../Includes/OpenNet/UserBuffer.h
Internal/Processor_Internal.o: Internal/Processor_Internal.h
Internal/SetupTool_Internal.o: Component.h ../Import/Includes/KmsBase.h
Internal/SetupTool_Internal.o: ../Import/Includes/SafeAPI.h
Internal/SetupTool_Internal.o: ../Import/Includes/WindowsToLinux.h
Internal/SetupTool_Internal.o: ../Import/Includes/KmsLib/Exception.h
Internal/SetupTool_Internal.o: FolderFinder.h Utils.h
Internal/SetupTool_Internal.o: ../Includes/OpenNet/Status.h
Internal/SetupTool_Internal.o: ../Includes/OpenNet/OpenNet.h
Internal/SetupTool_Internal.o: Internal/SetupTool_Internal.h
Internal/SetupTool_Internal.o: ../Includes/OpenNet/SetupTool.h
Internal/System_Internal.o: Component.h ../Import/Includes/KmsBase.h
Internal/System_Internal.o: ../Import/Includes/SafeAPI.h
Internal/System_Internal.o: ../Import/Includes/WindowsToLinux.h
Internal/System_Internal.o: ../Import/Includes/KmsLib/DriverHandle.h
Internal/System_Internal.o: ../Import/Includes/KmsLib/FileHandle.h
Internal/System_Internal.o: ../Common/Constants.h Constants.h Thread.h
Internal/System_Internal.o: ../Import/Includes/KmsLib/ThreadBase.h
Internal/System_Internal.o: Internal/Adapter_Internal.h
Internal/System_Internal.o: ../Import/Includes/KmsLib/DebugLog.h
Internal/System_Internal.o: ../Import/Includes/KmsLib/Exception.h
Internal/System_Internal.o: ../Import/Includes/KmsLib/TextFile.h
Internal/System_Internal.o: ../Includes/OpenNet/Adapter.h
Internal/System_Internal.o: ../Includes/OpenNet/Processor.h
Internal/System_Internal.o: ../Includes/OpenNet/OpenNet.h
Internal/System_Internal.o: ../Includes/OpenNet/Status.h
Internal/System_Internal.o: ../Includes/OpenNet/StatisticsProvider.h
Internal/System_Internal.o: ../Includes/OpenNetK/Adapter_Types.h
Internal/System_Internal.o: ../Includes/OpenNet/System.h
Internal/System_Internal.o: ../Includes/OpenNetK/Constants.h
Internal/System_Internal.o: ../Includes/OpenNetK/PacketGenerator_Types.h
Internal/System_Internal.o: ../Common/IoCtl.h
Internal/System_Internal.o: ../Common/OpenNet/Adapter_Statistics.h
Internal/System_Internal.o: Internal/Buffer_Internal.h
Internal/System_Internal.o: ../Includes/OpenNet/Buffer.h
Internal/System_Internal.o: ../Includes/OpenNetK/Types.h
Internal/System_Internal.o: Internal/Processor_Internal.h
Internal/System_Internal.o: ../Includes/OpenNet/Kernel.h
Internal/System_Internal.o: ../Includes/OpenNet/SourceCode.h
Internal/System_Internal.o: ../Includes/OpenNet/UserBuffer.h
Internal/System_Internal.o: Internal/Processor_Internal.h Utils.h
Internal/System_Internal.o: Internal/Adapter_Internal.h
Internal/System_Internal.o: Internal/System_Internal.h
Internal/UserBuffer_Internal.o: Component.h ../Import/Includes/KmsBase.h
Internal/UserBuffer_Internal.o: ../Import/Includes/SafeAPI.h
Internal/UserBuffer_Internal.o: ../Import/Includes/WindowsToLinux.h Utils.h
Internal/UserBuffer_Internal.o: ../Import/Includes/KmsLib/Exception.h
Internal/UserBuffer_Internal.o: ../Includes/OpenNet/Status.h
Internal/UserBuffer_Internal.o: ../Includes/OpenNet/OpenNet.h
Internal/UserBuffer_Internal.o: Internal/UserBuffer_Internal.h
Internal/UserBuffer_Internal.o: ../Includes/OpenNet/UserBuffer.h
Kernel.o: Component.h ../Import/Includes/KmsBase.h
Kernel.o: ../Import/Includes/SafeAPI.h ../Import/Includes/WindowsToLinux.h
Kernel.o: ../Import/Includes/KmsLib/Exception.h
Kernel.o: ../Import/Includes/KmsLib/ValueVector.h
Kernel.o: ../Includes/OpenNet/Kernel.h ../Includes/OpenNet/SourceCode.h
Kernel.o: ../Includes/OpenNet/OpenNet.h ../Includes/OpenNet/Status.h
Kernel.o: ../Includes/OpenNet/StatisticsProvider.h
Kernel.o: ../Common/OpenNet/Kernel_Statistics.h Constants.h Event.h
Kernel.o: CUDA/UserBuffer_CUDA.h /usr/local/cuda-10.0/include/cuda.h
Kernel.o: Internal/UserBuffer_Internal.h ../Includes/OpenNet/UserBuffer.h
Kernel_Forward.o: Component.h ../Import/Includes/KmsBase.h
Kernel_Forward.o: ../Import/Includes/SafeAPI.h
Kernel_Forward.o: ../Import/Includes/WindowsToLinux.h
Kernel_Forward.o: ../Includes/OpenNet/Adapter.h
Kernel_Forward.o: ../Includes/OpenNet/Processor.h
Kernel_Forward.o: ../Includes/OpenNet/OpenNet.h ../Includes/OpenNet/Status.h
Kernel_Forward.o: ../Includes/OpenNet/StatisticsProvider.h
Kernel_Forward.o: ../Includes/OpenNetK/Adapter_Types.h
Kernel_Forward.o: ../Includes/OpenNet/Kernel_Forward.h
Kernel_Forward.o: ../Includes/OpenNet/Kernel.h
Kernel_Forward.o: ../Includes/OpenNet/SourceCode.h SourceCode_Forward.h
Kernel_Functions.o: Component.h ../Import/Includes/KmsBase.h
Kernel_Functions.o: ../Import/Includes/SafeAPI.h
Kernel_Functions.o: ../Import/Includes/WindowsToLinux.h
Kernel_Functions.o: ../Import/Includes/KmsLib/Exception.h Kernel_Functions.h
Kernel_Functions.o: ../Includes/OpenNet/Function.h
Kernel_Functions.o: ../Includes/OpenNet/SourceCode.h
Kernel_Functions.o: ../Includes/OpenNet/OpenNet.h
Kernel_Functions.o: ../Includes/OpenNet/Status.h ../Includes/OpenNet/Kernel.h
Kernel_Functions.o: ../Includes/OpenNet/StatisticsProvider.h
Linux/Adapter_Linux.o: Component.h ../Import/Includes/KmsBase.h
Linux/Adapter_Linux.o: ../Import/Includes/SafeAPI.h
Linux/Adapter_Linux.o: ../Import/Includes/WindowsToLinux.h
Linux/Adapter_Linux.o: ../Includes/OpenNet/Function.h
Linux/Adapter_Linux.o: ../Includes/OpenNet/SourceCode.h
Linux/Adapter_Linux.o: ../Includes/OpenNet/OpenNet.h
Linux/Adapter_Linux.o: ../Includes/OpenNet/Status.h ../Common/Constants.h
Linux/Adapter_Linux.o: ../Common/IoCtl.h Thread_Functions.h
Linux/Adapter_Linux.o: Kernel_Functions.h ../Includes/OpenNet/Kernel.h
Linux/Adapter_Linux.o: ../Includes/OpenNet/StatisticsProvider.h Thread.h
Linux/Adapter_Linux.o: ../Import/Includes/KmsLib/ThreadBase.h
Linux/Adapter_Linux.o: Internal/Adapter_Internal.h
Linux/Adapter_Linux.o: ../Import/Includes/KmsLib/DebugLog.h
Linux/Adapter_Linux.o: ../Import/Includes/KmsLib/Exception.h
Linux/Adapter_Linux.o: ../Import/Includes/KmsLib/TextFile.h
Linux/Adapter_Linux.o: ../Import/Includes/KmsLib/DriverHandle.h
Linux/Adapter_Linux.o: ../Import/Includes/KmsLib/FileHandle.h
Linux/Adapter_Linux.o: ../Includes/OpenNet/Adapter.h
Linux/Adapter_Linux.o: ../Includes/OpenNet/Processor.h
Linux/Adapter_Linux.o: ../Includes/OpenNetK/Adapter_Types.h
Linux/Adapter_Linux.o: ../Includes/OpenNet/System.h
Linux/Adapter_Linux.o: ../Includes/OpenNetK/Constants.h
Linux/Adapter_Linux.o: ../Includes/OpenNetK/PacketGenerator_Types.h
Linux/Adapter_Linux.o: ../Common/OpenNet/Adapter_Statistics.h
Linux/Adapter_Linux.o: Internal/Buffer_Internal.h
Linux/Adapter_Linux.o: ../Includes/OpenNet/Buffer.h
Linux/Adapter_Linux.o: ../Includes/OpenNetK/Types.h
Linux/Adapter_Linux.o: Internal/Processor_Internal.h
Linux/Adapter_Linux.o: ../Includes/OpenNet/UserBuffer.h
Linux/Adapter_Linux.o: Internal/Processor_Internal.h CUDA/CUW.h
Linux/Adapter_Linux.o: /usr/local/cuda-10.0/include/cuda.h
Linux/Adapter_Linux.o: CUDA/Thread_Kernel_CUDA.h Thread_Kernel.h
Linux/Adapter_Linux.o: CUDA/Thread_CUDA.h Internal/Adapter_Internal.h
Linux/Adapter_Linux.o: CUDA/Processor_CUDA.h
Linux/Adapter_Linux.o: /usr/local/cuda-10.0/include/nvrtc.h
Linux/Adapter_Linux.o: Internal/Processor_Internal.h Linux/Adapter_Linux.h
Linux/FolderFinder_Linux.o: Component.h ../Import/Includes/KmsBase.h
Linux/FolderFinder_Linux.o: ../Import/Includes/SafeAPI.h
Linux/FolderFinder_Linux.o: ../Import/Includes/WindowsToLinux.h
Linux/FolderFinder_Linux.o: ../Common/Version.h FolderFinder.h
Linux/SetupTool_Linux.o: Component.h ../Import/Includes/KmsBase.h
Linux/SetupTool_Linux.o: ../Import/Includes/SafeAPI.h
Linux/SetupTool_Linux.o: ../Import/Includes/WindowsToLinux.h SetupTool_Text.h
Linux/SetupTool_Linux.o: Linux/SetupTool_Linux.h
Linux/SetupTool_Linux.o: Internal/SetupTool_Internal.h
Linux/SetupTool_Linux.o: ../Includes/OpenNet/SetupTool.h
Linux/SetupTool_Linux.o: ../Includes/OpenNet/Status.h
Linux/SetupTool_Linux.o: ../Includes/OpenNet/OpenNet.h
PacketGenerator.o: Component.h ../Import/Includes/KmsBase.h
PacketGenerator.o: ../Import/Includes/SafeAPI.h
PacketGenerator.o: ../Import/Includes/WindowsToLinux.h
PacketGenerator.o: ../Import/Includes/KmsLib/ValueVector.h
PacketGenerator.o: Internal/PacketGenerator_Internal.h
PacketGenerator.o: ../Import/Includes/KmsLib/DebugLog.h
PacketGenerator.o: ../Import/Includes/KmsLib/Exception.h
PacketGenerator.o: ../Import/Includes/KmsLib/TextFile.h
PacketGenerator.o: ../Includes/OpenNet/PacketGenerator.h
PacketGenerator.o: ../Includes/OpenNet/OpenNet.h ../Includes/OpenNet/Status.h
PacketGenerator.o: ../Includes/OpenNetK/Adapter_Types.h
PacketGenerator.o: ../Includes/OpenNetK/PacketGenerator_Types.h
PacketGenerator.o: Internal/Adapter_Internal.h
PacketGenerator.o: ../Import/Includes/KmsLib/DriverHandle.h
PacketGenerator.o: ../Import/Includes/KmsLib/FileHandle.h
PacketGenerator.o: ../Import/Includes/KmsLib/ThreadBase.h
PacketGenerator.o: ../Includes/OpenNet/Adapter.h
PacketGenerator.o: ../Includes/OpenNet/Processor.h
PacketGenerator.o: ../Includes/OpenNet/StatisticsProvider.h
PacketGenerator.o: ../Includes/OpenNet/System.h
PacketGenerator.o: ../Includes/OpenNetK/Constants.h ../Common/IoCtl.h
PacketGenerator.o: ../Common/OpenNet/Adapter_Statistics.h
PacketGenerator.o: Internal/Buffer_Internal.h ../Includes/OpenNet/Buffer.h
PacketGenerator.o: ../Includes/OpenNetK/Types.h Internal/Processor_Internal.h
PacketGenerator.o: ../Includes/OpenNet/Kernel.h
PacketGenerator.o: ../Includes/OpenNet/SourceCode.h
PacketGenerator.o: ../Includes/OpenNet/UserBuffer.h
Processor.o: Component.h ../Import/Includes/KmsBase.h
Processor.o: ../Import/Includes/SafeAPI.h ../Import/Includes/WindowsToLinux.h
Processor.o: ../Includes/OpenNet/Processor.h ../Includes/OpenNet/OpenNet.h
Processor.o: ../Includes/OpenNet/Status.h
SetupTool.o: Component.h ../Import/Includes/KmsBase.h
SetupTool.o: ../Import/Includes/SafeAPI.h ../Import/Includes/WindowsToLinux.h
SetupTool.o: ../Includes/OpenNet/SetupTool.h ../Includes/OpenNet/Status.h
SetupTool.o: ../Includes/OpenNet/OpenNet.h Linux/SetupTool_Linux.h
SetupTool.o: Internal/SetupTool_Internal.h
SourceCode.o: Component.h ../Import/Includes/KmsBase.h
SourceCode.o: ../Import/Includes/SafeAPI.h
SourceCode.o: ../Import/Includes/WindowsToLinux.h
SourceCode.o: ../Includes/OpenNet/SourceCode.h ../Includes/OpenNet/OpenNet.h
SourceCode.o: ../Includes/OpenNet/Status.h
SourceCode_Forward.o: Component.h ../Import/Includes/KmsBase.h
SourceCode_Forward.o: ../Import/Includes/SafeAPI.h
SourceCode_Forward.o: ../Import/Includes/WindowsToLinux.h
SourceCode_Forward.o: ../Includes/OpenNet/SourceCode.h
SourceCode_Forward.o: ../Includes/OpenNet/OpenNet.h
SourceCode_Forward.o: ../Includes/OpenNet/Status.h SourceCode_Forward.h
SourceCode_Forward.o: ../Includes/OpenNet/Adapter.h
SourceCode_Forward.o: ../Includes/OpenNet/Processor.h
SourceCode_Forward.o: ../Includes/OpenNet/StatisticsProvider.h
SourceCode_Forward.o: ../Includes/OpenNetK/Adapter_Types.h
StatisticsProvider.o: Component.h ../Import/Includes/KmsBase.h
StatisticsProvider.o: ../Import/Includes/SafeAPI.h
StatisticsProvider.o: ../Import/Includes/WindowsToLinux.h
StatisticsProvider.o: ../Import/Includes/KmsLib/ValueVector.h
StatisticsProvider.o: ../Includes/OpenNet/StatisticsProvider.h
StatisticsProvider.o: ../Includes/OpenNet/OpenNet.h
StatisticsProvider.o: ../Includes/OpenNet/Status.h
Status.o: Component.h ../Import/Includes/KmsBase.h
Status.o: ../Import/Includes/SafeAPI.h ../Import/Includes/WindowsToLinux.h
Status.o: ../Includes/OpenNet/Status.h ../Includes/OpenNet/OpenNet.h
System.o: Component.h ../Import/Includes/KmsBase.h
System.o: ../Import/Includes/SafeAPI.h ../Import/Includes/WindowsToLinux.h
System.o: ../Import/Includes/KmsLib/Exception.h
System.o: ../Import/Includes/KmsLib/ValueVector.h
System.o: ../Includes/OpenNet/System.h ../Includes/OpenNet/OpenNet.h
System.o: ../Includes/OpenNet/Status.h ../Includes/OpenNetK/Adapter_Types.h
System.o: ../Common/OpenNet/System_Statistics.h Internal/Adapter_Internal.h
System.o: ../Import/Includes/KmsLib/DebugLog.h
System.o: ../Import/Includes/KmsLib/TextFile.h
System.o: ../Import/Includes/KmsLib/DriverHandle.h
System.o: ../Import/Includes/KmsLib/FileHandle.h
System.o: ../Import/Includes/KmsLib/ThreadBase.h
System.o: ../Includes/OpenNet/Adapter.h ../Includes/OpenNet/Processor.h
System.o: ../Includes/OpenNet/StatisticsProvider.h
System.o: ../Includes/OpenNetK/Constants.h
System.o: ../Includes/OpenNetK/PacketGenerator_Types.h ../Common/IoCtl.h
System.o: ../Common/OpenNet/Adapter_Statistics.h Internal/Buffer_Internal.h
System.o: ../Includes/OpenNet/Buffer.h ../Includes/OpenNetK/Types.h
System.o: Internal/Processor_Internal.h ../Includes/OpenNet/Kernel.h
System.o: ../Includes/OpenNet/SourceCode.h ../Includes/OpenNet/UserBuffer.h
System.o: CUDA/System_CUDA.h Internal/System_Internal.h
Thread.o: Component.h ../Import/Includes/KmsBase.h
Thread.o: ../Import/Includes/SafeAPI.h ../Import/Includes/WindowsToLinux.h
Thread.o: Event.h Thread.h ../Import/Includes/KmsLib/ThreadBase.h
Thread.o: Internal/Adapter_Internal.h ../Import/Includes/KmsLib/DebugLog.h
Thread.o: ../Import/Includes/KmsLib/Exception.h
Thread.o: ../Import/Includes/KmsLib/TextFile.h
Thread.o: ../Import/Includes/KmsLib/DriverHandle.h
Thread.o: ../Import/Includes/KmsLib/FileHandle.h
Thread.o: ../Includes/OpenNet/Adapter.h ../Includes/OpenNet/Processor.h
Thread.o: ../Includes/OpenNet/OpenNet.h ../Includes/OpenNet/Status.h
Thread.o: ../Includes/OpenNet/StatisticsProvider.h
Thread.o: ../Includes/OpenNetK/Adapter_Types.h ../Includes/OpenNet/System.h
Thread.o: ../Includes/OpenNetK/Constants.h
Thread.o: ../Includes/OpenNetK/PacketGenerator_Types.h ../Common/IoCtl.h
Thread.o: ../Common/OpenNet/Adapter_Statistics.h Internal/Buffer_Internal.h
Thread.o: ../Includes/OpenNet/Buffer.h ../Includes/OpenNetK/Types.h
Thread.o: Internal/Processor_Internal.h ../Includes/OpenNet/Kernel.h
Thread.o: ../Includes/OpenNet/SourceCode.h ../Includes/OpenNet/UserBuffer.h
Thread.o: Internal/Processor_Internal.h
Thread_Functions.o: Component.h ../Import/Includes/KmsBase.h
Thread_Functions.o: ../Import/Includes/SafeAPI.h
Thread_Functions.o: ../Import/Includes/WindowsToLinux.h Event.h
Thread_Functions.o: Thread_Functions.h ../Includes/OpenNet/Function.h
Thread_Functions.o: ../Includes/OpenNet/SourceCode.h
Thread_Functions.o: ../Includes/OpenNet/OpenNet.h
Thread_Functions.o: ../Includes/OpenNet/Status.h Kernel_Functions.h
Thread_Functions.o: ../Includes/OpenNet/Kernel.h
Thread_Functions.o: ../Includes/OpenNet/StatisticsProvider.h Thread.h
Thread_Functions.o: ../Import/Includes/KmsLib/ThreadBase.h
Thread_Functions.o: Internal/Adapter_Internal.h
Thread_Functions.o: ../Import/Includes/KmsLib/DebugLog.h
Thread_Functions.o: ../Import/Includes/KmsLib/Exception.h
Thread_Functions.o: ../Import/Includes/KmsLib/TextFile.h
Thread_Functions.o: ../Import/Includes/KmsLib/DriverHandle.h
Thread_Functions.o: ../Import/Includes/KmsLib/FileHandle.h
Thread_Functions.o: ../Includes/OpenNet/Adapter.h
Thread_Functions.o: ../Includes/OpenNet/Processor.h
Thread_Functions.o: ../Includes/OpenNetK/Adapter_Types.h
Thread_Functions.o: ../Includes/OpenNet/System.h
Thread_Functions.o: ../Includes/OpenNetK/Constants.h
Thread_Functions.o: ../Includes/OpenNetK/PacketGenerator_Types.h
Thread_Functions.o: ../Common/IoCtl.h ../Common/OpenNet/Adapter_Statistics.h
Thread_Functions.o: Internal/Buffer_Internal.h ../Includes/OpenNet/Buffer.h
Thread_Functions.o: ../Includes/OpenNetK/Types.h
Thread_Functions.o: Internal/Processor_Internal.h
Thread_Functions.o: ../Includes/OpenNet/UserBuffer.h
Thread_Functions.o: Internal/Processor_Internal.h
Thread_Kernel.o: Component.h ../Import/Includes/KmsBase.h
Thread_Kernel.o: ../Import/Includes/SafeAPI.h
Thread_Kernel.o: ../Import/Includes/WindowsToLinux.h Thread_Kernel.h Thread.h
Thread_Kernel.o: ../Import/Includes/KmsLib/ThreadBase.h
Thread_Kernel.o: Internal/Adapter_Internal.h
Thread_Kernel.o: ../Import/Includes/KmsLib/DebugLog.h
Thread_Kernel.o: ../Import/Includes/KmsLib/Exception.h
Thread_Kernel.o: ../Import/Includes/KmsLib/TextFile.h
Thread_Kernel.o: ../Import/Includes/KmsLib/DriverHandle.h
Thread_Kernel.o: ../Import/Includes/KmsLib/FileHandle.h
Thread_Kernel.o: ../Includes/OpenNet/Adapter.h
Thread_Kernel.o: ../Includes/OpenNet/Processor.h
Thread_Kernel.o: ../Includes/OpenNet/OpenNet.h ../Includes/OpenNet/Status.h
Thread_Kernel.o: ../Includes/OpenNet/StatisticsProvider.h
Thread_Kernel.o: ../Includes/OpenNetK/Adapter_Types.h
Thread_Kernel.o: ../Includes/OpenNet/System.h
Thread_Kernel.o: ../Includes/OpenNetK/Constants.h
Thread_Kernel.o: ../Includes/OpenNetK/PacketGenerator_Types.h
Thread_Kernel.o: ../Common/IoCtl.h ../Common/OpenNet/Adapter_Statistics.h
Thread_Kernel.o: Internal/Buffer_Internal.h ../Includes/OpenNet/Buffer.h
Thread_Kernel.o: ../Includes/OpenNetK/Types.h Internal/Processor_Internal.h
Thread_Kernel.o: ../Includes/OpenNet/Kernel.h
Thread_Kernel.o: ../Includes/OpenNet/SourceCode.h
Thread_Kernel.o: ../Includes/OpenNet/UserBuffer.h
Thread_Kernel.o: Internal/Processor_Internal.h
UserBuffer.o: Component.h ../Import/Includes/KmsBase.h
UserBuffer.o: ../Import/Includes/SafeAPI.h
UserBuffer.o: ../Import/Includes/WindowsToLinux.h
UserBuffer.o: ../Includes/OpenNet/UserBuffer.h ../Includes/OpenNet/Status.h
UserBuffer.o: ../Includes/OpenNet/OpenNet.h
Utils.o: Component.h ../Import/Includes/KmsBase.h
Utils.o: ../Import/Includes/SafeAPI.h ../Import/Includes/WindowsToLinux.h
Utils.o: Utils.h ../Import/Includes/KmsLib/Exception.h
Utils.o: ../Includes/OpenNet/Status.h ../Includes/OpenNet/OpenNet.h
VersionInfo.o: Component.h ../Import/Includes/KmsBase.h
VersionInfo.o: ../Import/Includes/SafeAPI.h
VersionInfo.o: ../Import/Includes/WindowsToLinux.h VersionInfo.h
VersionInfo.o: ../Includes/OpenNet/Status.h ../Includes/OpenNet/OpenNet.h
VersionInfo.o: ../Includes/OpenNetK/Adapter_Types.h
