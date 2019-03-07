
# Author     KMS - Martin Dubois, ing.
# Copyright  (C) 2018-2019 KMS. All rights reserved.
# Product    OpenNet
# File       OpenNet/makefile

include ../User.mk

INCLUDES = -I ../Includes $(INCLUDE_IMPORT) -I $(CUDA_FOLDER)/include

LIBRARIES = $(KMS_LIB_A)

OUTPUT = ../Binaries/OpenNet.so

SOURCES =	Adapter.cpp                  \
			Adapter_Internal.cpp		 \
			Adapter_Linux.cpp            \
			Buffer_Data.cpp              \
			Buffer_Data_CUDA.cpp         \
			CUDAW.cpp                    \
			CUW.cpp                      \
			Event.cpp                    \
			Event_CUDA.cpp               \
			Function.cpp                 \
			Function_Forward.cpp         \
			Kernel.cpp                   \
			Kernel_Forward.cpp           \
			Kernel_Functions.cpp		 \
			EthernetAddress.cpp          \
			NVRTCW.cpp                   \
			PacketGenerator.cpp          \
			PacketGenerator_Internal.cpp \
			Processor.cpp                \
			Processor_CUDA.cpp           \
			Processor_Internal.cpp       \
			SourceCode.cpp			     \
			SourceCode_Forward.cpp       \
			StatisticsProvider.cpp       \
			Status.cpp                   \
			System.cpp                   \
			System_CUDA.cpp              \
			System_Internal.cpp          \
			Thread.cpp                   \
			Thread_CUDA.cpp              \
			Thread_Functions.cpp		 \
			Thread_Functions_CUDA.cpp    \
			Thread_Kernel.cpp            \
			Thread_Kernel_CUDA.cpp       \
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

Adapter.o: ../Import/Includes/KmsBase.h ../Import/Includes/SafeAPI.h
Adapter.o: ../Import/Includes/WindowsToLinux.h
Adapter.o: ../Import/Includes/KmsLib/ValueVector.h
Adapter.o: ../Includes/OpenNet/Adapter.h ../Includes/OpenNet/Processor.h
Adapter.o: ../Includes/OpenNet/OpenNet.h ../Includes/OpenNet/Status.h
Adapter.o: ../Includes/OpenNet/StatisticsProvider.h
Adapter.o: ../Includes/OpenNetK/Adapter_Types.h
Adapter.o: ../Includes/OpenNetK/Constants.h ../Common/Constants.h
Adapter.o: ../Common/IoCtl.h EthernetAddress.h
Adapter.o: ../Common/OpenNet/EthernetAddress.h VersionInfo.h
Adapter_Internal.o: ../Import/Includes/KmsBase.h ../Import/Includes/SafeAPI.h
Adapter_Internal.o: ../Import/Includes/WindowsToLinux.h
Adapter_Internal.o: ../Import/Includes/KmsLib/Exception.h
Adapter_Internal.o: ../Common/Constants.h EthernetAddress.h
Adapter_Internal.o: ../Includes/OpenNet/Status.h
Adapter_Internal.o: ../Includes/OpenNet/OpenNet.h
Adapter_Internal.o: ../Common/OpenNet/EthernetAddress.h
Adapter_Internal.o: ../Includes/OpenNetK/Adapter_Types.h Thread_Functions.h
Adapter_Internal.o: ../Includes/OpenNet/Function.h
Adapter_Internal.o: ../Includes/OpenNet/SourceCode.h Kernel_Functions.h
Adapter_Internal.o: ../Includes/OpenNet/Kernel.h
Adapter_Internal.o: ../Includes/OpenNet/StatisticsProvider.h Thread.h
Adapter_Internal.o: ../Import/Includes/KmsLib/ThreadBase.h Adapter_Internal.h
Adapter_Internal.o: ../Import/Includes/KmsLib/DebugLog.h
Adapter_Internal.o: ../Import/Includes/KmsLib/TextFile.h
Adapter_Internal.o: ../Import/Includes/KmsLib/DriverHandle.h
Adapter_Internal.o: ../Import/Includes/KmsLib/FileHandle.h
Adapter_Internal.o: ../Includes/OpenNet/Adapter.h
Adapter_Internal.o: ../Includes/OpenNet/Processor.h
Adapter_Internal.o: ../Includes/OpenNet/System.h
Adapter_Internal.o: ../Includes/OpenNetK/Constants.h
Adapter_Internal.o: ../Includes/OpenNetK/PacketGenerator_Types.h
Adapter_Internal.o: ../Common/IoCtl.h ../Common/OpenNet/Adapter_Statistics.h
Adapter_Internal.o: Buffer_Data.h Processor_Internal.h
Adapter_Linux.o: ../Import/Includes/KmsBase.h ../Import/Includes/SafeAPI.h
Adapter_Linux.o: ../Import/Includes/WindowsToLinux.h
Adapter_Linux.o: ../Includes/OpenNet/Function.h
Adapter_Linux.o: ../Includes/OpenNet/SourceCode.h
Adapter_Linux.o: ../Includes/OpenNet/OpenNet.h ../Includes/OpenNet/Status.h
Adapter_Linux.o: ../Common/IoCtl.h CUW.h /usr/local/cuda-10.0/include/cuda.h
Adapter_Linux.o: Thread_Functions.h Kernel_Functions.h
Adapter_Linux.o: ../Includes/OpenNet/Kernel.h
Adapter_Linux.o: ../Includes/OpenNet/StatisticsProvider.h Thread.h
Adapter_Linux.o: ../Import/Includes/KmsLib/ThreadBase.h Adapter_Internal.h
Adapter_Linux.o: ../Import/Includes/KmsLib/DebugLog.h
Adapter_Linux.o: ../Import/Includes/KmsLib/Exception.h
Adapter_Linux.o: ../Import/Includes/KmsLib/TextFile.h
Adapter_Linux.o: ../Import/Includes/KmsLib/DriverHandle.h
Adapter_Linux.o: ../Import/Includes/KmsLib/FileHandle.h
Adapter_Linux.o: ../Includes/OpenNet/Adapter.h
Adapter_Linux.o: ../Includes/OpenNet/Processor.h
Adapter_Linux.o: ../Includes/OpenNetK/Adapter_Types.h
Adapter_Linux.o: ../Includes/OpenNet/System.h
Adapter_Linux.o: ../Includes/OpenNetK/Constants.h
Adapter_Linux.o: ../Includes/OpenNetK/PacketGenerator_Types.h
Adapter_Linux.o: ../Common/OpenNet/Adapter_Statistics.h Buffer_Data.h
Adapter_Linux.o: Processor_Internal.h Thread_Kernel_CUDA.h Thread_CUDA.h
Adapter_Linux.o: Processor_CUDA.h /usr/local/cuda-10.0/include/nvrtc.h
Adapter_Linux.o: Thread_Kernel.h Adapter_Linux.h
Buffer_Data.o: ../Import/Includes/KmsBase.h ../Import/Includes/SafeAPI.h
Buffer_Data.o: ../Import/Includes/WindowsToLinux.h Buffer_Data.h
Buffer_Data_CUDA.o: ../Import/Includes/KmsBase.h ../Import/Includes/SafeAPI.h
Buffer_Data_CUDA.o: ../Import/Includes/WindowsToLinux.h CUW.h
Buffer_Data_CUDA.o: /usr/local/cuda-10.0/include/cuda.h Buffer_Data_CUDA.h
Buffer_Data_CUDA.o: Event_CUDA.h Event.h Buffer_Data.h
CUDAW.o: ../Import/Includes/KmsLib/Exception.h CUDAW.h
CUDAW.o: /usr/local/cuda-10.0/include/cuda_runtime.h
CUDAW.o: /usr/local/cuda-10.0/include/crt/host_config.h
CUDAW.o: /usr/local/cuda-10.0/include/builtin_types.h
CUDAW.o: /usr/local/cuda-10.0/include/device_types.h
CUDAW.o: /usr/local/cuda-10.0/include/crt/host_defines.h
CUDAW.o: /usr/local/cuda-10.0/include/driver_types.h
CUDAW.o: /usr/local/cuda-10.0/include/vector_types.h
CUDAW.o: /usr/local/cuda-10.0/include/surface_types.h
CUDAW.o: /usr/local/cuda-10.0/include/texture_types.h
CUDAW.o: /usr/local/cuda-10.0/include/library_types.h
CUDAW.o: /usr/local/cuda-10.0/include/channel_descriptor.h
CUDAW.o: /usr/local/cuda-10.0/include/cuda_runtime_api.h
CUDAW.o: /usr/local/cuda-10.0/include/cuda_device_runtime_api.h
CUDAW.o: /usr/local/cuda-10.0/include/driver_functions.h
CUDAW.o: /usr/local/cuda-10.0/include/vector_functions.h
CUDAW.o: /usr/local/cuda-10.0/include/vector_functions.hpp
CUW.o: ../Import/Includes/KmsLib/Exception.h CUW.h
CUW.o: /usr/local/cuda-10.0/include/cuda.h
Event.o: Event.h
Event_CUDA.o: CUW.h /usr/local/cuda-10.0/include/cuda.h Event_CUDA.h Event.h
Function.o: ../Import/Includes/KmsBase.h ../Import/Includes/SafeAPI.h
Function.o: ../Import/Includes/WindowsToLinux.h
Function.o: ../Includes/OpenNet/Function.h ../Includes/OpenNet/SourceCode.h
Function.o: ../Includes/OpenNet/OpenNet.h ../Includes/OpenNet/Status.h
Function_Forward.o: ../Includes/OpenNet/Adapter.h
Function_Forward.o: ../Includes/OpenNet/Processor.h
Function_Forward.o: ../Includes/OpenNet/OpenNet.h
Function_Forward.o: ../Includes/OpenNet/Status.h
Function_Forward.o: ../Includes/OpenNet/StatisticsProvider.h
Function_Forward.o: ../Includes/OpenNetK/Adapter_Types.h
Function_Forward.o: ../Includes/OpenNet/Function_Forward.h
Function_Forward.o: ../Includes/OpenNet/Function.h
Function_Forward.o: ../Includes/OpenNet/SourceCode.h SourceCode_Forward.h
Kernel.o: ../Import/Includes/KmsBase.h ../Import/Includes/SafeAPI.h
Kernel.o: ../Import/Includes/WindowsToLinux.h
Kernel.o: ../Import/Includes/KmsLib/ValueVector.h
Kernel.o: ../Includes/OpenNet/Kernel.h ../Includes/OpenNet/SourceCode.h
Kernel.o: ../Includes/OpenNet/OpenNet.h ../Includes/OpenNet/Status.h
Kernel.o: ../Includes/OpenNet/StatisticsProvider.h
Kernel.o: ../Common/OpenNet/Kernel_Statistics.h Constants.h Event.h
Kernel_Forward.o: ../Import/Includes/KmsBase.h ../Import/Includes/SafeAPI.h
Kernel_Forward.o: ../Import/Includes/WindowsToLinux.h
Kernel_Forward.o: ../Includes/OpenNet/Adapter.h
Kernel_Forward.o: ../Includes/OpenNet/Processor.h
Kernel_Forward.o: ../Includes/OpenNet/OpenNet.h ../Includes/OpenNet/Status.h
Kernel_Forward.o: ../Includes/OpenNet/StatisticsProvider.h
Kernel_Forward.o: ../Includes/OpenNetK/Adapter_Types.h
Kernel_Forward.o: ../Includes/OpenNet/Kernel_Forward.h
Kernel_Forward.o: ../Includes/OpenNet/Kernel.h
Kernel_Forward.o: ../Includes/OpenNet/SourceCode.h SourceCode_Forward.h
Kernel_Functions.o: ../Import/Includes/KmsBase.h ../Import/Includes/SafeAPI.h
Kernel_Functions.o: ../Import/Includes/WindowsToLinux.h
Kernel_Functions.o: ../Import/Includes/KmsLib/Exception.h Kernel_Functions.h
Kernel_Functions.o: ../Includes/OpenNet/Function.h
Kernel_Functions.o: ../Includes/OpenNet/SourceCode.h
Kernel_Functions.o: ../Includes/OpenNet/OpenNet.h
Kernel_Functions.o: ../Includes/OpenNet/Status.h ../Includes/OpenNet/Kernel.h
Kernel_Functions.o: ../Includes/OpenNet/StatisticsProvider.h
EthernetAddress.o: ../Import/Includes/KmsBase.h ../Import/Includes/SafeAPI.h
EthernetAddress.o: ../Import/Includes/WindowsToLinux.h
EthernetAddress.o: ../Common/OpenNet/EthernetAddress.h
EthernetAddress.o: ../Includes/OpenNet/OpenNet.h ../Includes/OpenNet/Status.h
EthernetAddress.o: ../Includes/OpenNetK/Adapter_Types.h
NVRTCW.o: ../Import/Includes/KmsLib/Exception.h NVRTCW.h
NVRTCW.o: /usr/local/cuda-10.0/include/nvrtc.h
PacketGenerator.o: ../Import/Includes/KmsBase.h ../Import/Includes/SafeAPI.h
PacketGenerator.o: ../Import/Includes/WindowsToLinux.h
PacketGenerator.o: ../Import/Includes/KmsLib/ValueVector.h
PacketGenerator.o: PacketGenerator_Internal.h
PacketGenerator.o: ../Import/Includes/KmsLib/DebugLog.h
PacketGenerator.o: ../Import/Includes/KmsLib/Exception.h
PacketGenerator.o: ../Import/Includes/KmsLib/TextFile.h
PacketGenerator.o: ../Includes/OpenNet/PacketGenerator.h
PacketGenerator.o: ../Includes/OpenNet/OpenNet.h ../Includes/OpenNet/Status.h
PacketGenerator.o: ../Includes/OpenNetK/Adapter_Types.h
PacketGenerator.o: ../Includes/OpenNetK/PacketGenerator_Types.h
PacketGenerator.o: Adapter_Internal.h
PacketGenerator.o: ../Import/Includes/KmsLib/DriverHandle.h
PacketGenerator.o: ../Import/Includes/KmsLib/FileHandle.h
PacketGenerator.o: ../Includes/OpenNet/Adapter.h
PacketGenerator.o: ../Includes/OpenNet/Processor.h
PacketGenerator.o: ../Includes/OpenNet/StatisticsProvider.h
PacketGenerator.o: ../Includes/OpenNet/System.h
PacketGenerator.o: ../Includes/OpenNetK/Constants.h ../Common/IoCtl.h
PacketGenerator.o: ../Common/OpenNet/Adapter_Statistics.h Buffer_Data.h
PacketGenerator.o: Processor_Internal.h ../Includes/OpenNet/Kernel.h
PacketGenerator.o: ../Includes/OpenNet/SourceCode.h
PacketGenerator_Internal.o: ../Import/Includes/KmsBase.h
PacketGenerator_Internal.o: ../Import/Includes/SafeAPI.h
PacketGenerator_Internal.o: ../Import/Includes/WindowsToLinux.h
PacketGenerator_Internal.o: ../Includes/OpenNet/Adapter.h
PacketGenerator_Internal.o: ../Includes/OpenNet/Processor.h
PacketGenerator_Internal.o: ../Includes/OpenNet/OpenNet.h
PacketGenerator_Internal.o: ../Includes/OpenNet/Status.h
PacketGenerator_Internal.o: ../Includes/OpenNet/StatisticsProvider.h
PacketGenerator_Internal.o: ../Includes/OpenNetK/Adapter_Types.h
PacketGenerator_Internal.o: ../Common/Constants.h Adapter_Internal.h
PacketGenerator_Internal.o: ../Import/Includes/KmsLib/DebugLog.h
PacketGenerator_Internal.o: ../Import/Includes/KmsLib/Exception.h
PacketGenerator_Internal.o: ../Import/Includes/KmsLib/TextFile.h
PacketGenerator_Internal.o: ../Import/Includes/KmsLib/DriverHandle.h
PacketGenerator_Internal.o: ../Import/Includes/KmsLib/FileHandle.h
PacketGenerator_Internal.o: ../Includes/OpenNet/System.h
PacketGenerator_Internal.o: ../Includes/OpenNetK/Constants.h
PacketGenerator_Internal.o: ../Includes/OpenNetK/PacketGenerator_Types.h
PacketGenerator_Internal.o: ../Common/IoCtl.h
PacketGenerator_Internal.o: ../Common/OpenNet/Adapter_Statistics.h
PacketGenerator_Internal.o: Buffer_Data.h Processor_Internal.h
PacketGenerator_Internal.o: ../Includes/OpenNet/Kernel.h
PacketGenerator_Internal.o: ../Includes/OpenNet/SourceCode.h Constants.h
PacketGenerator_Internal.o: PacketGenerator_Internal.h
PacketGenerator_Internal.o: ../Includes/OpenNet/PacketGenerator.h
Processor.o: ../Import/Includes/KmsBase.h ../Import/Includes/SafeAPI.h
Processor.o: ../Import/Includes/WindowsToLinux.h
Processor.o: ../Includes/OpenNet/Processor.h ../Includes/OpenNet/OpenNet.h
Processor.o: ../Includes/OpenNet/Status.h
Processor_CUDA.o: ../Import/Includes/KmsBase.h ../Import/Includes/SafeAPI.h
Processor_CUDA.o: ../Import/Includes/WindowsToLinux.h
Processor_CUDA.o: ../Includes/OpenNetK/Types.h Buffer_Data_CUDA.h
Processor_CUDA.o: /usr/local/cuda-10.0/include/cuda.h Event_CUDA.h Event.h
Processor_CUDA.o: Buffer_Data.h CUW.h NVRTCW.h
Processor_CUDA.o: /usr/local/cuda-10.0/include/nvrtc.h
Processor_CUDA.o: Thread_Functions_CUDA.h Thread_CUDA.h
Processor_CUDA.o: ../Includes/OpenNet/Kernel.h
Processor_CUDA.o: ../Includes/OpenNet/SourceCode.h
Processor_CUDA.o: ../Includes/OpenNet/OpenNet.h ../Includes/OpenNet/Status.h
Processor_CUDA.o: ../Includes/OpenNet/StatisticsProvider.h Adapter_Internal.h
Processor_CUDA.o: ../Import/Includes/KmsLib/DebugLog.h
Processor_CUDA.o: ../Import/Includes/KmsLib/Exception.h
Processor_CUDA.o: ../Import/Includes/KmsLib/TextFile.h
Processor_CUDA.o: ../Import/Includes/KmsLib/DriverHandle.h
Processor_CUDA.o: ../Import/Includes/KmsLib/FileHandle.h
Processor_CUDA.o: ../Includes/OpenNet/Adapter.h
Processor_CUDA.o: ../Includes/OpenNet/Processor.h
Processor_CUDA.o: ../Includes/OpenNetK/Adapter_Types.h
Processor_CUDA.o: ../Includes/OpenNet/System.h
Processor_CUDA.o: ../Includes/OpenNetK/Constants.h
Processor_CUDA.o: ../Includes/OpenNetK/PacketGenerator_Types.h
Processor_CUDA.o: ../Common/IoCtl.h ../Common/OpenNet/Adapter_Statistics.h
Processor_CUDA.o: Processor_Internal.h Processor_CUDA.h Thread_Functions.h
Processor_CUDA.o: ../Includes/OpenNet/Function.h Kernel_Functions.h Thread.h
Processor_CUDA.o: ../Import/Includes/KmsLib/ThreadBase.h
Processor_Internal.o: ../Import/Includes/KmsBase.h
Processor_Internal.o: ../Import/Includes/SafeAPI.h
Processor_Internal.o: ../Import/Includes/WindowsToLinux.h
Processor_Internal.o: ../Import/Includes/KmsLib/Exception.h
Processor_Internal.o: Thread_Functions.h ../Includes/OpenNet/Function.h
Processor_Internal.o: ../Includes/OpenNet/SourceCode.h
Processor_Internal.o: ../Includes/OpenNet/OpenNet.h
Processor_Internal.o: ../Includes/OpenNet/Status.h Kernel_Functions.h
Processor_Internal.o: ../Includes/OpenNet/Kernel.h
Processor_Internal.o: ../Includes/OpenNet/StatisticsProvider.h Thread.h
Processor_Internal.o: ../Import/Includes/KmsLib/ThreadBase.h
Processor_Internal.o: Adapter_Internal.h ../Import/Includes/KmsLib/DebugLog.h
Processor_Internal.o: ../Import/Includes/KmsLib/TextFile.h
Processor_Internal.o: ../Import/Includes/KmsLib/DriverHandle.h
Processor_Internal.o: ../Import/Includes/KmsLib/FileHandle.h
Processor_Internal.o: ../Includes/OpenNet/Adapter.h
Processor_Internal.o: ../Includes/OpenNet/Processor.h
Processor_Internal.o: ../Includes/OpenNetK/Adapter_Types.h
Processor_Internal.o: ../Includes/OpenNet/System.h
Processor_Internal.o: ../Includes/OpenNetK/Constants.h
Processor_Internal.o: ../Includes/OpenNetK/PacketGenerator_Types.h
Processor_Internal.o: ../Common/IoCtl.h
Processor_Internal.o: ../Common/OpenNet/Adapter_Statistics.h Buffer_Data.h
Processor_Internal.o: Processor_Internal.h
SourceCode.o: ../Import/Includes/KmsBase.h ../Import/Includes/SafeAPI.h
SourceCode.o: ../Import/Includes/WindowsToLinux.h
SourceCode.o: ../Includes/OpenNet/SourceCode.h ../Includes/OpenNet/OpenNet.h
SourceCode.o: ../Includes/OpenNet/Status.h
SourceCode_Forward.o: ../Import/Includes/KmsBase.h
SourceCode_Forward.o: ../Import/Includes/SafeAPI.h
SourceCode_Forward.o: ../Import/Includes/WindowsToLinux.h
SourceCode_Forward.o: ../Includes/OpenNet/SourceCode.h
SourceCode_Forward.o: ../Includes/OpenNet/OpenNet.h
SourceCode_Forward.o: ../Includes/OpenNet/Status.h SourceCode_Forward.h
SourceCode_Forward.o: ../Includes/OpenNet/Adapter.h
SourceCode_Forward.o: ../Includes/OpenNet/Processor.h
SourceCode_Forward.o: ../Includes/OpenNet/StatisticsProvider.h
SourceCode_Forward.o: ../Includes/OpenNetK/Adapter_Types.h
StatisticsProvider.o: ../Import/Includes/KmsLib/ValueVector.h
StatisticsProvider.o: ../Includes/OpenNet/StatisticsProvider.h
StatisticsProvider.o: ../Includes/OpenNet/OpenNet.h
StatisticsProvider.o: ../Includes/OpenNet/Status.h
Status.o: ../Includes/OpenNet/Status.h ../Includes/OpenNet/OpenNet.h
System.o: ../Import/Includes/KmsBase.h ../Import/Includes/SafeAPI.h
System.o: ../Import/Includes/WindowsToLinux.h
System.o: ../Import/Includes/KmsLib/Exception.h
System.o: ../Import/Includes/KmsLib/ValueVector.h
System.o: ../Includes/OpenNet/System.h ../Includes/OpenNet/OpenNet.h
System.o: ../Includes/OpenNet/Status.h ../Common/OpenNet/System_Statistics.h
System.o: Adapter_Internal.h ../Import/Includes/KmsLib/DebugLog.h
System.o: ../Import/Includes/KmsLib/TextFile.h
System.o: ../Import/Includes/KmsLib/DriverHandle.h
System.o: ../Import/Includes/KmsLib/FileHandle.h
System.o: ../Includes/OpenNet/Adapter.h ../Includes/OpenNet/Processor.h
System.o: ../Includes/OpenNet/StatisticsProvider.h
System.o: ../Includes/OpenNetK/Adapter_Types.h
System.o: ../Includes/OpenNetK/Constants.h
System.o: ../Includes/OpenNetK/PacketGenerator_Types.h ../Common/IoCtl.h
System.o: ../Common/OpenNet/Adapter_Statistics.h Buffer_Data.h
System.o: Processor_Internal.h ../Includes/OpenNet/Kernel.h
System.o: ../Includes/OpenNet/SourceCode.h System_CUDA.h System_Internal.h
System_CUDA.o: ../Import/Includes/KmsBase.h ../Import/Includes/SafeAPI.h
System_CUDA.o: ../Import/Includes/WindowsToLinux.h ../Common/Constants.h
System_CUDA.o: Adapter_Linux.h /usr/local/cuda-10.0/include/cuda.h
System_CUDA.o: Adapter_Internal.h ../Import/Includes/KmsLib/DebugLog.h
System_CUDA.o: ../Import/Includes/KmsLib/Exception.h
System_CUDA.o: ../Import/Includes/KmsLib/TextFile.h
System_CUDA.o: ../Import/Includes/KmsLib/DriverHandle.h
System_CUDA.o: ../Import/Includes/KmsLib/FileHandle.h
System_CUDA.o: ../Includes/OpenNet/Adapter.h ../Includes/OpenNet/Processor.h
System_CUDA.o: ../Includes/OpenNet/OpenNet.h ../Includes/OpenNet/Status.h
System_CUDA.o: ../Includes/OpenNet/StatisticsProvider.h
System_CUDA.o: ../Includes/OpenNetK/Adapter_Types.h
System_CUDA.o: ../Includes/OpenNet/System.h ../Includes/OpenNetK/Constants.h
System_CUDA.o: ../Includes/OpenNetK/PacketGenerator_Types.h ../Common/IoCtl.h
System_CUDA.o: ../Common/OpenNet/Adapter_Statistics.h Buffer_Data.h
System_CUDA.o: Processor_Internal.h ../Includes/OpenNet/Kernel.h
System_CUDA.o: ../Includes/OpenNet/SourceCode.h CUW.h Processor_CUDA.h
System_CUDA.o: /usr/local/cuda-10.0/include/nvrtc.h System_CUDA.h
System_CUDA.o: System_Internal.h
System_Internal.o: ../Import/Includes/KmsBase.h ../Import/Includes/SafeAPI.h
System_Internal.o: ../Import/Includes/WindowsToLinux.h
System_Internal.o: ../Import/Includes/KmsLib/Exception.h
System_Internal.o: ../Import/Includes/KmsLib/DriverHandle.h
System_Internal.o: ../Import/Includes/KmsLib/FileHandle.h
System_Internal.o: ../Common/Constants.h Adapter_Internal.h
System_Internal.o: ../Import/Includes/KmsLib/DebugLog.h
System_Internal.o: ../Import/Includes/KmsLib/TextFile.h
System_Internal.o: ../Includes/OpenNet/Adapter.h
System_Internal.o: ../Includes/OpenNet/Processor.h
System_Internal.o: ../Includes/OpenNet/OpenNet.h ../Includes/OpenNet/Status.h
System_Internal.o: ../Includes/OpenNet/StatisticsProvider.h
System_Internal.o: ../Includes/OpenNetK/Adapter_Types.h
System_Internal.o: ../Includes/OpenNet/System.h
System_Internal.o: ../Includes/OpenNetK/Constants.h
System_Internal.o: ../Includes/OpenNetK/PacketGenerator_Types.h
System_Internal.o: ../Common/IoCtl.h ../Common/OpenNet/Adapter_Statistics.h
System_Internal.o: Buffer_Data.h Processor_Internal.h
System_Internal.o: ../Includes/OpenNet/Kernel.h
System_Internal.o: ../Includes/OpenNet/SourceCode.h Constants.h Thread.h
System_Internal.o: ../Import/Includes/KmsLib/ThreadBase.h System_Internal.h
Thread.o: ../Import/Includes/KmsBase.h ../Import/Includes/SafeAPI.h
Thread.o: ../Import/Includes/WindowsToLinux.h Event.h Thread.h
Thread.o: ../Import/Includes/KmsLib/ThreadBase.h Adapter_Internal.h
Thread.o: ../Import/Includes/KmsLib/DebugLog.h
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
Thread.o: ../Common/OpenNet/Adapter_Statistics.h Buffer_Data.h
Thread.o: Processor_Internal.h ../Includes/OpenNet/Kernel.h
Thread.o: ../Includes/OpenNet/SourceCode.h
Thread_CUDA.o: ../Import/Includes/KmsBase.h ../Import/Includes/SafeAPI.h
Thread_CUDA.o: ../Import/Includes/WindowsToLinux.h Adapter_Linux.h
Thread_CUDA.o: /usr/local/cuda-10.0/include/cuda.h Adapter_Internal.h
Thread_CUDA.o: ../Import/Includes/KmsLib/DebugLog.h
Thread_CUDA.o: ../Import/Includes/KmsLib/Exception.h
Thread_CUDA.o: ../Import/Includes/KmsLib/TextFile.h
Thread_CUDA.o: ../Import/Includes/KmsLib/DriverHandle.h
Thread_CUDA.o: ../Import/Includes/KmsLib/FileHandle.h
Thread_CUDA.o: ../Includes/OpenNet/Adapter.h ../Includes/OpenNet/Processor.h
Thread_CUDA.o: ../Includes/OpenNet/OpenNet.h ../Includes/OpenNet/Status.h
Thread_CUDA.o: ../Includes/OpenNet/StatisticsProvider.h
Thread_CUDA.o: ../Includes/OpenNetK/Adapter_Types.h
Thread_CUDA.o: ../Includes/OpenNet/System.h ../Includes/OpenNetK/Constants.h
Thread_CUDA.o: ../Includes/OpenNetK/PacketGenerator_Types.h ../Common/IoCtl.h
Thread_CUDA.o: ../Common/OpenNet/Adapter_Statistics.h Buffer_Data.h
Thread_CUDA.o: Processor_Internal.h ../Includes/OpenNet/Kernel.h
Thread_CUDA.o: ../Includes/OpenNet/SourceCode.h CUW.h Event_CUDA.h Event.h
Thread_CUDA.o: Thread_CUDA.h Processor_CUDA.h
Thread_CUDA.o: /usr/local/cuda-10.0/include/nvrtc.h
Thread_Functions.o: ../Import/Includes/KmsBase.h ../Import/Includes/SafeAPI.h
Thread_Functions.o: ../Import/Includes/WindowsToLinux.h Event.h
Thread_Functions.o: Thread_Functions.h ../Includes/OpenNet/Function.h
Thread_Functions.o: ../Includes/OpenNet/SourceCode.h
Thread_Functions.o: ../Includes/OpenNet/OpenNet.h
Thread_Functions.o: ../Includes/OpenNet/Status.h Kernel_Functions.h
Thread_Functions.o: ../Includes/OpenNet/Kernel.h
Thread_Functions.o: ../Includes/OpenNet/StatisticsProvider.h Thread.h
Thread_Functions.o: ../Import/Includes/KmsLib/ThreadBase.h Adapter_Internal.h
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
Thread_Functions.o: Buffer_Data.h Processor_Internal.h
Thread_Functions_CUDA.o: ../Import/Includes/KmsBase.h
Thread_Functions_CUDA.o: ../Import/Includes/SafeAPI.h
Thread_Functions_CUDA.o: ../Import/Includes/WindowsToLinux.h
Thread_Functions_CUDA.o: Buffer_Data_CUDA.h
Thread_Functions_CUDA.o: /usr/local/cuda-10.0/include/cuda.h Event_CUDA.h
Thread_Functions_CUDA.o: Event.h Buffer_Data.h CUW.h Thread_Functions_CUDA.h
Thread_Functions_CUDA.o: Thread_CUDA.h ../Includes/OpenNet/Kernel.h
Thread_Functions_CUDA.o: ../Includes/OpenNet/SourceCode.h
Thread_Functions_CUDA.o: ../Includes/OpenNet/OpenNet.h
Thread_Functions_CUDA.o: ../Includes/OpenNet/Status.h
Thread_Functions_CUDA.o: ../Includes/OpenNet/StatisticsProvider.h
Thread_Functions_CUDA.o: Adapter_Internal.h
Thread_Functions_CUDA.o: ../Import/Includes/KmsLib/DebugLog.h
Thread_Functions_CUDA.o: ../Import/Includes/KmsLib/Exception.h
Thread_Functions_CUDA.o: ../Import/Includes/KmsLib/TextFile.h
Thread_Functions_CUDA.o: ../Import/Includes/KmsLib/DriverHandle.h
Thread_Functions_CUDA.o: ../Import/Includes/KmsLib/FileHandle.h
Thread_Functions_CUDA.o: ../Includes/OpenNet/Adapter.h
Thread_Functions_CUDA.o: ../Includes/OpenNet/Processor.h
Thread_Functions_CUDA.o: ../Includes/OpenNetK/Adapter_Types.h
Thread_Functions_CUDA.o: ../Includes/OpenNet/System.h
Thread_Functions_CUDA.o: ../Includes/OpenNetK/Constants.h
Thread_Functions_CUDA.o: ../Includes/OpenNetK/PacketGenerator_Types.h
Thread_Functions_CUDA.o: ../Common/IoCtl.h
Thread_Functions_CUDA.o: ../Common/OpenNet/Adapter_Statistics.h
Thread_Functions_CUDA.o: Processor_Internal.h Processor_CUDA.h
Thread_Functions_CUDA.o: /usr/local/cuda-10.0/include/nvrtc.h
Thread_Functions_CUDA.o: Thread_Functions.h ../Includes/OpenNet/Function.h
Thread_Functions_CUDA.o: Kernel_Functions.h Thread.h
Thread_Functions_CUDA.o: ../Import/Includes/KmsLib/ThreadBase.h
Thread_Kernel.o: ../Import/Includes/KmsBase.h ../Import/Includes/SafeAPI.h
Thread_Kernel.o: ../Import/Includes/WindowsToLinux.h Thread_Kernel.h Thread.h
Thread_Kernel.o: ../Import/Includes/KmsLib/ThreadBase.h Adapter_Internal.h
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
Thread_Kernel.o: Buffer_Data.h Processor_Internal.h
Thread_Kernel.o: ../Includes/OpenNet/Kernel.h
Thread_Kernel.o: ../Includes/OpenNet/SourceCode.h
Thread_Kernel_CUDA.o: ../Import/Includes/KmsBase.h
Thread_Kernel_CUDA.o: ../Import/Includes/SafeAPI.h
Thread_Kernel_CUDA.o: ../Import/Includes/WindowsToLinux.h Adapter_Windows.h
Thread_Kernel_CUDA.o: /usr/local/cuda-10.0/include/CL/opencl.h
Thread_Kernel_CUDA.o: /usr/local/cuda-10.0/include/CL/cl.h
Thread_Kernel_CUDA.o: /usr/local/cuda-10.0/include/CL/cl_platform.h
Thread_Kernel_CUDA.o: /usr/local/cuda-10.0/include/CL/cl_gl.h
Thread_Kernel_CUDA.o: /usr/local/cuda-10.0/include/CL/cl_gl_ext.h
Thread_Kernel_CUDA.o: /usr/local/cuda-10.0/include/CL/cl_ext.h
Thread_Kernel_CUDA.o: Adapter_Internal.h ../Import/Includes/KmsLib/DebugLog.h
Thread_Kernel_CUDA.o: ../Import/Includes/KmsLib/Exception.h
Thread_Kernel_CUDA.o: ../Import/Includes/KmsLib/TextFile.h
Thread_Kernel_CUDA.o: ../Import/Includes/KmsLib/DriverHandle.h
Thread_Kernel_CUDA.o: ../Import/Includes/KmsLib/FileHandle.h
Thread_Kernel_CUDA.o: ../Includes/OpenNet/Adapter.h
Thread_Kernel_CUDA.o: ../Includes/OpenNet/Processor.h
Thread_Kernel_CUDA.o: ../Includes/OpenNet/OpenNet.h
Thread_Kernel_CUDA.o: ../Includes/OpenNet/Status.h
Thread_Kernel_CUDA.o: ../Includes/OpenNet/StatisticsProvider.h
Thread_Kernel_CUDA.o: ../Includes/OpenNetK/Adapter_Types.h
Thread_Kernel_CUDA.o: ../Includes/OpenNet/System.h
Thread_Kernel_CUDA.o: ../Includes/OpenNetK/Constants.h
Thread_Kernel_CUDA.o: ../Includes/OpenNetK/PacketGenerator_Types.h
Thread_Kernel_CUDA.o: ../Common/IoCtl.h
Thread_Kernel_CUDA.o: ../Common/OpenNet/Adapter_Statistics.h Buffer_Data.h
Thread_Kernel_CUDA.o: Processor_Internal.h ../Includes/OpenNet/Kernel.h
Thread_Kernel_CUDA.o: ../Includes/OpenNet/SourceCode.h Buffer_Data_CUDA.h
Thread_Kernel_CUDA.o: /usr/local/cuda-10.0/include/cuda.h Event_CUDA.h
Thread_Kernel_CUDA.o: Event.h CUDAW.h
Thread_Kernel_CUDA.o: /usr/local/cuda-10.0/include/cuda_runtime.h
Thread_Kernel_CUDA.o: /usr/local/cuda-10.0/include/crt/host_config.h
Thread_Kernel_CUDA.o: /usr/local/cuda-10.0/include/builtin_types.h
Thread_Kernel_CUDA.o: /usr/local/cuda-10.0/include/device_types.h
Thread_Kernel_CUDA.o: /usr/local/cuda-10.0/include/crt/host_defines.h
Thread_Kernel_CUDA.o: /usr/local/cuda-10.0/include/driver_types.h
Thread_Kernel_CUDA.o: /usr/local/cuda-10.0/include/vector_types.h
Thread_Kernel_CUDA.o: /usr/local/cuda-10.0/include/surface_types.h
Thread_Kernel_CUDA.o: /usr/local/cuda-10.0/include/texture_types.h
Thread_Kernel_CUDA.o: /usr/local/cuda-10.0/include/library_types.h
Thread_Kernel_CUDA.o: /usr/local/cuda-10.0/include/channel_descriptor.h
Thread_Kernel_CUDA.o: /usr/local/cuda-10.0/include/cuda_runtime_api.h
Thread_Kernel_CUDA.o: /usr/local/cuda-10.0/include/cuda_device_runtime_api.h
Thread_Kernel_CUDA.o: /usr/local/cuda-10.0/include/driver_functions.h
Thread_Kernel_CUDA.o: /usr/local/cuda-10.0/include/vector_functions.h
Thread_Kernel_CUDA.o: /usr/local/cuda-10.0/include/vector_functions.hpp
Thread_Kernel_CUDA.o: Processor_CUDA.h /usr/local/cuda-10.0/include/nvrtc.h
Thread_Kernel_CUDA.o: Thread_Kernel_CUDA.h Thread_CUDA.h Thread_Kernel.h
Thread_Kernel_CUDA.o: Thread.h ../Import/Includes/KmsLib/ThreadBase.h
VersionInfo.o: VersionInfo.h ../Includes/OpenNet/Status.h
VersionInfo.o: ../Includes/OpenNet/OpenNet.h
VersionInfo.o: ../Includes/OpenNetK/Adapter_Types.h
