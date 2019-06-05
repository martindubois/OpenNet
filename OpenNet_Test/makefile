
# Author     KMS - Martin Dubois, ing.
# Copyright  (C) 2018-2019 KMS. All rights reserved.
# Product    OpenNet
# File       OpenNet_Test/makefile

include ../User.mk

INCLUDES = -I ../Includes $(INCLUDE_IMPORT)

LIBRARIES = $(KMS_LIB_A)

OUTPUT = ../Binaries/OpenNet_Test

SOURCES =	Adapter.cpp         \
			Base.cpp            \
			EthernetAddress.cpp \
			Kernel.cpp          \
			Kernel_Forward.cpp  \
			OpenNetK.cpp        \
			Processor.cpp		\
			SourceCode.cpp		\
			Status.cpp          \
			System.cpp			\
			OpenNet_Test.cpp    \
            SetupA.cpp          \
			SetupC.cpp          \
			SetupTool.cpp

# ===== Rules / Regles ======================================================

.cpp.o:
	g++ -c $(COMPILE_FLAGS) -o $@ $(INCLUDES) $<

# ===== Macros ==============================================================

OBJECTS = $(SOURCES:.cpp=.o)

# ===== Targets / Cibles ====================================================

$(OUTPUT) : $(OBJECTS) $(LIBRARIES)
	g++ -pthread -o $@ $^ -L ../Binaries -L $(CUDA_FOLDER)/lib64 -L $(CUDA_FOLDER)/lib64/stubs -lOpenNet -l cudart -l cuda -l nvrtc

depend:
	makedepend -Y $(INCLUDES) $(SOURCES)

# DO NOT DELETE

Adapter.o: Component.h ../Import/Includes/KmsBase.h
Adapter.o: ../Import/Includes/SafeAPI.h ../Import/Includes/WindowsToLinux.h
Adapter.o: ../Import/Includes/KmsTest.h ../Includes/OpenNet/Adapter.h
Adapter.o: ../Includes/OpenNet/Processor.h ../Includes/OpenNet/OpenNet.h
Adapter.o: ../Includes/OpenNet/Status.h
Adapter.o: ../Includes/OpenNet/StatisticsProvider.h
Adapter.o: ../Includes/OpenNetK/Adapter_Types.h ../Includes/OpenNet/System.h
Adapter.o: ../Common/Constants.h ../Common/IoCtl.h
Adapter.o: ../Common/OpenNet/EthernetAddress.h SetupA.h
Adapter.o: ../Import/Includes/KmsLib/ValueVector.h
Adapter.o: ../Includes/OpenNet/Kernel_Forward.h ../Includes/OpenNet/Kernel.h
Adapter.o: ../Includes/OpenNet/SourceCode.h Base.h
Adapter.o: ../Includes/OpenNetK/Hardware_Statistics.h
Adapter.o: ../Common/OpenNet/Adapter_Statistics.h
Adapter.o: ../Common/OpenNetK/Adapter_Statistics.h
Base.o: Component.h ../Import/Includes/KmsBase.h ../Import/Includes/SafeAPI.h
Base.o: ../Import/Includes/WindowsToLinux.h ../Import/Includes/KmsTest.h
Base.o: Base.h ../Import/Includes/KmsLib/ValueVector.h
Base.o: ../Includes/OpenNet/System.h ../Includes/OpenNet/OpenNet.h
Base.o: ../Includes/OpenNet/Status.h ../Includes/OpenNetK/Adapter_Types.h
Base.o: ../Includes/OpenNetK/Hardware_Statistics.h
Base.o: ../Common/OpenNet/Adapter_Statistics.h
Base.o: ../Common/OpenNetK/Adapter_Statistics.h
EthernetAddress.o: Component.h ../Import/Includes/KmsBase.h
EthernetAddress.o: ../Import/Includes/SafeAPI.h
EthernetAddress.o: ../Import/Includes/WindowsToLinux.h
EthernetAddress.o: ../Import/Includes/KmsTest.h
EthernetAddress.o: ../Common/OpenNet/EthernetAddress.h
EthernetAddress.o: ../Includes/OpenNet/OpenNet.h ../Includes/OpenNet/Status.h
EthernetAddress.o: ../Includes/OpenNetK/Adapter_Types.h
Kernel.o: Component.h ../Import/Includes/KmsBase.h
Kernel.o: ../Import/Includes/SafeAPI.h ../Import/Includes/WindowsToLinux.h
Kernel.o: ../Import/Includes/KmsTest.h ../Includes/OpenNet/Kernel.h
Kernel.o: ../Includes/OpenNet/SourceCode.h ../Includes/OpenNet/OpenNet.h
Kernel.o: ../Includes/OpenNet/Status.h
Kernel.o: ../Includes/OpenNet/StatisticsProvider.h
Kernel_Forward.o: Component.h ../Import/Includes/KmsBase.h
Kernel_Forward.o: ../Import/Includes/SafeAPI.h
Kernel_Forward.o: ../Import/Includes/WindowsToLinux.h
Kernel_Forward.o: ../Import/Includes/KmsTest.h
Kernel_Forward.o: ../Includes/OpenNet/Kernel_Forward.h
Kernel_Forward.o: ../Includes/OpenNet/Kernel.h
Kernel_Forward.o: ../Includes/OpenNet/SourceCode.h
Kernel_Forward.o: ../Includes/OpenNet/OpenNet.h ../Includes/OpenNet/Status.h
Kernel_Forward.o: ../Includes/OpenNet/StatisticsProvider.h
OpenNetK.o: Component.h ../Import/Includes/KmsBase.h
OpenNetK.o: ../Import/Includes/SafeAPI.h ../Import/Includes/WindowsToLinux.h
OpenNetK.o: ../Import/Includes/KmsTest.h ../Includes/OpenNetK/Types.h
OpenNetK.o: ../Includes/OpenNetK/ARP.h ../Includes/OpenNetK/ByteOrder.h
OpenNetK.o: ../Includes/OpenNetK/Ethernet.h ../Includes/OpenNetK/IPv4.h
OpenNetK.o: ../Includes/OpenNetK/IPv6.h ../Includes/OpenNetK/RegEx.h
OpenNetK.o: ../Includes/OpenNetK/TCP.h ../Includes/OpenNetK/UDP.h
Processor.o: Component.h ../Import/Includes/KmsBase.h
Processor.o: ../Import/Includes/SafeAPI.h ../Import/Includes/WindowsToLinux.h
Processor.o: ../Import/Includes/KmsTest.h ../Includes/OpenNet/Processor.h
Processor.o: ../Includes/OpenNet/OpenNet.h ../Includes/OpenNet/Status.h
SourceCode.o: Component.h ../Import/Includes/KmsBase.h
SourceCode.o: ../Import/Includes/SafeAPI.h
SourceCode.o: ../Import/Includes/WindowsToLinux.h
SourceCode.o: ../Import/Includes/KmsTest.h ../Includes/OpenNet/SourceCode.h
SourceCode.o: ../Includes/OpenNet/OpenNet.h ../Includes/OpenNet/Status.h
Status.o: Component.h ../Import/Includes/KmsBase.h
Status.o: ../Import/Includes/SafeAPI.h ../Import/Includes/WindowsToLinux.h
Status.o: ../Import/Includes/KmsTest.h ../Includes/OpenNet/Status.h
Status.o: ../Includes/OpenNet/OpenNet.h
System.o: Component.h ../Import/Includes/KmsBase.h
System.o: ../Import/Includes/SafeAPI.h ../Import/Includes/WindowsToLinux.h
System.o: ../Import/Includes/KmsTest.h ../Import/Includes/KmsLib/ThreadBase.h
System.o: ../Includes/OpenNet/System.h ../Includes/OpenNet/OpenNet.h
System.o: ../Includes/OpenNet/Status.h ../Includes/OpenNetK/Adapter_Types.h
System.o: Base.h ../Import/Includes/KmsLib/ValueVector.h
System.o: ../Includes/OpenNetK/Hardware_Statistics.h
System.o: ../Common/OpenNet/Adapter_Statistics.h
System.o: ../Common/OpenNetK/Adapter_Statistics.h SetupA.h
System.o: ../Includes/OpenNet/Adapter.h ../Includes/OpenNet/Processor.h
System.o: ../Includes/OpenNet/StatisticsProvider.h
System.o: ../Includes/OpenNet/Kernel_Forward.h ../Includes/OpenNet/Kernel.h
System.o: ../Includes/OpenNet/SourceCode.h SetupC.h
OpenNet_Test.o: Component.h ../Import/Includes/KmsBase.h
OpenNet_Test.o: ../Import/Includes/SafeAPI.h
OpenNet_Test.o: ../Import/Includes/WindowsToLinux.h
OpenNet_Test.o: ../Import/Includes/KmsTest.h
SetupA.o: Component.h ../Import/Includes/KmsBase.h
SetupA.o: ../Import/Includes/SafeAPI.h ../Import/Includes/WindowsToLinux.h
SetupA.o: ../Import/Includes/KmsTest.h ../Import/Includes/KmsLib/ThreadBase.h
SetupA.o: SetupA.h ../Import/Includes/KmsLib/ValueVector.h
SetupA.o: ../Includes/OpenNet/Adapter.h ../Includes/OpenNet/Processor.h
SetupA.o: ../Includes/OpenNet/OpenNet.h ../Includes/OpenNet/Status.h
SetupA.o: ../Includes/OpenNet/StatisticsProvider.h
SetupA.o: ../Includes/OpenNetK/Adapter_Types.h
SetupA.o: ../Includes/OpenNet/Kernel_Forward.h ../Includes/OpenNet/Kernel.h
SetupA.o: ../Includes/OpenNet/SourceCode.h ../Includes/OpenNet/System.h
SetupA.o: Base.h ../Includes/OpenNetK/Hardware_Statistics.h
SetupA.o: ../Common/OpenNet/Adapter_Statistics.h
SetupA.o: ../Common/OpenNetK/Adapter_Statistics.h
SetupC.o: Component.h ../Import/Includes/KmsBase.h
SetupC.o: ../Import/Includes/SafeAPI.h ../Import/Includes/WindowsToLinux.h
SetupC.o: ../Import/Includes/KmsTest.h ../Import/Includes/KmsLib/ThreadBase.h
SetupC.o: SetupC.h ../Import/Includes/KmsLib/ValueVector.h
SetupC.o: ../Includes/OpenNet/Adapter.h ../Includes/OpenNet/Processor.h
SetupC.o: ../Includes/OpenNet/OpenNet.h ../Includes/OpenNet/Status.h
SetupC.o: ../Includes/OpenNet/StatisticsProvider.h
SetupC.o: ../Includes/OpenNetK/Adapter_Types.h
SetupC.o: ../Includes/OpenNet/Kernel_Forward.h ../Includes/OpenNet/Kernel.h
SetupC.o: ../Includes/OpenNet/SourceCode.h ../Includes/OpenNet/System.h
SetupC.o: Base.h ../Includes/OpenNetK/Hardware_Statistics.h
SetupC.o: ../Common/OpenNet/Adapter_Statistics.h
SetupC.o: ../Common/OpenNetK/Adapter_Statistics.h
SetupTool.o: Component.h ../Import/Includes/KmsBase.h
SetupTool.o: ../Import/Includes/SafeAPI.h ../Import/Includes/WindowsToLinux.h
SetupTool.o: ../Import/Includes/KmsTest.h ../Includes/OpenNet/SetupTool.h
SetupTool.o: ../Includes/OpenNet/Status.h ../Includes/OpenNet/OpenNet.h
