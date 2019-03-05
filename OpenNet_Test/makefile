
# Author     KMS - Martin Dubois, ing.
# Copyright  (C) 2018-2019 KMS. All rights reserved.
# Product    OpenNet
# File       OpenNet_Test/makefile

include ../User_Release.mk

INCLUDES = -I ../Includes -I ../Import/Includes

LIBRARIES = ../Binaries/OpenNet.so ../Import/Libraries/KmsLib.a

OUTPUT = ../Binaries/OpenNet_Test

SOURCES =	Adapter.cpp         \
			Base.cpp            \
			EthernetAddress.cpp \
			Kernel.cpp          \
			Kernel_Forward.cpp  \
			Processor.cpp		\
			SourceCode.cpp		\
			Status.cpp          \
			System.cpp			\
			OpenNet_Test.cpp    \
            SetupA.cpp          \
			SetupC.cpp

# ===== Rules / Regles ======================================================

.cpp.o:
	g++ -c $(CFLAGS) -o $@ $(INCLUDES) $<

# ===== Macros ==============================================================

OBJECTS = $(SOURCES:.cpp=.o)

# ===== Targets / Cibles ====================================================

$(OUTPUT) : $(OBJECTS) $(LIBRARIES)
	g++ -pthread -o $@ $^ -L /usr/local/cuda/lib64 -L /usr/local/cuda/lib64/stubs -l cudart -l cuda -l nvrtc

depend:
	makedepend -Y $(INCLUDES) $(SOURCES)

# DO NOT DELETE

Adapter.o: ../Import/Includes/KmsTest.h ../Import/Includes/KmsBase.h
Adapter.o: ../Import/Includes/SafeAPI.h ../Import/Includes/WindowsToLinux.h
Adapter.o: ../Includes/OpenNet/Adapter.h ../Includes/OpenNet/Processor.h
Adapter.o: ../Includes/OpenNet/OpenNet.h ../Includes/OpenNet/Status.h
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
Base.o: ../Import/Includes/KmsBase.h ../Import/Includes/SafeAPI.h
Base.o: ../Import/Includes/WindowsToLinux.h Base.h
Base.o: ../Import/Includes/KmsLib/ValueVector.h ../Includes/OpenNet/System.h
Base.o: ../Includes/OpenNet/OpenNet.h ../Includes/OpenNet/Status.h
Base.o: ../Includes/OpenNetK/Hardware_Statistics.h
Base.o: ../Common/OpenNet/Adapter_Statistics.h
Base.o: ../Common/OpenNetK/Adapter_Statistics.h
EthernetAddress.o: ../Import/Includes/KmsTest.h ../Import/Includes/KmsBase.h
EthernetAddress.o: ../Import/Includes/SafeAPI.h
EthernetAddress.o: ../Import/Includes/WindowsToLinux.h
EthernetAddress.o: ../Common/OpenNet/EthernetAddress.h
EthernetAddress.o: ../Includes/OpenNet/OpenNet.h ../Includes/OpenNet/Status.h
EthernetAddress.o: ../Includes/OpenNetK/Adapter_Types.h
Kernel.o: ../Import/Includes/KmsTest.h ../Import/Includes/KmsBase.h
Kernel.o: ../Import/Includes/SafeAPI.h ../Import/Includes/WindowsToLinux.h
Kernel.o: ../Includes/OpenNet/Kernel.h ../Includes/OpenNet/SourceCode.h
Kernel.o: ../Includes/OpenNet/OpenNet.h ../Includes/OpenNet/Status.h
Kernel.o: ../Includes/OpenNet/StatisticsProvider.h
Kernel_Forward.o: ../Import/Includes/KmsTest.h ../Import/Includes/KmsBase.h
Kernel_Forward.o: ../Import/Includes/SafeAPI.h
Kernel_Forward.o: ../Import/Includes/WindowsToLinux.h
Kernel_Forward.o: ../Includes/OpenNet/Kernel_Forward.h
Kernel_Forward.o: ../Includes/OpenNet/Kernel.h
Kernel_Forward.o: ../Includes/OpenNet/SourceCode.h
Kernel_Forward.o: ../Includes/OpenNet/OpenNet.h ../Includes/OpenNet/Status.h
Kernel_Forward.o: ../Includes/OpenNet/StatisticsProvider.h
Processor.o: ../Import/Includes/KmsTest.h ../Import/Includes/KmsBase.h
Processor.o: ../Import/Includes/SafeAPI.h ../Import/Includes/WindowsToLinux.h
Processor.o: ../Includes/OpenNet/Processor.h ../Includes/OpenNet/OpenNet.h
Processor.o: ../Includes/OpenNet/Status.h
SourceCode.o: ../Import/Includes/KmsTest.h ../Import/Includes/KmsBase.h
SourceCode.o: ../Import/Includes/SafeAPI.h
SourceCode.o: ../Import/Includes/WindowsToLinux.h
SourceCode.o: ../Includes/OpenNet/SourceCode.h ../Includes/OpenNet/OpenNet.h
SourceCode.o: ../Includes/OpenNet/Status.h
Status.o: ../Import/Includes/KmsTest.h ../Import/Includes/KmsBase.h
Status.o: ../Import/Includes/SafeAPI.h ../Import/Includes/WindowsToLinux.h
Status.o: ../Includes/OpenNet/Status.h ../Includes/OpenNet/OpenNet.h
System.o: ../Import/Includes/KmsBase.h ../Import/Includes/SafeAPI.h
System.o: ../Import/Includes/WindowsToLinux.h
System.o: ../Import/Includes/KmsLib/ThreadBase.h ../Import/Includes/KmsTest.h
System.o: ../Includes/OpenNet/System.h ../Includes/OpenNet/OpenNet.h
System.o: ../Includes/OpenNet/Status.h Base.h
System.o: ../Import/Includes/KmsLib/ValueVector.h
System.o: ../Includes/OpenNetK/Hardware_Statistics.h
System.o: ../Common/OpenNet/Adapter_Statistics.h
System.o: ../Common/OpenNetK/Adapter_Statistics.h SetupA.h
System.o: ../Includes/OpenNet/Adapter.h ../Includes/OpenNet/Processor.h
System.o: ../Includes/OpenNet/StatisticsProvider.h
System.o: ../Includes/OpenNetK/Adapter_Types.h
System.o: ../Includes/OpenNet/Kernel_Forward.h ../Includes/OpenNet/Kernel.h
System.o: ../Includes/OpenNet/SourceCode.h SetupC.h
OpenNet_Test.o: ../Import/Includes/KmsTest.h ../Import/Includes/KmsBase.h
OpenNet_Test.o: ../Import/Includes/SafeAPI.h
OpenNet_Test.o: ../Import/Includes/WindowsToLinux.h
SetupA.o: ../Import/Includes/KmsBase.h ../Import/Includes/SafeAPI.h
SetupA.o: ../Import/Includes/WindowsToLinux.h
SetupA.o: ../Import/Includes/KmsLib/ThreadBase.h SetupA.h
SetupA.o: ../Import/Includes/KmsLib/ValueVector.h
SetupA.o: ../Includes/OpenNet/Adapter.h ../Includes/OpenNet/Processor.h
SetupA.o: ../Includes/OpenNet/OpenNet.h ../Includes/OpenNet/Status.h
SetupA.o: ../Includes/OpenNet/StatisticsProvider.h
SetupA.o: ../Includes/OpenNetK/Adapter_Types.h
SetupA.o: ../Includes/OpenNet/Kernel_Forward.h ../Includes/OpenNet/Kernel.h
SetupA.o: ../Includes/OpenNet/SourceCode.h ../Includes/OpenNet/System.h
SetupA.o: Base.h ../Includes/OpenNetK/Hardware_Statistics.h
SetupA.o: ../Common/OpenNet/Adapter_Statistics.h
SetupA.o: ../Common/OpenNetK/Adapter_Statistics.h
SetupC.o: ../Import/Includes/KmsBase.h ../Import/Includes/SafeAPI.h
SetupC.o: ../Import/Includes/WindowsToLinux.h
SetupC.o: ../Import/Includes/KmsLib/ThreadBase.h SetupC.h
SetupC.o: ../Import/Includes/KmsLib/ValueVector.h
SetupC.o: ../Includes/OpenNet/Adapter.h ../Includes/OpenNet/Processor.h
SetupC.o: ../Includes/OpenNet/OpenNet.h ../Includes/OpenNet/Status.h
SetupC.o: ../Includes/OpenNet/StatisticsProvider.h
SetupC.o: ../Includes/OpenNetK/Adapter_Types.h
SetupC.o: ../Includes/OpenNet/Kernel_Forward.h ../Includes/OpenNet/Kernel.h
SetupC.o: ../Includes/OpenNet/SourceCode.h ../Includes/OpenNet/System.h
SetupC.o: Base.h ../Includes/OpenNetK/Hardware_Statistics.h
SetupC.o: ../Common/OpenNet/Adapter_Statistics.h
SetupC.o: ../Common/OpenNetK/Adapter_Statistics.h
