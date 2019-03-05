
# Author     KMS - Martin Dubois, ing.
# Copyright  (C) 2018-2019 KMS. All rights reserved.
# Product    OpenNet
# File       OpenNet_Tool/makefile

include ../User_Release.mk

INCLUDES = -I ../Includes -I ../Import/Includes

LIBRARIES = ../Binaries/OpenNet.so ../Libraries/TestLib.a ../Import/Libraries/KmsLib.a

OUTPUT = ../Binaries/OpenNet_Tool

SOURCES =	OpenNet_Tool.cpp

# ===== Rules / Regles ======================================================

.cpp.o:
	g++ -c $(CFLAGS) -o $@ $(INCLUDES) $<

# ===== Macros ==============================================================

OBJECTS = $(SOURCES:.cpp=.o)

# ===== Targets / Cibles ====================================================

$(OUTPUT) : $(OBJECTS) $(LIBRARIES)
	g++ -pthread -o $@ $^  -L /usr/local/cuda/lib64 -L /usr/local/cuda/lib64/stubs -l cudart -l cuda -l nvrtc

depend:
	makedepend -Y $(INCLUDES) $(SOURCES)

# DO NOT DELETE

OpenNet_Tool.o: ../Import/Includes/KmsBase.h ../Import/Includes/SafeAPI.h
OpenNet_Tool.o: ../Import/Includes/WindowsToLinux.h
OpenNet_Tool.o: ../Import/Includes/KmsLib/ToolBase.h
OpenNet_Tool.o: ../Import/Includes/KmsLib/Exception.h
OpenNet_Tool.o: ../Import/Includes/KmsTool.h ../Includes/OpenNet/Adapter.h
OpenNet_Tool.o: ../Includes/OpenNet/Processor.h ../Includes/OpenNet/OpenNet.h
OpenNet_Tool.o: ../Includes/OpenNet/Status.h
OpenNet_Tool.o: ../Includes/OpenNet/StatisticsProvider.h
OpenNet_Tool.o: ../Includes/OpenNetK/Adapter_Types.h
OpenNet_Tool.o: ../Includes/OpenNet/Kernel_Forward.h
OpenNet_Tool.o: ../Includes/OpenNet/Kernel.h ../Includes/OpenNet/SourceCode.h
OpenNet_Tool.o: ../Includes/OpenNetK/Constants.h ../Common/Version.h
OpenNet_Tool.o: ../Common/TestLib/TestFactory.h Test.h
OpenNet_Tool.o: ../Common/TestLib/Tester.h
OpenNet_Tool.o: ../Import/Includes/KmsLib/ValueVector.h
OpenNet_Tool.o: ../Includes/OpenNet/Function_Forward.h
OpenNet_Tool.o: ../Includes/OpenNet/Function.h
OpenNet_Tool.o: ../Includes/OpenNet/PacketGenerator.h
OpenNet_Tool.o: ../Includes/OpenNet/System.h
OpenNet_Tool.o: ../Common/OpenNet/Adapter_Statistics.h
OpenNet_Tool.o: ../Common/OpenNetK/Adapter_Statistics.h
