
# Author     KMS - Martin Dubois, ing.
# Copyright  (C) 2018-2019 KMS. All rights reserved.
# Product    OpenNet
# File       OpenNet_Tool/makefile

include ../User.mk

INCLUDES = -I ../Includes $(INCLUDE_IMPORT)

LIBRARIES = ../Binaries/OpenNet.so ../Libraries/TestLib.a $(KMS_LIB_A)

OUTPUT = ../Binaries/OpenNet_Tool

SOURCES =	Adapter.cpp         \
			Globals.cpp         \
			OpenNet_Tool.cpp    \
			PacketGenerator.cpp \
			Processor.cpp       \
            Test.cpp            \
			Utils.cpp

# ===== Rules / Regles ======================================================

.cpp.o:
	g++ -c $(COMPILE_FLAGS) -o $@ $(INCLUDES) $<

# ===== Macros ==============================================================

OBJECTS = $(SOURCES:.cpp=.o)

# ===== Targets / Cibles ====================================================

$(OUTPUT) : $(OBJECTS) $(LIBRARIES)
	g++ -pthread -o $@ $^  -L $(CUDA_FOLDER)/lib64 -L $(CUDA_FOLDER)/lib64/stubs -l cudart -l cuda -l nvrtc

depend:
	makedepend -Y $(INCLUDES) $(SOURCES)

# DO NOT DELETE

Adapter.o: ../Import/Includes/KmsBase.h ../Import/Includes/SafeAPI.h
Adapter.o: ../Import/Includes/WindowsToLinux.h
Adapter.o: ../Import/Includes/KmsLib/ToolBase.h
Adapter.o: ../Import/Includes/KmsLib/Exception.h Globals.h
Adapter.o: ../Includes/OpenNet/Adapter.h ../Includes/OpenNet/Processor.h
Adapter.o: ../Includes/OpenNet/OpenNet.h ../Includes/OpenNet/Status.h
Adapter.o: ../Includes/OpenNet/StatisticsProvider.h
Adapter.o: ../Includes/OpenNetK/Adapter_Types.h ../Includes/OpenNet/System.h
Adapter.o: Utils.h Adapter.h
Globals.o: ../Import/Includes/KmsBase.h ../Import/Includes/SafeAPI.h
Globals.o: ../Import/Includes/WindowsToLinux.h Globals.h
Globals.o: ../Includes/OpenNet/Adapter.h ../Includes/OpenNet/Processor.h
Globals.o: ../Includes/OpenNet/OpenNet.h ../Includes/OpenNet/Status.h
Globals.o: ../Includes/OpenNet/StatisticsProvider.h
Globals.o: ../Includes/OpenNetK/Adapter_Types.h ../Includes/OpenNet/System.h
OpenNet_Tool.o: ../Import/Includes/KmsBase.h ../Import/Includes/SafeAPI.h
OpenNet_Tool.o: ../Import/Includes/WindowsToLinux.h
OpenNet_Tool.o: ../Import/Includes/KmsLib/Exception.h
OpenNet_Tool.o: ../Import/Includes/KmsLib/ToolBase.h
OpenNet_Tool.o: ../Import/Includes/KmsTool.h
OpenNet_Tool.o: ../Includes/OpenNet/Kernel_Forward.h
OpenNet_Tool.o: ../Includes/OpenNet/Kernel.h ../Includes/OpenNet/SourceCode.h
OpenNet_Tool.o: ../Includes/OpenNet/OpenNet.h ../Includes/OpenNet/Status.h
OpenNet_Tool.o: ../Includes/OpenNet/StatisticsProvider.h
OpenNet_Tool.o: ../Includes/OpenNetK/Constants.h ../Common/Version.h
OpenNet_Tool.o: Adapter.h Globals.h ../Includes/OpenNet/Adapter.h
OpenNet_Tool.o: ../Includes/OpenNet/Processor.h
OpenNet_Tool.o: ../Includes/OpenNetK/Adapter_Types.h
OpenNet_Tool.o: ../Includes/OpenNet/System.h PacketGenerator.h Processor.h
OpenNet_Tool.o: Test.h Utils.h
PacketGenerator.o: ../Import/Includes/KmsBase.h ../Import/Includes/SafeAPI.h
PacketGenerator.o: ../Import/Includes/WindowsToLinux.h
PacketGenerator.o: ../Import/Includes/KmsLib/ToolBase.h
PacketGenerator.o: ../Import/Includes/KmsLib/Exception.h
PacketGenerator.o: ../Includes/OpenNet/PacketGenerator.h
PacketGenerator.o: ../Includes/OpenNet/OpenNet.h ../Includes/OpenNet/Status.h
PacketGenerator.o: ../Includes/OpenNetK/Adapter_Types.h Globals.h
PacketGenerator.o: ../Includes/OpenNet/Adapter.h
PacketGenerator.o: ../Includes/OpenNet/Processor.h
PacketGenerator.o: ../Includes/OpenNet/StatisticsProvider.h
PacketGenerator.o: ../Includes/OpenNet/System.h Utils.h PacketGenerator.h
Processor.o: ../Import/Includes/KmsBase.h ../Import/Includes/SafeAPI.h
Processor.o: ../Import/Includes/WindowsToLinux.h
Processor.o: ../Import/Includes/KmsLib/ToolBase.h
Processor.o: ../Import/Includes/KmsLib/Exception.h Globals.h
Processor.o: ../Includes/OpenNet/Adapter.h ../Includes/OpenNet/Processor.h
Processor.o: ../Includes/OpenNet/OpenNet.h ../Includes/OpenNet/Status.h
Processor.o: ../Includes/OpenNet/StatisticsProvider.h
Processor.o: ../Includes/OpenNetK/Adapter_Types.h
Processor.o: ../Includes/OpenNet/System.h Utils.h Processor.h
Test.o: ../Import/Includes/KmsBase.h ../Import/Includes/SafeAPI.h
Test.o: ../Import/Includes/WindowsToLinux.h
Test.o: ../Import/Includes/KmsLib/ToolBase.h
Test.o: ../Import/Includes/KmsLib/Exception.h ../Common/TestLib/TestFactory.h
Test.o: ../Common/TestLib/Code.h Test.h
Utils.o: ../Import/Includes/KmsLib/ToolBase.h
Utils.o: ../Import/Includes/KmsLib/Exception.h Utils.h
Utils.o: ../Includes/OpenNet/Status.h ../Includes/OpenNet/OpenNet.h
