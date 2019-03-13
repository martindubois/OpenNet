
# Author     KMS - Martin Dubois, ing.
# Copyright  (C) 2018-2019 KMS. All rights reserved.
# Product    OpenNet
# File       OpenNet_Tool/makefile

include ../User.mk

INCLUDES = -I ../Includes $(INCLUDE_IMPORT)

LIBRARIES = ../Binaries/OpenNet.so ../Libraries/TestLib.a $(KMS_LIB_A)

OUTPUT = ../Binaries/OpenNet_Tool

SOURCES =	OpenNet_Tool.cpp \
            Test.cpp

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

OpenNet_Tool.o: ../Import/Includes/KmsBase.h ../Import/Includes/SafeAPI.h
OpenNet_Tool.o: ../Import/Includes/WindowsToLinux.h
OpenNet_Tool.o: ../Import/Includes/KmsLib/Exception.h
OpenNet_Tool.o: ../Import/Includes/KmsLib/ToolBase.h
OpenNet_Tool.o: ../Import/Includes/KmsTool.h ../Includes/OpenNet/Adapter.h
OpenNet_Tool.o: ../Includes/OpenNet/Processor.h ../Includes/OpenNet/OpenNet.h
OpenNet_Tool.o: ../Includes/OpenNet/Status.h
OpenNet_Tool.o: ../Includes/OpenNet/StatisticsProvider.h
OpenNet_Tool.o: ../Includes/OpenNetK/Adapter_Types.h
OpenNet_Tool.o: ../Includes/OpenNet/Kernel_Forward.h
OpenNet_Tool.o: ../Includes/OpenNet/Kernel.h ../Includes/OpenNet/SourceCode.h
OpenNet_Tool.o: ../Includes/OpenNet/System.h ../Includes/OpenNetK/Constants.h
OpenNet_Tool.o: ../Common/Version.h Test.h
Test.o: ../Import/Includes/KmsBase.h ../Import/Includes/SafeAPI.h
Test.o: ../Import/Includes/WindowsToLinux.h
Test.o: ../Import/Includes/KmsLib/ToolBase.h
Test.o: ../Import/Includes/KmsLib/Exception.h ../Common/TestLib/TestFactory.h
Test.o: ../Common/TestLib/Code.h Test.h
