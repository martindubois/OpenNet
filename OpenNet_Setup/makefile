
# Author     KMS - Martin Dubois, ing.
# Copyright  (C) 2019 KMS. All rights reserved.
# Product    OpenNet
# File       OpenNet_Setup/makefile

include ../User.mk

INCLUDES = -I ../Includes $(INCLUDE_IMPORT)

OUTPUT = ../Binaries/OpenNet_Setup

SOURCES =	OpenNet_Setup.cpp \
			OSDep_Linux.cpp

# ===== Rules / Regles ======================================================

.cpp.o:
	g++ -c $(COMPILE_FLAGS) -o $@ $(INCLUDES) $<

# ===== Macros ==============================================================

OBJECTS = $(SOURCES:.cpp=.o)

# ===== Targets / Cibles ====================================================

$(OUTPUT) : $(OBJECTS)
	g++ -pthread -o $@ $^ -L ../Binaries -L $(CUDA_FOLDER)/lib64 -L $(CUDA_FOLDER)/lib64/stubs -l OpenNet -l cudart -l cuda -l nvrtc

depend:
	makedepend -Y $(INCLUDES) $(SOURCES)

# DO NOT DELETE

OpenNet_Setup.o: Component.h ../Import/Includes/KmsBase.h
OpenNet_Setup.o: ../Import/Includes/SafeAPI.h
OpenNet_Setup.o: ../Import/Includes/WindowsToLinux.h
OpenNet_Setup.o: ../Import/Includes/KmsTool.h ../Includes/OpenNet/SetupTool.h
OpenNet_Setup.o: ../Includes/OpenNet/Status.h ../Includes/OpenNet/OpenNet.h
OpenNet_Setup.o: ../Common/Version.h OSDep.h
OSDep_Linux.o: Component.h ../Import/Includes/KmsBase.h
OSDep_Linux.o: ../Import/Includes/SafeAPI.h
OSDep_Linux.o: ../Import/Includes/WindowsToLinux.h
