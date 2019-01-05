
# Author     KMS - Martin Dubois, ing.
# Copyright  (C) 2018-2019 KMS. All rights reserved.
# Product    OpenNet
# File       OpenNet/makefile

CFLAGS = -fpic -g

INCLUDES = -I ../Includes -I ../Import/Includes

LIBRARIES = ../Import/Libraries/KmsLib.a

OUTPUT = ../Binaries/OpenNet.so

SOURCES =	Adapter.cpp                  \
			Adapter_Internal.cpp		 \
			Buffer_Data.cpp              \
			Function.cpp                 \
			Function_Forward.cpp         \
			Kernel.cpp                   \
			Kernel_Forward.cpp           \
			Kernel_Functions.cpp		 \
			EthernetAddress.cpp          \
			PacketGenerator.cpp          \
			PacketGenerator_Internal.cpp \
			Processor.cpp                \
			Processor_Internal.cpp       \
			SourceCode.cpp			     \
			SourceCode_Forward.cpp       \
			StatisticsProvider.cpp       \
			Status.cpp                   \
			System.cpp                   \
			System_Internal.cpp          \
			Thread.cpp                   \
			Thread_Functions.cpp		 \
			Thread_Kernel.cpp            \
			VersionInfo.cpp

# ===== Rules / Regles ======================================================

.cpp.o:
	g++ -c $(CFLAGS) -o $@ $(INCLUDES) $<

# ===== Macros ==============================================================

OBJECTS = $(SOURCES:.cpp=.o)

# ===== Targets ==============================================================

$(OUTPUT) : $(OBJECTS) $(LIBRARIES)
	g++ -shared -o $@ $^

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
